"""Tuned lens vs raw logit lens on the cached all-layer last-token states.

For each layer we fit a ridge map from that layer's last-token state to the
final-layer state on dev, apply it to test, then use the model's final norm and
LM head to read the yes/no margin. Raw logit lens applies the head directly.

Output: results/diagnostics/tuned_lens.json + results/figures/tuned_lens.png
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, load_model, yes_no_ids, load_processor

dev = torch.load(ROOT / "results/diagnostics/all_layers_dev.pt", map_location="cpu", weights_only=False)
test = torch.load(ROOT / "results/diagnostics/all_layers_test.pt", map_location="cpu", weights_only=False)
Hd, Ht = dev["H"].float(), test["H"].float()
yd = torch.tensor([m["label"] for m in dev["meta"]])
yt = torch.tensor([m["label"] for m in test["meta"]])
n_layers = dev["n_layers"]

proc = load_processor(splitting=False)
model = load_model(attn="sdpa")
yes, no = yes_no_ids(proc)
from peft import PeftModel
core = model.base_model.model if isinstance(model, PeftModel) else model
norm, head = core.model.text_model.norm, core.lm_head

def margins(H):
    with torch.no_grad():
        h = norm(H.to("cuda").to(model.dtype))
        lp = torch.log_softmax(head(h).float(), -1)
        return (torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)).cpu().numpy()

alpha = 10.0
out = {"alpha": alpha, "layers": [], "raw_auc": [], "tuned_auc": []}
Y = Hd[:, -1, :]
for i in range(n_layers):
    X = Hd[:, i, :]
    K = X @ X.T
    A = torch.linalg.solve(K + alpha * torch.eye(len(X)), Y)
    W = X.T @ A  # (d, d)
    pred = Ht[:, i, :] @ W
    raw = margins(Ht[:, i, :])
    tuned = margins(pred)
    ra, ta = roc_auc_score(yt, raw), roc_auc_score(yt, tuned)
    out["layers"].append(i); out["raw_auc"].append(float(ra)); out["tuned_auc"].append(float(ta))
    print(f"L{i:02d} raw {ra:.3f} tuned {ta:.3f}")

dst = ROOT / "results/diagnostics/tuned_lens.json"
dst.write_text(json.dumps(out, indent=1))
print(f"saved {dst}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(out["layers"], out["raw_auc"], "o-", label="raw logit lens")
ax.plot(out["layers"], out["tuned_auc"], "s-", label="tuned lens")
ax.axhline(0.5, color="k", lw=0.6, ls=":")
ax.set_xlabel("layer"); ax.set_ylabel("yes/no AUC")
ax.set_title("Raw logit lens vs tuned lens (last prompt token)")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "results/figures/tuned_lens.png", dpi=150)
print("saved figure")
