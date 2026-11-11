import argparse, json, sys
from pathlib import Path

for _k in list(sys.modules):
    if _k.startswith(("mpl_toolkits", "matplotlib")):
        del sys.modules[_k]
sys.path_importer_cache.clear()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import token_box_9

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors.pt"
STAGES = ["S0", "S1", "S6", "S12"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def capture_condition(model, inp, layers, connector_hook=None):
    cap = {}
    handles = []
    if connector_hook is not None:
        def combined(mod, args, out):
            y = connector_hook(mod, args, out)
            cap["S1"] = y.detach()
            return y
        handles.append(model.model.connector.register_forward_hook(combined))
    else:
        handles.append(model.model.connector.register_forward_hook(
            lambda m, i, o: cap.__setitem__("S1", o.detach())))
    handles.append(model.model.vision_model.post_layernorm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("S0", o.detach())))
    for i in layers:
        def pre(i):
            def f(mod, args, kwargs):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cap[f"S{i}"] = x.detach()
            return f
        handles.append(model.model.text_model.layers[i].register_forward_pre_hook(pre(i), with_kwargs=True))
    with torch.no_grad():
        model(**inp, use_cache=False)
    for h in handles:
        h.remove()
    return cap


def visual_matrix(cap, stage, v0):
    x = cap[stage]
    if stage == "S0":
        x = x[0].float().reshape(27, 27, -1)
        d = x.shape[-1]
        x = x.reshape(9, 3, 9, 3, d).mean(dim=(1, 3))
        return x.cpu().numpy()
    if stage == "S1":
        x = x[0].float()
    else:
        x = x[0, v0:v0 + 81, :].float()
    return x.reshape(9, 9, -1).cpu().numpy()


def cosine_map(a, b):
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=1).reshape(9, 9).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="woman|to the left of|man")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "results/figures/representation_3d.png"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{args.split}"]) if r["group"] == args.group]
    r = rows[0]
    images = load_images([r["source_image_id"], r["target_image_id"]])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    means = torch.load(MEANS, map_location="cpu")["means"]
    src_in = prepare(proc, r["question"], images[r["source_image_id"]])
    tgt_in = prepare(proc, r["question"], images[r["target_image_id"]])
    pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    v0 = int(pos[0])

    tgt_tokens = token_box_9(r["source_target"], 384, 384)

    src_cap = capture_condition(model, src_in, [6, 12])
    tgt_cap = capture_condition(model, tgt_in, [6, 12])

    def ablate_hook(mod, args, out):
        x = out.clone()
        x[:, tgt_tokens, :] = means["S1"].to(dtype=x.dtype, device=x.device)
        return x

    abl_cap = capture_condition(model, tgt_in, [6, 12], connector_hook=ablate_hook)

    def full_hook(mod, args, out):
        return src_cap["S1"]

    full_cap = capture_condition(model, tgt_in, [6, 12], connector_hook=full_hook)

    sim_st = {s: cosine_map(torch.from_numpy(visual_matrix(src_cap, s, v0)),
                             torch.from_numpy(visual_matrix(tgt_cap, s, v0))) for s in STAGES}
    sim_at = {s: cosine_map(torch.from_numpy(visual_matrix(abl_cap, s, v0)),
                            torch.from_numpy(visual_matrix(tgt_cap, s, v0))) for s in STAGES if s != "S0"}
    sim_fs = {s: cosine_map(torch.from_numpy(visual_matrix(full_cap, s, v0)),
                            torch.from_numpy(visual_matrix(src_cap, s, v0))) for s in STAGES if s != "S0"}

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    X, Y = np.meshgrid(np.arange(9), np.arange(9))
    for zi, s in enumerate(STAGES):
        Z = np.full_like(X, zi, dtype=float)
        C = 1.0 - sim_st[s]
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.inferno(C), rstride=1, cstride=1, shade=False, alpha=0.9)
    ax.set_zticks(range(len(STAGES)), STAGES)
    ax.set_xlabel("visual token col")
    ax.set_ylabel("visual token row")
    ax.set_zlabel("stage")
    ax.set_title("source vs target: 1 - cosine(token)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    tokens = np.arange(81)
    src_tok = src_cap["S1"][0].float().cpu().numpy()
    tgt_tok = tgt_cap["S1"][0].float().cpu().numpy()
    abl_tok = abl_cap["S1"][0].float().cpu().numpy()
    allx = np.concatenate([src_tok, tgt_tok, abl_tok])
    p = PCA(n_components=3).fit_transform(allx)
    a, b, c = p[:81], p[81:162], p[162:]
    ax2.scatter(a[:, 0], a[:, 1], a[:, 2], s=18, c="tab:blue", label="source")
    ax2.scatter(b[:, 0], b[:, 1], b[:, 2], s=18, c="tab:red", label="target")
    ax2.scatter(c[:, 0], c[:, 1], c[:, 2], s=18, c="tab:orange", label="target, target-region ablated")
    for i in tgt_tokens:
        ax2.scatter(*b[i], s=60, facecolors="none", edgecolors="black")
    ax2.legend(fontsize=7)
    ax2.set_title("S1 visual tokens, PCA")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(STAGES, [sim_st[s].mean() for s in STAGES], marker="o", label="source vs target")
    ax3.plot([s for s in STAGES if s != "S0"], [sim_at[s].mean() for s in STAGES if s != "S0"], marker="o", label="target vs ablated")
    ax3.plot([s for s in STAGES if s != "S0"], [sim_fs[s].mean() for s in STAGES if s != "S0"], marker="o", label="source vs patched")
    ax3.set_ylim(0, 1.02)
    ax3.set_ylabel("mean cosine similarity")
    ax3.set_title("stage-wise representation similarity")
    ax3.legend(fontsize=7)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"saved {out}")
    print("mean sim source-target", {s: float(sim_st[s].mean()) for s in STAGES})
    print("mean sim target-ablated", {s: float(sim_at[s].mean()) for s in sim_at})
    print("mean sim source-patched", {s: float(sim_fs[s].mean()) for s in sim_fs})


if __name__ == "__main__":
    main()
