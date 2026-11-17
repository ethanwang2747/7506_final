"""Compare the model's yes/no margin with the geometry rule that defines the label.

For every test-core group we fit a logistic probe on the 12-dimensional box
description of the dev examples of that group, then apply it to the test
examples. This is an oracle geometry reader: it shows how much of the label is
determined by the two boxes alone. We then correlate its scores with the
model's own margin on the same examples.

Output:
    results/diagnostics/geometry_vs_model.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

PAYLOADS = {
    "train": "results/readout/features_dev_core.pt",
    "test": "results/readout/features_test_core.pt",
}


def load(path):
    p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
    return p


def main():
    data = {k: load(v) for k, v in PAYLOADS.items()}
    by_group = {}
    for e in data["train"]["examples"]:
        by_group.setdefault(e["group"], []).append(e)

    scores = np.full(len(data["test"]["examples"]), np.nan)
    for g, ex in by_group.items():
        idx = [i for i, e in enumerate(data["test"]["examples"]) if e["group"] == g]
        if len(idx) < 5 or len({e["label"] for e in ex}) < 2:
            continue
        X = np.array([e["position"] for e in ex], dtype=np.float64)
        y = np.array([e["label"] for e in ex])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        Xte = np.array([data["test"]["examples"][i]["position"] for i in idx], dtype=np.float64)
        scores[idx] = clf.decision_function(Xte)

    ok = ~np.isnan(scores)
    y = np.array([e["label"] for e in data["test"]["examples"]])[ok]
    margin = data["test"]["margin"].numpy()[ok]
    geo = scores[ok]

    out = {
        "n_examples": int(ok.sum()),
        "position_auc": float(roc_auc_score(y, geo)),
        "position_acc": float(((geo > 0).astype(int) == y).mean()),
        "model_auc": float(roc_auc_score(y, margin)),
        "model_acc": float(((margin > 0).astype(int) == y).mean()),
        "spearman": float(spearmanr(geo, margin).correlation),
        "sign_agreement": float(((geo > 0) == (margin > 0)).mean()),
        "note": "position probe is fitted per group on dev; boxes are oracle at test time",
    }
    dst = ROOT / "results/diagnostics/geometry_vs_model.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
