"""Is the absolute side of each object readable where the answer forms?

Loads the cached last-token states from `extract_side_features.py` and trains
linear probes for four concepts at each layer:

  target_left / ref_left    object centre in the left half of the image
  target_top  / ref_top     object centre in the top half
  answer                    the pair label (relation true vs false)

A concept that is decodable while the answer is not means the model has the
ingredients but does not combine them. Position regression (R^2 for the centre
coordinates) is reported as a continuous version of the same test.

All probes use 5-fold GroupKFold by image id; every number is out-of-fold.

Output:
    results/diagnostics/side_probe.json
    results/figures/side_probe.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT


def load(path):
    return torch.load(ROOT / path, map_location="cpu", weights_only=False)


def oof_scores(X, y, groups, folds, seed=0, center_groups=None):
    scores = np.full(len(y), np.nan)
    for tr, te in folds:
        if center_groups is not None:
            mu = {}
            for g in np.unique(center_groups[tr]):
                mu[g] = X[tr][center_groups[tr] == g].mean(axis=0)
            Xtr = X[tr] - np.stack([mu[g] for g in center_groups[tr]])
            Xte = X[te] - np.stack([mu.get(g, X[tr].mean(axis=0)) for g in center_groups[te]])
        else:
            Xtr, Xte = X[tr], X[te]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(sc.transform(Xtr), y[tr])
        scores[te] = clf.decision_function(sc.transform(Xte))
    return scores


def oof_r2(X, y, folds, alpha=1000.0):
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred[te] = reg.predict(sc.transform(X[te]))
    sse = float(((pred - y) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/diagnostics/side_features_test_core.pt")
    ap.add_argument("--out", default="results/diagnostics/side_probe.json")
    ap.add_argument("--fig", default="results/figures/side_probe.png")
    args = ap.parse_args()

    p = load(args.features)
    meta = p["meta"]
    layers = p["layers"]
    groups = np.array([m["image_id"] for m in meta])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(meta)), groups=groups))

    targets = {
        "target_left": np.array([m["target"]["left"] for m in meta]),
        "ref_left": np.array([m["ref"]["left"] for m in meta]),
        "target_top": np.array([m["target"]["top"] for m in meta]),
        "ref_top": np.array([m["ref"]["top"] for m in meta]),
        "answer": np.array([m["label"] for m in meta], dtype=float),
    }
    cont = {
        "target_cx": np.array([m["target"]["cx"] for m in meta]),
        "ref_cx": np.array([m["ref"]["cx"] for m in meta]),
    }

    out = {"features": args.features, "n_examples": len(meta),
           "n_images": int(len(set(groups))), "cv": "GroupKFold(5) by image_id",
           "layers": layers, "probes": {}, "position_r2": {}}
    grp = np.array([m["group"] for m in meta])
    out["probes_centered"] = {}
    for name, y in targets.items():
        out["probes"][name] = {}
        out["probes_centered"][name] = {}
        for layer in layers:
            X = p[f"H{layer}"].float().numpy()
            s = oof_scores(X, y.astype(int), groups, folds)
            out["probes"][name][layer] = {
                "auc": float(roc_auc_score(y, s)),
                "acc": float(((s > 0).astype(int) == y).mean()),
            }
            sc = oof_scores(X, y.astype(int), groups, folds, center_groups=grp)
            out["probes_centered"][name][layer] = {
                "auc": float(roc_auc_score(y, sc)),
                "acc": float(((sc > 0).astype(int) == y).mean()),
            }
    for name, y in cont.items():
        out["position_r2"][name] = {}
        for layer in layers:
            X = p[f"H{layer}"].float().numpy()
            out["position_r2"][name][layer] = float(oof_r2(X, y, folds))

    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    for name in targets:
        print(name, {l: round(out["probes"][name][l]["auc"], 3) for l in layers})
    for name in targets:
        print(name + " (centered)",
              {l: round(out["probes_centered"][name][l]["auc"], 3) for l in layers})
    for name in cont:
        print(name, {l: round(out["position_r2"][name][l], 3) for l in layers})
    print(f"saved {dst}")
    make_figure(out, ROOT / args.fig)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = res["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for name, style in (("target_left", "o-"), ("ref_left", "s-"), ("target_top", "^-"),
                        ("ref_top", "v-"), ("answer", "k--")):
        ax.plot(layers, [res["probes"][name][l]["auc"] for l in layers], style,
                label=name)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("text layer (input)")
    ax.set_ylabel("AUC")
    ax.set_title("Concepts readable from the last prompt token")
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, style in (("target_cx", "o-"), ("ref_cx", "s-")):
        ax.plot(layers, [res["position_r2"][name][l] for l in layers], style, label=name)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("text layer (input)")
    ax.set_ylabel("cross-validated R^2")
    ax.set_title("Object centre x decodable from the last token")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
