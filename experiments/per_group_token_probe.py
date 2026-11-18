"""Per-group token probe: with the question fixed, can one visual token answer?

Train pairs come from GQA train images (600 per group, balanced), test pairs
from the held-out val core set. For each group we train one logistic probe per
9x9 cell and stage, then report held-out accuracy per cell and the model's own
accuracy on the same test examples.

This is the control missing from the earlier per-token map: there the probe had
no question context and mixed relation types, so chance-level accuracy was
expected. Here the group (and therefore the question) is fixed.

Output:
    results/diagnostics/per_group_token_probe.json
    results/figures/per_group_token_probe.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

GRID = 9
NCELL = GRID * GRID
STAGES = ["S1", "S6", "S12"]


def load(path):
    p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
    return p


def pca_fit(X, n_comp=128, seed=0):
    flat = X.reshape(-1, X.shape[-1])
    if flat.shape[0] > 20000:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.shape[0], 20000, replace=False)]
    p = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    p.fit(flat)
    return p


def residualize(Xtr, Ptr, Xte, Pte):
    A = np.concatenate([Ptr, np.ones((len(Ptr), 1), dtype=np.float32)], axis=1)
    B = np.linalg.lstsq(A, Xtr.reshape(len(Xtr), -1), rcond=None)[0]
    Ate = np.concatenate([Pte, np.ones((len(Pte), 1), dtype=np.float32)], axis=1)
    return (Xtr - (A @ B).reshape(Xtr.shape), Xte - (Ate @ B).reshape(Xte.shape))


def run_group(Xtr, Xte, ytr, yte, Ptr, Pte, n_comp, resid=False):
    if resid:
        Xtr, Xte = residualize(Xtr, Ptr, Xte, Pte)
    p = pca_fit(Xtr, n_comp)
    Ztr = p.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(len(Xtr), NCELL, -1)
    Zte = p.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(len(Xte), NCELL, -1)
    acc = np.full(NCELL, np.nan)
    auc = np.full(NCELL, np.nan)
    for c in range(NCELL):
        if len(np.unique(ytr)) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(Ztr[:, c], ytr)
        s = clf.decision_function(Zte[:, c])
        acc[c] = ((s > 0).astype(int) == yte).mean()
        auc[c] = roc_auc_score(yte, s)
    return acc.reshape(GRID, GRID), auc.reshape(GRID, GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results/readout/features_train_groups.pt")
    ap.add_argument("--test", default="results/readout/features_test_core.pt")
    ap.add_argument("--out", default="results/diagnostics/per_group_token_probe.json")
    ap.add_argument("--fig", default="results/figures/per_group_token_probe.png")
    ap.add_argument("--n-comp", type=int, default=128)
    args = ap.parse_args()

    train = load(args.train)
    test = load(args.test)
    groups = sorted({e["group"] for e in test["examples"]})
    print(f"{len(train['examples'])} train, {len(test['examples'])} test, {len(groups)} groups")

    out = {"train": args.train, "test": args.test, "n_comp": args.n_comp, "groups": {}}
    for g in groups:
        tr = [e for e in train["examples"] if e["group"] == g]
        te = [e for e in test["examples"] if e["group"] == g]
        if len(tr) < 50 or len(te) < 20:
            continue
        # per-group payloads keep the same tensors; index them by the subset
        tr_idx = [train["k2i"][e["key"]] for e in tr]
        te_idx = [test["k2i"][e["key"]] for e in te]
        tr_p = {s: train[s][tr_idx] for s in STAGES}
        te_p = {s: test[s][te_idx] for s in STAGES}
        entry = {"n_train": len(tr), "n_test": len(te)}
        ytr = np.array([e["label"] for e in tr], dtype=np.float32)
        yte = np.array([e["label"] for e in te], dtype=np.float32)
        Ptr = np.array([e["position"] for e in tr], dtype=np.float32)
        Pte = np.array([e["position"] for e in te], dtype=np.float32)
        for s in STAGES:
            Xtr = tr_p[s].float().numpy()
            Xte = te_p[s].float().numpy()
            acc_map, auc_map = run_group(Xtr, Xte, ytr, yte, Ptr, Pte, args.n_comp)
            entry[s] = {"acc_map": np.round(acc_map, 3).tolist(),
                        "acc_mean": float(np.nanmean(acc_map)),
                        "acc_peak": float(np.nanmax(acc_map)),
                        "auc_mean": float(np.nanmean(auc_map)),
                        "auc_peak": float(np.nanmax(auc_map))}
            print(f"  {g} {s}: acc mean {entry[s]['acc_mean']:.3f} "
                  f"peak {entry[s]['acc_peak']:.3f} auc peak {entry[s]['auc_peak']:.3f}")
        # model's own accuracy on this group
        m = test["margin"].numpy()[te_idx]
        y = np.array([e["label"] for e in te])
        entry["model_acc"] = float(((m > 0).astype(int) == y).mean())
        entry["model_auc"] = float(roc_auc_score(y, m))
        out["groups"][g] = entry
        print(f"  {g} model acc {entry['model_acc']:.3f} auc {entry['model_auc']:.3f}")

    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")
    make_figure(out, ROOT / args.fig)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(res["groups"])
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 3, figsize=(10, 2.6 * len(groups)))
    if len(groups) == 1:
        axes = axes[None, :]
    for r, g in enumerate(groups):
        for c, s in enumerate(STAGES):
            ax = axes[r, c]
            m = np.array(res["groups"][g][s]["acc_map"])
            im = ax.imshow(m, cmap="viridis", vmin=0.3, vmax=0.8)
            ax.set_title(f"{g.split('|')[0]} {res['groups'][g][s]['acc_peak']:.2f} {s}",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Per-group token probe accuracy (question fixed)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
