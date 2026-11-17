"""Matched reference readers for the per-token probe map.

The per-token map shows that no single visual token answers the relation.
This script adds two reference readers under exactly the same grouped CV,
features and question vector:

  mean_all   mean over all 81 visual tokens + question embedding
  oracle_pair  mean over target-box tokens, mean over reference-box tokens,
               their difference, and the question embedding

The contrast isolates selection: same features, same reader, same CV; the
only difference is whether the two object regions are handed to the reader.
Boxes are used at test time here, so oracle_pair is an upper bound.

Output:
    results/diagnostics/region_pool_reference.json
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT
from experiments.readout_experiment import REGION_PATHS, attach_regions, read_jsonl

STAGES = ["S1", "S6", "S12"]


def load_payload(path):
    p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    recs = {}
    for rp in REGION_PATHS:
        for r in read_jsonl(rp):
            recs[(r["pair_id"], "source")] = (r["source_target"], r["source_reference"])
            recs[(r["pair_id"], "target")] = (r["target_target"], r["target_reference"])
    attach_regions(p["examples"], recs)
    return p


def pooled(p, e, stage, tokens):
    x = p[stage][p["k2i"][e["key"]]].float()
    idx = list(tokens)
    if not idx:
        return torch.zeros(x.shape[-1])
    return x[idx].mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/readout/features_dev_strict.pt")
    ap.add_argument("--out", default="results/diagnostics/region_pool_reference.json")
    ap.add_argument("--n-comp", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    p = load_payload(args.features)
    p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
    meta = {m["key"]: m for m in p["meta"]}
    ex = p["examples"]
    n = len(ex)
    groups = np.array([str(meta[e["key"]]["image_id"]) for e in ex])
    folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(n), groups=groups))
    y = np.array([e["label"] for e in ex])

    feats = {}
    for name in ("mean_all", "oracle_pair"):
        rows = []
        for e in ex:
            parts = []
            for s in STAGES:
                if name == "mean_all":
                    parts.append(pooled(p, e, s, range(81)).numpy())
                else:
                    ft = pooled(p, e, s, e.get("region_target") or [])
                    fr = pooled(p, e, s, e.get("region_ref") or [])
                    parts.extend([ft.numpy(), fr.numpy(), (ft - fr).numpy()])
            parts.append(p["E_mean"][p["k2i"][e["key"]]].float().numpy())
            rows.append(np.concatenate(parts))
        feats[name] = np.stack(rows)

    out = {"features": args.features, "n_examples": n, "n_images": len(set(groups)),
           "cv": f"GroupKFold({args.folds}) by image_id", "readers": {}}
    for name, X in feats.items():
        scores = np.full(n, np.nan)
        for tr, te in folds:
            pca = PCA(n_components=args.n_comp, svd_solver="randomized", random_state=0)
            pca.fit(X[tr])
            clf = LogisticRegression(class_weight="balanced", max_iter=2000)
            clf.fit(pca.transform(X[tr]), y[tr])
            scores[te] = clf.decision_function(pca.transform(X[te]))
        auc = float(roc_auc_score(y, scores))
        acc = float(((scores > 0).astype(int) == y).mean())
        out["readers"][name] = {"auc": auc, "acc": acc, "dim": int(X.shape[1])}
        print(f"{name:12s} dim {X.shape[1]:5d} acc {acc:.3f} auc {auc:.3f}")

    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
