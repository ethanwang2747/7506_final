"""Oracle region readout probe on cached SmolVLM2 features.

Object boxes from GQA scene graphs are used to pool visual tokens for the
target and reference objects. This is a diagnostic upper bound, not a
deployable reader: boxes are available at test time in this probe only.

The probe is trained per relation group on dev pairs and evaluated on held-out
test pairs and on same-image horizontal flips. Position-only and the model's
own yes/no margin are reported on the same examples.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT
from experiments.readout_experiment import attach_regions, load_regions

STAGES = ["S1", "S6", "S12"]
PAYLOADS = {
    "train_strict": "results/readout/features_dev_strict.pt",
    "train_core": "results/readout/features_dev_core.pt",
    "train_flip": "results/readout/features_dev_strict_flip.pt",
    "test": "results/readout/features_test_core.pt",
    "test_flip": "results/readout/features_test_core_flip.pt",
    "test_strict": "results/readout/features_test_core_strict.pt",
    "test_strict_flip": "results/readout/features_test_core_strict_flip.pt",
}


def load(path):
    p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
    for e in p["examples"]:
        e["kind"] = e.get("kind", "pairs")
    return p


def region_feat(p, e, stages, use_pos=False, use_q=True):
    k = p["k2i"][e["key"]]
    tt = e.get("region_target") or e["region"]
    rt = e.get("region_ref") or e["region"]
    out = []
    for s in stages:
        x = p[s][k].float()
        if tt and rt:
            ft = x[tt].mean(dim=0)
            fr = x[rt].mean(dim=0)
            out.extend([ft.numpy(), fr.numpy(), (ft - fr).numpy()])
        else:
            out.append(x.mean(dim=0).numpy())
    if use_q:
        out.append(p["E_mean"][k].float().numpy())
    if use_pos:
        out.append(np.array(e["position"], dtype=np.float32))
    return np.concatenate(out)


def pair_bootstrap(labels, scores, pair_ids, n=1000, seed=0):
    pairs = sorted(set(pair_ids))
    by = collections.defaultdict(list)
    for i, p in enumerate(pair_ids):
        by[p].append(i)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(pairs, size=len(pairs), replace=True)
        idx = [i for p in sample for i in by[p]]
        y = labels[idx]
        if len(set(y.tolist())) < 2:
            continue
        vals.append(roc_auc_score(y, scores[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def fit_eval(train, test, feat_fn, pos_fn, bootstrap_n, residualize=False):
    Xtr = np.array([feat_fn(e) for e in train])
    ytr = np.array([e["label"] for e in train])
    Xte = np.array([feat_fn(e) for e in test])
    yte = np.array([e["label"] for e in test])
    if residualize:
        Ptr = np.array([pos_fn(e) for e in train])
        Pte = np.array([pos_fn(e) for e in test])
        lr = LinearRegression().fit(Ptr, Xtr)
        Xtr = Xtr - lr.predict(Ptr)
        Xte = Xte - lr.predict(Pte)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    scores = clf.predict_proba(sc.transform(Xte))[:, 1]
    return {
        "n_train": len(ytr),
        "n_test": len(yte),
        "auc": float(roc_auc_score(yte, scores)),
        "auc_ci95": pair_bootstrap(yte, scores, [e["pair_id"] for e in test], n=bootstrap_n),
        "acc": float(((scores >= 0.5) == yte).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/readout/region_probe.json")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    data = {k: load(v) for k, v in PAYLOADS.items()}
    regions = load_regions()
    all_examples = []
    for k, p in data.items():
        for e in p["examples"]:
            e["payload"] = k
            e["key_idx"] = p["k2i"][e["key"]]
            all_examples.append(e)

    for e in all_examples:
        if "region" not in e:
            attach_regions([e], regions)

    groups = sorted(set(e["group"] for e in data["test"]["examples"]))
    results = {"config": vars(args), "groups": {}, "aggregate": {}}

    for g in groups:
        train = [e for e in all_examples if e["payload"] in ("train_strict", "train_core")
                 and e["group"] == g and e["is_flip"] == 0]
        train_aug = train + [e for e in all_examples if e["payload"] == "train_flip"
                             and e["group"] == g and e["is_flip"] == 1]
        test = [e for e in data["test"]["examples"] if e["group"] == g]
        test_flip = [e for e in data["test_flip"]["examples"] if e["group"] == g]
        test_strict = [e for e in data["test_strict"]["examples"] if e["group"] == g]
        test_strict_flip = [e for e in data["test_strict_flip"]["examples"] if e["group"] == g]

        # Deduplicate train examples from strict and core payloads.
        seen = set()
        train_dedup = []
        for e in train:
            sig = (e["pair_id"], e["role"])
            if sig in seen:
                continue
            seen.add(sig)
            train_dedup.append(e)
        train = train_dedup
        seen = set()
        train_aug_dedup = []
        for e in train_aug:
            sig = (e["pair_id"], e["role"], e["is_flip"])
            if sig in seen:
                continue
            seen.add(sig)
            train_aug_dedup.append(e)
        train_aug = train_aug_dedup

        row = {}
        for name, tr, te in [
            ("test", train, test),
            ("test_aug", train_aug, test),
            ("test_strict", train, test_strict),
            ("flip", train, test_flip),
            ("flip_aug", train_aug, test_flip),
            ("flip_strict", train, test_strict_flip),
            ("flip_strict_aug", train_aug, test_strict_flip),
        ]:
            if not te or len(set(e["label"] for e in te)) < 2:
                continue
            feat_fn = lambda e: region_feat(data[e["payload"]], e, STAGES)
            pos_fn = lambda e: np.array(e["position"], dtype=np.float32)
            row[name] = fit_eval(tr, te, feat_fn, pos_fn, args.bootstrap)
            row[name]["residual"] = fit_eval(
                tr, te, feat_fn, pos_fn, args.bootstrap, residualize=True)
            row[name]["position"] = fit_eval(tr, te, pos_fn, pos_fn, args.bootstrap)
            scores = np.array([data[e["payload"]]["margin"][e["key_idx"]].item() for e in te])
            labels = np.array([e["label"] for e in te])
            row[name]["original"] = {
                "auc": float(roc_auc_score(labels, scores)),
                "auc_ci95": pair_bootstrap(labels, scores, [e["pair_id"] for e in te], n=args.bootstrap),
                "acc": float(((scores >= 0) == labels).mean()),
            }
        results["groups"][g] = row

    for split in ["test", "test_aug", "test_strict", "flip", "flip_aug", "flip_strict"]:
        rows = [v[split] for v in results["groups"].values() if split in v]
        if not rows:
            continue
        results["aggregate"][split] = {
            "n_groups": len(rows),
            "region_auc_mean": float(np.mean([r["auc"] for r in rows])),
            "region_auc_min": float(np.min([r["auc"] for r in rows])),
            "region_auc_max": float(np.max([r["auc"] for r in rows])),
            "position_auc_mean": float(np.mean([r["position"]["auc"] for r in rows])),
            "residual_auc_mean": float(np.mean([r["residual"]["auc"] for r in rows])),
            "residual_auc_min": float(np.min([r["residual"]["auc"] for r in rows])),
            "residual_auc_max": float(np.max([r["residual"]["auc"] for r in rows])),
            "original_auc_mean": float(np.mean([r["original"]["auc"] for r in rows])),
        }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for split in ["test", "test_aug", "test_strict", "flip", "flip_aug", "flip_strict"]:
        if split not in results["aggregate"]:
            continue
        a = results["aggregate"][split]
        print(f"== {split}: region={a['region_auc_mean']:.3f} "
              f"[{a['region_auc_min']:.3f},{a['region_auc_max']:.3f}] "
              f"resid={a['residual_auc_mean']:.3f} "
              f"[{a['residual_auc_min']:.3f},{a['residual_auc_max']:.3f}] "
              f"position={a['position_auc_mean']:.3f} original={a['original_auc_mean']:.3f}")
        for g, row in results["groups"].items():
            if split in row:
                r = row[split]
                print(f"  {g:35s} region={r['auc']:.3f} CI{r['auc_ci95']} "
                      f"resid={r['residual']['auc']:.3f} pos={r['position']['auc']:.3f} "
                      f"orig={r['original']['auc']:.3f}")


if __name__ == "__main__":
    main()
