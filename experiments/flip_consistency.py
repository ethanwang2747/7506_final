"""Training-free flip-consistency readout for spatial yes/no questions.

For a left/right question, a relation-consistent model should give a higher
yes margin on the image and a lower yes margin on its horizontal flip. The
score `margin(original) - margin(flipped)` therefore cancels image-level and
yes/no biases while preserving the relation signal. No boxes and no training
are used.

For above/below questions a horizontal flip preserves truth, so the score is
the sum of the two margins instead.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT


def load(path):
    p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
    return p


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


def paired_bootstrap_delta(rows, score_a, score_b, n=1000, seed=0):
    pairs = sorted(set(r["pair_id"] for r in rows))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["pair_id"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(pairs, size=len(pairs), replace=True)
        sub = [r for p in sample for r in by[p]]
        y = np.array([r["label"] for r in sub])
        if len(set(y.tolist())) < 2:
            continue
        a = np.array([score_a(r) for r in sub])
        b = np.array([score_b(r) for r in sub])
        vals.append(roc_auc_score(y, a) - roc_auc_score(y, b))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def pair_accuracy(labels, scores, pair_ids):
    by = collections.defaultdict(list)
    for y, s, p in zip(labels, scores, pair_ids):
        by[p].append((float(y), float(s)))
    ok = []
    for vals in by.values():
        src = [s for y, s in vals if y == 1]
        tgt = [s for y, s in vals if y == 0]
        if src and tgt:
            ok.append(src[0] > tgt[0])
    return float(np.mean(ok)) if ok else float("nan")


def evaluate(rows, score_fn, bootstrap_n):
    labels = np.array([r["label"] for r in rows])
    scores = np.array([score_fn(r) for r in rows])
    pair_ids = [r["pair_id"] for r in rows]
    out = {
        "n_examples": int(len(labels)),
        "n_pairs": int(len(set(pair_ids))),
        "auc": float(roc_auc_score(labels, scores)),
        "auc_ci95": pair_bootstrap(labels, scores, pair_ids, n=bootstrap_n),
        "acc": float(((scores >= 0) == labels).mean()),
        "pair_acc": pair_accuracy(labels, scores, pair_ids),
    }
    per_group = {}
    for g in sorted(set(r["group"] for r in rows)):
        sub = [r for r in rows if r["group"] == g]
        y = np.array([r["label"] for r in sub])
        s = np.array([score_fn(r) for r in sub])
        pid = [r["pair_id"] for r in sub]
        per_group[g] = {
            "n": len(sub),
            "auc": float(roc_auc_score(y, s)),
            "auc_ci95": pair_bootstrap(y, s, pid, n=bootstrap_n),
            "acc": float(((s >= 0) == y).mean()),
            "pair_acc": pair_accuracy(y, s, pid),
        }
    out["per_group"] = per_group
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--flip-source", required=True)
    ap.add_argument("--flip-target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    pairs = load(args.pairs)
    src = load(args.flip_source)
    tgt = load(args.flip_target)

    rows = []
    missing = 0
    for e in pairs["examples"]:
        image_id, question = e["key"].split("|")[:2]
        k0 = f"{image_id}|{question}|0"
        k1 = f"{image_id}|{question}|1"
        flip_payload = src if e["role"] == "source" else tgt
        if k0 not in pairs["k2i"] or k1 not in flip_payload["k2i"]:
            missing += 1
            continue
        m0 = pairs["margin"][pairs["k2i"][k0]].item()
        m1 = flip_payload["margin"][flip_payload["k2i"][k1]].item()
        rows.append({
            "pair_id": e["pair_id"],
            "group": e["group"],
            "label": e["label"],
            "m_original": m0,
            "m_flipped": m1,
        })

    def score(r):
        if "above" in r["group"] or "below" in r["group"]:
            return r["m_original"] + r["m_flipped"]
        return r["m_original"] - r["m_flipped"]

    results = {
        "config": vars(args),
        "n_examples": len(rows),
        "n_missing": missing,
        "original": evaluate(rows, lambda r: r["m_original"], args.bootstrap),
        "flipped": evaluate(rows, lambda r: r["m_flipped"], args.bootstrap),
        "consistency": evaluate(rows, score, args.bootstrap),
        "sum": evaluate(rows, lambda r: r["m_original"] + r["m_flipped"], args.bootstrap),
    }
    results["consistency_minus_original"] = {
        "delta_auc": results["consistency"]["auc"] - results["original"]["auc"],
        "delta_ci95": paired_bootstrap_delta(
            rows, score, lambda r: r["m_original"], n=args.bootstrap),
        "delta_pair_acc": results["consistency"]["pair_acc"] - results["original"]["pair_acc"],
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out} (n={len(rows)}, missing={missing})")
    for name in ["original", "flipped", "consistency", "sum"]:
        r = results[name]
        print(f"  {name:12s} AUC={r['auc']:.3f} CI{r['auc_ci95']} "
              f"acc={r['acc']:.3f} pair_acc={r['pair_acc']:.3f}")
    print("  per-group consistency:")
    for g, r in results["consistency"]["per_group"].items():
        print(f"    {g:35s} AUC={r['auc']:.3f} CI{r['auc_ci95']} acc={r['acc']:.3f}")


if __name__ == "__main__":
    main()
