"""Blank-image control for the fine-grained same-image pairs.

Replaces the image with a uniform gray canvas. The two questions in each pair
are then answered from the question text alone. Comparing this with the real
image separates visual evidence from language priors.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare


def bootstrap_acc(rows, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [float(np.mean([rows[i]["delta_blank"] > 0 for i in rng.integers(0, len(rows), len(rows))]))
            for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(rows):
    y, s = [], []
    for r in rows:
        y.extend([1, 0])
        s.extend([r["m_pos_blank"], r["m_neg_blank"]])
    return {
        "n_pairs": len(rows),
        "pair_acc": float(np.mean([r["delta_blank"] > 0 for r in rows])),
        "pair_acc_ci95": bootstrap_acc(rows),
        "auc": float(roc_auc_score(y, s)),
        "yes_rate_pos": float(np.mean([r["m_pos_blank"] > 0 for r in rows])),
        "yes_rate_neg": float(np.mean([r["m_neg_blank"] > 0 for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/finegrained/test.json"))
    ap.add_argument("--out", default=str(ROOT / "results/finegrained/test_blank.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    pairs = data["per_pair"]
    questions = sorted({r["question_pos"] for r in pairs} | {r["question_neg"] for r in pairs})
    print(f"{len(pairs)} pairs, {len(questions)} unique questions")

    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    blank = Image.new("RGB", (384, 384), (128, 128, 128))
    cache = {}
    for i, q in enumerate(questions):
        inp = prepare(proc, q, blank)
        cache[q] = margin(model, inp, yes_ids, no_ids)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(questions)}")

    for r in pairs:
        r["m_pos_blank"] = cache[r["question_pos"]]
        r["m_neg_blank"] = cache[r["question_neg"]]
        r["delta_blank"] = r["m_pos_blank"] - r["m_neg_blank"]

    out = {"overall": summarize(pairs), "tasks": {}, "subtypes": {}, "per_pair": pairs}
    for task in sorted(set(r["task"] for r in pairs)):
        out["tasks"][task] = summarize([r for r in pairs if r["task"] == task])
    for key in ("subtype", "negative_type"):
        for val in sorted(set(r.get(key) for r in pairs if r.get(key))):
            out["subtypes"][f"{key}:{val}"] = summarize([r for r in pairs if r.get(key) == val])

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    for name, row in [("overall", out["overall"])] + list(out["tasks"].items()):
        print(f"  {name:12s} n={row['n_pairs']:3d} blank_acc={row['pair_acc']:.3f} "
              f"CI{row['pair_acc_ci95']} AUC={row['auc']:.3f} "
              f"yes+/yes-={row['yes_rate_pos']:.2f}/{row['yes_rate_neg']:.2f}")


if __name__ == "__main__":
    main()
