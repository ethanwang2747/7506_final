"""POPE small-sample evaluation for SmolVLM2.

POPE asks "Is there a <object> in the image?" with balanced yes/no labels.
We report accuracy, yes/no precision-recall-F1, yes-rate, AUC, and a
horizontal-flip consistency check. Confidence intervals cluster by image.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics(rows, key="margin"):
    y = np.array([1 if r["label"] == "yes" else 0 for r in rows])
    s = np.array([r[key] for r in rows])
    pred = (s > 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows), "accuracy": float((pred == y).mean()),
        "precision": precision, "recall": recall, "f1": f1,
        "yes_rate": float(pred.mean()), "auc": float(roc_auc_score(y, s)),
    }


def cluster_bootstrap(rows, fn, n=1000, seed=0):
    images = sorted(set(r["image"] for r in rows))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["image"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(images, size=len(images), replace=True)
        sub = [r for im in sample for r in by[im]]
        vals.append(fn(sub))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data/raw/pope/sample_300.jsonl"))
    ap.add_argument("--image-dir", default=str(ROOT / "data/raw/pope/images"))
    ap.add_argument("--out", default=str(ROOT / "results/pope/pope_300.json"))
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.sample)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} questions, {len(set(r['image'] for r in rows))} images")
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    for i, r in enumerate(rows):
        img = Image.open(Path(args.image_dir) / r["image"]).convert("RGB")
        r["margin"] = margin(model, prepare(proc, r["text"], img), yes_ids, no_ids)
        r["pred"] = "yes" if r["margin"] > 0 else "no"
        if args.flip:
            flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            r["margin_flip"] = margin(model, prepare(proc, r["text"], flip), yes_ids, no_ids)
            r["pred_flip"] = "yes" if r["margin_flip"] > 0 else "no"
            r["combined_margin"] = r["margin"] + r["margin_flip"]
            r["flip_agree"] = r["pred"] == r["pred_flip"]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    results = {"config": vars(args), "overall": {}, "splits": {}, "per_question": rows}
    results["overall"] = metrics(rows)
    results["overall"]["accuracy_ci95"] = cluster_bootstrap(
        rows, lambda sub: float(np.mean([(r["margin"] > 0) == (r["label"] == "yes") for r in sub])))
    results["overall"]["auc_ci95"] = cluster_bootstrap(
        rows, lambda sub: float(roc_auc_score(
            [1 if r["label"] == "yes" else 0 for r in sub], [r["margin"] for r in sub])))
    if args.flip:
        results["overall"]["flip_agreement"] = float(np.mean([r["flip_agree"] for r in rows]))
        results["overall"]["combined_accuracy"] = float(np.mean(
            [(r["combined_margin"] > 0) == (r["label"] == "yes") for r in rows]))
        results["overall"]["combined_auc"] = float(roc_auc_score(
            [1 if r["label"] == "yes" else 0 for r in rows], [r["combined_margin"] for r in rows]))

    for split in sorted(set(r["split"] for r in rows)):
        sub = [r for r in rows if r["split"] == split]
        results["splits"][split] = metrics(sub)
        if args.flip:
            results["splits"][split]["flip_agreement"] = float(np.mean([r["flip_agree"] for r in sub]))
            results["splits"][split]["combined_accuracy"] = float(np.mean(
                [(r["combined_margin"] > 0) == (r["label"] == "yes") for r in sub]))
            results["splits"][split]["combined_auc"] = float(roc_auc_score(
                [1 if r["label"] == "yes" else 0 for r in sub], [r["combined_margin"] for r in sub]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for name, row in [("overall", results["overall"])] + list(results["splits"].items()):
        extra = ""
        if args.flip:
            extra = (f" flip_agree={row['flip_agreement']:.3f} "
                     f"comb_acc={row['combined_accuracy']:.3f} comb_auc={row['combined_auc']:.3f}")
        print(f"  {name:12s} n={row['n']:3d} acc={row['accuracy']:.3f} "
              f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
              f"yes={row['yes_rate']:.3f} AUC={row['auc']:.3f}{extra}")


if __name__ == "__main__":
    main()
