"""Evaluate same-image fine-grained discrimination pairs on SmolVLM2.

Each pair uses one image and two questions (positive vs negative). We measure
the model's candidate margin difference, horizontal-flip consistency, and a
grayscale-image control. The same image is used throughout, so image content
and spatial layout are held fixed.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap_pair_acc(rows, key, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        vals.append(float(np.mean([rows[i][key] > 0 for i in idx])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_auc(rows, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        y, s = [], []
        for i in idx:
            y.extend([1, 0])
            s.extend([rows[i]["m_pos"], rows[i]["m_neg"]])
        if len(set(y)) > 1:
            vals.append(roc_auc_score(y, s))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def summarize(rows, label):
    if not rows:
        return None
    acc = float(np.mean([r["delta"] > 0 for r in rows]))
    cons = float(np.mean([r["delta_flip"] > 0 for r in rows]))
    comb = float(np.mean([r["delta"] + r["delta_flip"] > 0 for r in rows]))
    gray = float(np.mean([r["delta_gray"] > 0 for r in rows])) if rows[0].get("delta_gray") is not None else None
    agree = float(np.mean([(r["delta"] > 0) == (r["delta_flip"] > 0) for r in rows]))
    y, s = [], []
    for r in rows:
        y.extend([1, 0])
        s.extend([r["m_pos"], r["m_neg"]])
    return {
        "label": label,
        "n_pairs": len(rows),
        "pair_acc": acc,
        "pair_acc_ci95": bootstrap_pair_acc(rows, "delta"),
        "auc": float(roc_auc_score(y, s)),
        "auc_ci95": bootstrap_auc(rows),
        "flip_pair_acc": cons,
        "combined_pair_acc": comb,
        "gray_pair_acc": gray,
        "flip_agreement": agree,
        "mean_delta": float(np.mean([r["delta"] for r in rows])),
        "yes_rate_pos": float(np.mean([r["m_pos"] > 0 for r in rows])),
        "yes_rate_neg": float(np.mean([r["m_neg"] > 0 for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data/derived/finegrained_pairs_test.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/finegrained/test.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gray", action="store_true", default=True)
    ap.add_argument("--no-gray", dest="gray", action="store_false")
    args = ap.parse_args()

    rows = read_jsonl(args.pairs)
    if args.limit:
        rows = rows[: args.limit]
    image_ids = sorted({r["image_id"] for r in rows})
    print(f"{len(rows)} pairs, {len(image_ids)} images")
    images = load_images(image_ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    cache = {}

    def get_margin(image_id, question, transform):
        key = (image_id, question, transform)
        if key in cache:
            return cache[key]
        img = images[image_id]
        if transform == "flip":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif transform == "gray":
            img = ImageOps.grayscale(img).convert("RGB")
        inp = prepare(proc, question, img)
        m = margin(model, inp, yes_ids, no_ids)
        cache[key] = m
        return m

    per_pair = []
    t0 = time.time()
    for i, r in enumerate(rows):
        image_id = r["image_id"]
        m_pos = get_margin(image_id, r["question_pos"], "orig")
        m_neg = get_margin(image_id, r["question_neg"], "orig")
        m_pos_f = get_margin(image_id, r["question_pos"], "flip")
        m_neg_f = get_margin(image_id, r["question_neg"], "flip")
        m_pos_g = get_margin(image_id, r["question_pos"], "gray") if args.gray else None
        m_neg_g = get_margin(image_id, r["question_neg"], "gray") if args.gray else None
        rec = {
            **{k: r[k] for k in ("pair_id", "task", "subtype", "group", "negative_type",
                                 "image_id", "question_pos", "question_neg") if k in r},
            "m_pos": m_pos, "m_neg": m_neg,
            "m_pos_flip": m_pos_f, "m_neg_flip": m_neg_f,
            "delta": m_pos - m_neg,
            "delta_flip": m_pos_f - m_neg_f,
            "m_pos_gray": m_pos_g, "m_neg_gray": m_neg_g,
            "delta_gray": (m_pos_g - m_neg_g) if args.gray else None,
        }
        per_pair.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")

    tasks = sorted(set(r["task"] for r in per_pair))
    results = {"config": vars(args), "overall": summarize(per_pair, "overall"), "tasks": {}}
    for task in tasks:
        results["tasks"][task] = summarize([r for r in per_pair if r["task"] == task], task)

    results["subtypes"] = {}
    for key in ("subtype", "negative_type"):
        for val in sorted(set(r.get(key) for r in per_pair if r.get(key))):
            rows_sub = [r for r in per_pair if r.get(key) == val]
            results["subtypes"][f"{key}:{val}"] = summarize(rows_sub, f"{key}:{val}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": results, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for row in [results["overall"]] + list(results["tasks"].values()):
        if not row:
            continue
        print(f"  {row['label']:12s} n={row['n_pairs']:3d} acc={row['pair_acc']:.3f} "
              f"CI{row['pair_acc_ci95']} AUC={row['auc']:.3f} "
              f"flip_acc={row['flip_pair_acc']:.3f} comb={row['combined_pair_acc']:.3f} "
              f"gray={row['gray_pair_acc'] if row['gray_pair_acc'] is None else round(row['gray_pair_acc'], 3)} "
              f"agree={row['flip_agreement']:.3f} yes+/yes-={row['yes_rate_pos']:.2f}/{row['yes_rate_neg']:.2f}")


if __name__ == "__main__":
    main()
