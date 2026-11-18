"""Flip-consistency readout with an optional adapter.

For every test example we compute the yes/no margin on the image and on its
horizontal mirror. For left/right relations the mirrored image should be
answered less positively, so `margin(original) - margin(flipped)` is a
training-free consistency score. We report its AUC and pair accuracy next to the
plain margin, for the base model and for a LoRA adapter.

Output:
    results/sft/eval_flip_<tag>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, batch_margins, load_images, load_model, load_processor, yes_no_ids

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records, max_per_group):
    counts, out = {}, []
    for r in records:
        if r["group"] not in GROUPS:
            continue
        n = counts.get(r["group"], 0)
        if max_per_group and n >= max_per_group:
            continue
        counts[r["group"]] = n + 1
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = select(read_jsonl(ROOT / args.pairs), args.max_per_group)
    image_ids = sorted({r[f"{role}_image_id"] for r in rows for role in ("source", "target")})
    images = load_images(image_ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    yes_ids, no_ids = yes_no_ids(proc)

    items, flips, meta = [], [], []
    for r in rows:
        for role in ("source", "target"):
            img = images[r[f"{role}_image_id"]]
            items.append({"question": r["question"], "image": img})
            flips.append({"question": r["question"], "image": img.transpose(Image.FLIP_LEFT_RIGHT)})
            meta.append({"pair_id": r["pair_id"], "group": r["group"], "role": role,
                         "label": 1 if role == "source" else 0})
    m = np.array(batch_margins(model, proc, items, yes_ids, no_ids, batch_size=args.batch_size))
    mf = np.array(batch_margins(model, proc, flips, yes_ids, no_ids, batch_size=args.batch_size))
    y = np.array([x["label"] for x in meta])
    score = m - mf

    def pair_acc(scores):
        by = {}
        for i, x in enumerate(meta):
            by.setdefault(x["pair_id"], {})[x["role"]] = scores[i]
        vals = [v["source"] > v["target"] for v in by.values() if len(v) == 2]
        return float(np.mean(vals)) if vals else None

    out = {"adapter": args.adapter, "n": len(y),
           "margin_auc": float(roc_auc_score(y, m)),
           "margin_acc": float(((m > 0).astype(int) == y).mean()),
           "flip_score_auc": float(roc_auc_score(y, score)),
           "flip_score_acc": float(((score > 0).astype(int) == y).mean()),
           "margin_pair_acc": pair_acc(m),
           "flip_pair_acc": pair_acc(score),
           "per_group": {}}
    for g in GROUPS:
        idx = [i for i, x in enumerate(meta) if x["group"] == g]
        if idx:
            yy = y[idx]
            out["per_group"][g] = {
                "margin_auc": float(roc_auc_score(yy, m[idx])),
                "flip_auc": float(roc_auc_score(yy, score[idx])),
            }
    dst = ROOT / (args.out or f"results/sft/eval_flip_{args.tag}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("margin_auc", "margin_acc", "flip_score_auc",
                                          "flip_score_acc", "margin_pair_acc",
                                          "flip_pair_acc")}, indent=1))
    for g, v in out["per_group"].items():
        print(f"  {g:32s} margin {v['margin_auc']:.3f} flip {v['flip_auc']:.3f}")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
