"""Blank-image control for the SFT adapters.

Replace the image with a uniform gray canvas and ask the relation question.
If an adapter's accuracy stays above chance without any image, the gain is a
language prior rather than visual readout.

Output:
    results/sft/eval_blank_<tag>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, batch_margins, load_model, load_processor, yes_no_ids

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
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    yes_ids, no_ids = yes_no_ids(proc)
    blank = Image.new("RGB", (384, 384), (128, 128, 128))

    items, meta = [], []
    for r in rows:
        for role in ("source", "target"):
            items.append({"question": r["question"], "image": blank})
            meta.append({"pair_id": r["pair_id"], "group": r["group"],
                         "label": 1 if role == "source" else 0})
    margins = np.array(batch_margins(model, proc, items, yes_ids, no_ids,
                                     batch_size=args.batch_size))
    y = np.array([m["label"] for m in meta])
    out = {"adapter": args.adapter, "n": len(y),
           "acc": float(((margins > 0).astype(int) == y).mean()),
           "auc": float(roc_auc_score(y, margins)),
           "yes_rate": float((margins > 0).mean()),
           "per_group": {}}
    for g in GROUPS:
        idx = [i for i, m in enumerate(meta) if m["group"] == g]
        if idx:
            yy, mm = y[idx], margins[idx]
            out["per_group"][g] = {"n": len(idx),
                                   "acc": float(((mm > 0).astype(int) == yy).mean()),
                                   "auc": float(roc_auc_score(yy, mm))}
    dst = ROOT / (args.out or f"results/sft/eval_blank_{args.tag}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "acc", "auc", "yes_rate")}, indent=1))
    for g, v in out["per_group"].items():
        print(f"  {g:32s} acc {v['acc']:.3f} auc {v['auc']:.3f}")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
