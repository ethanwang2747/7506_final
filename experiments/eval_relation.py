"""Evaluate the relation task (question only) with an optional LoRA adapter.

Same examples and groups as the readout and side experiments; margins come from
the model's yes/no logits, so numbers are comparable to the earlier tables.

Output:
    results/sft/eval_relation_<tag>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, batch_margins, load_images, load_model, load_processor, yes_no_ids

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records, groups, max_per_group):
    counts, out = {}, []
    for r in records:
        if r["group"] not in set(groups):
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
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = select(read_jsonl(ROOT / args.pairs), GROUPS, args.max_per_group)
    with open(ROOT / "data/derived/experiment_config.json") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)
    image_ids = sorted({r[f"{role}_image_id"] for r in rows for role in ("source", "target")})
    images = (load_images(image_ids, image_dir=ROOT / args.image_dir) if args.image_dir
              else load_images(image_ids))
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    yes_ids, no_ids = yes_no_ids(proc)

    items, meta = [], []
    for r in rows:
        for role in ("source", "target"):
            items.append({"question": r["question"], "image": images[r[f"{role}_image_id"]]})
            meta.append({"pair_id": r["pair_id"], "group": r["group"], "role": role,
                         "label": 1 if role == "source" else 0,
                         "image_id": r[f"{role}_image_id"]})
    margins = batch_margins(model, proc, items, yes_ids, no_ids, batch_size=args.batch_size)
    y = np.array([m["label"] for m in meta])
    m = np.array(margins)
    out = {"adapter": args.adapter, "n": len(y),
           "margins": [round(float(x), 4) for x in margins],
           "meta": meta,
           "acc": float(((m > 0).astype(int) == y).mean()),
           "auc": float(roc_auc_score(y, m)),
           "per_group": {}}
    for g in GROUPS:
        idx = [i for i, x in enumerate(meta) if x["group"] == g]
        if not idx:
            continue
        yy, mm = y[idx], m[idx]
        out["per_group"][g] = {"n": len(idx),
                               "acc": float(((mm > 0).astype(int) == yy).mean()),
                               "auc": float(roc_auc_score(yy, mm))}
    dst = ROOT / (args.out or f"results/sft/eval_relation_{args.tag}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({"n": out["n"], "acc": out["acc"], "auc": out["auc"]}, indent=1))
    for g, v in out["per_group"].items():
        print(f"  {g:32s} acc {v['acc']:.3f} auc {v['auc']:.3f}")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
