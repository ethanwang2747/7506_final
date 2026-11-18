"""Can the model report where each object is, when asked directly?

Stage 1 of the plan: before any training, test whether the horizontal position
that the relation question needs can be elicited by asking for it explicitly.
For every test-core example we ask each object's side on two axes:

    Is the {name} on the left half or the right half of the image?
    Is the {name} in the top half or the bottom half of the image?

Answers are compared with the GQA box. If the model reports horizontal sides
well but answers the relation poorly, the capability exists and is not used;
if it cannot report horizontal sides, the concept itself is missing.

Output:
    results/diagnostics/side_report.json
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, batch_generate, load_images, load_model, load_processor

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]
AXES = {
    "horizontal": ("left half", "right half", "left", "right"),
    "vertical": ("top half", "bottom half", "top", "bottom"),
}


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


def side_of(bbox, W, H):
    x, y, w, h = bbox
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    return {"horizontal": "left" if cx < 0.5 else "right",
            "vertical": "top" if cy < 0.5 else "bottom"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/diagnostics/side_report.json")
    args = ap.parse_args()

    rows = select(read_jsonl(ROOT / args.pairs), GROUPS, args.max_per_group)
    with open(ROOT / "data/derived/experiment_config.json") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)
    images = load_images(sorted({r[f"{role}_image_id"] for r in rows
                                 for role in ("source", "target")}))
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")

    items, truth, meta = [], [], []
    for r in rows:
        for role in ("source", "target"):
            iid = r[f"{role}_image_id"]
            g = sg[iid]
            W, H = g["width"], g["height"]
            for obj_key in ("target", "reference"):
                name = r[f"{role}_{obj_key}"]["name"]
                bbox = r[f"{role}_{obj_key}"]["bbox"]
                sides = side_of(bbox, W, H)
                for axis in ("horizontal", "vertical"):
                    lo, hi, short_lo, short_hi = AXES[axis]
                    for order, (a, b) in enumerate(((lo, hi), (hi, lo))):
                        q = (f"Is the {name} on the {a} or the {b} of the image? "
                             f"Answer with one word.")
                        items.append({"question": q, "image": images[iid]})
                        truth.append(sides[axis])
                        meta.append({"pair_id": r["pair_id"], "group": r["group"],
                                     "role": role, "object": obj_key, "name": name,
                                     "axis": axis, "truth": sides[axis], "image_id": iid,
                                     "order": order})

    texts = batch_generate(model, proc, items, batch_size=args.batch_size,
                           max_new_tokens=4, instruction="Answer with one word.")
    preds = []
    for t in texts:
        s = t.strip().lower()
        if re.search(r"\bleft\b", s):
            preds.append("left")
        elif re.search(r"\bright\b", s):
            preds.append("right")
        elif re.search(r"\btop\b", s):
            preds.append("top")
        elif re.search(r"\bbottom\b", s):
            preds.append("bottom")
        else:
            preds.append(None)

    out = {"n_items": len(items), "parse_rate": float(np.mean([p is not None for p in preds])),
           "by_axis": {}, "by_order": {}, "by_object": {}, "by_group": {},
           "by_axis_object": {}, "examples": []}
    for axis in ("horizontal", "vertical"):
        idx = [i for i, m in enumerate(meta) if m["axis"] == axis]
        ok = [preds[i] == truth[i] for i in idx]
        dist = collections.Counter(preds[i] for i in idx)
        conf = collections.Counter((truth[i], preds[i]) for i in idx)
        out["by_axis"][axis] = {
            "n": len(idx), "acc": float(np.mean(ok)),
            "pred_dist": {str(k): int(v) for k, v in dist.items()},
            "majority_baseline": float(max(dist.values()) / max(1, len(idx))),
            "confusion": {f"{a}->{b}": int(v) for (a, b), v in sorted(conf.items())},
        }
    for order in (0, 1):
        for axis in ("horizontal", "vertical"):
            idx = [i for i, m in enumerate(meta) if m["axis"] == axis and m["order"] == order]
            if not idx:
                continue
            dist = collections.Counter(preds[i] for i in idx)
            out["by_order"][f"{axis}:order{order}"] = {
                "n": len(idx),
                "acc": float(np.mean([preds[i] == truth[i] for i in idx])),
                "pred_dist": {str(k): int(v) for k, v in dist.items()},
            }
    for obj in ("target", "reference"):
        for axis in ("horizontal", "vertical"):
            idx = [i for i, m in enumerate(meta) if m["axis"] == axis and m["object"] == obj]
            out["by_axis_object"][f"{obj}:{axis}"] = {
                "n": len(idx),
                "acc": float(np.mean([preds[i] == truth[i] for i in idx])),
            }
    for g in GROUPS:
        idx = [i for i, m in enumerate(meta) if m["group"] == g and m["axis"] == "horizontal"]
        if idx:
            out["by_group"][g] = {
                "n": len(idx),
                "acc": float(np.mean([preds[i] == truth[i] for i in idx])),
            }
    for i in range(min(len(meta), 400)):
        out["examples"].append({**meta[i], "response": texts[i].strip()[:60],
                                "pred": preds[i], "correct": preds[i] == truth[i]})

    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_items", "parse_rate", "by_axis",
                                         "by_axis_object")}, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
