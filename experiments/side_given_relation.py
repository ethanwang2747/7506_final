"""If the two sides are given in the prompt, can the model compose them?

This is the composition half of the Stage-1 test. The previous script shows the
model cannot report an object's side when asked. Here we hand it the true sides
and ask the relation question, and we add a swapped-side condition as a
manipulation check:

  true sides    "The chair is on the left half. The table is on the right half.
                 Is the chair to the left of the table?"
  swapped sides same question, sides flipped

If accuracy rises with true sides and falls with swapped sides, composition
works once the ingredients are supplied, and the training target is side
reporting. If both stay at chance, composition itself is missing.

Output:
    results/diagnostics/side_given_relation.json
"""

import argparse
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


def side_word(cx, axis):
    if axis == "horizontal":
        return "left" if cx < 0.5 else "right"
    return "top" if cx < 0.5 else "bottom"


def flip(word):
    return {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[word]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="results/diagnostics/side_given_relation.json")
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
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))

    items, meta = [], []
    for r in rows:
        rel_axis = "vertical" if r["group"].split("|")[1] in ("above", "below") else "horizontal"
        for role in ("source", "target"):
            iid = r[f"{role}_image_id"]
            g = sg[iid]
            W, H = g["width"], g["height"]
            t = r[f"{role}_target"]
            ref = r[f"{role}_reference"]
            tx = (t["bbox"][0] + t["bbox"][2] / 2) / W
            ty = (t["bbox"][1] + t["bbox"][3] / 2) / H
            rx = (ref["bbox"][0] + ref["bbox"][2] / 2) / W
            ry = (ref["bbox"][1] + ref["bbox"][3] / 2) / H
            ts = side_word(tx if rel_axis == "horizontal" else ty, rel_axis)
            rs = side_word(rx if rel_axis == "horizontal" else ry, rel_axis)
            name_t, name_r = t["name"], ref["name"]
            q = r["question"]
            for cond in ("true", "swapped"):
                a, b = (ts, rs) if cond == "true" else (flip(ts), flip(rs))
                prompt = (f"The {name_t} is on the {a} half of the image. "
                          f"The {name_r} is on the {b} half. {q} "
                          f"Answer yes or no.")
                items.append({"question": prompt, "image": images[iid]})
                meta.append({"pair_id": r["pair_id"], "group": r["group"], "role": role,
                             "cond": cond, "label": 1 if role == "source" else 0,
                             "same_half": ts == rs, "target_side": ts, "ref_side": rs,
                             "image_id": iid})
    texts = batch_generate(model, proc, items, batch_size=args.batch_size,
                           max_new_tokens=2)
    preds = []
    for t in texts:
        s = t.strip().lower()
        preds.append(1 if re.search(r"\byes\b", s) else (0 if re.search(r"\bno\b", s) else None))

    out = {"n_items": len(items), "parse_rate": float(np.mean([p is not None for p in preds])),
           "conditions": {}}
    for cond in ("true", "swapped"):
        idx = [i for i, m in enumerate(meta) if m["cond"] == cond and preds[i] is not None]
        y = np.array([meta[i]["label"] for i in idx])
        p = np.array([preds[i] for i in idx])
        out["conditions"][cond] = {
            "n": len(idx), "acc": float((p == y).mean()),
            "yes_rate": float(p.mean()),
        }
        # examples where the two objects are in different halves
        diff = [i for i in idx if not meta[i]["same_half"]]
        if diff:
            yd = np.array([meta[i]["label"] for i in diff])
            pd = np.array([preds[i] for i in diff])
            out["conditions"][cond]["diff_half_n"] = len(diff)
            out["conditions"][cond]["diff_half_acc"] = float((pd == yd).mean())

    out["by_group"] = {}
    for g in GROUPS:
        entry = {}
        for cond in ("true", "swapped"):
            idx = [i for i, m in enumerate(meta) if m["group"] == g and m["cond"] == cond
                   and preds[i] is not None]
            if idx:
                y = np.array([meta[i]["label"] for i in idx])
                p = np.array([preds[i] for i in idx])
                entry[cond] = {"n": len(idx), "acc": float((p == y).mean())}
        out["by_group"][g] = entry

    out["examples"] = [{**meta[i], "response": texts[i].strip()[:40]}
                       for i in range(min(len(meta), 200))]
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({"n_items": out["n_items"], "parse_rate": out["parse_rate"],
                      "conditions": out["conditions"], "by_group": out["by_group"]}, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
