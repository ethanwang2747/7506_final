"""Compare cued yes/no recognition with open-ended spatial description.

For the same source/target pairs we ask:
  1. the original yes/no question (candidate margin), and
  2. "Describe the spatial relationship between A and B in one short sentence."

The description is parsed for left/right/above/below. This tests whether the
model can produce the relation it can sometimes recognise when cued.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (ROOT, build_prompt, load_images, load_model,
                                load_processor, yes_no_ids)
from experiments.e0_validate import margin, prepare

REL_WORDS = {
    "left": ["left"],
    "right": ["right"],
    "above": ["above", "over", "on top"],
    "below": ["below", "under", "beneath"],
}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_relation(text):
    t = text.lower()
    found = {}
    for rel, words in REL_WORDS.items():
        for w in words:
            if w in t:
                found[rel] = t.find(w)
                break
    if not found:
        return "none", []
    ordered = sorted(found.items(), key=lambda x: x[1])
    return ordered[0][0], [r for r, _ in ordered]


def generate_batch(model, proc, images, items, max_new_tokens):
    prompts = [build_prompt(proc, it["prompt"], instruction=None) for it in items]
    imgs = [[images[it["image_id"]]] for it in items]
    inputs = proc(text=prompts, images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.tokenizer.batch_decode(out[:, n:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data/derived/spatial_pairs_test.jsonl"))
    ap.add_argument("--groups", default=(
        "woman|to the left of|man,man|to the right of|woman,"
        "man|to the left of|person,person|to the right of|man,"
        "tree|to the left of|building,chair|to the left of|table"))
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/description/spatial_test.json"))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    groups = args.groups.split(",")
    rows = [r for r in read_jsonl(args.pairs) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[: args.pairs_per_group])

    items = []
    for r in selected:
        rel = r["group"].split("|")[1]
        rel_word = "left" if "left" in rel else "right"
        a = r["source_target"]["name"]
        b = r["source_reference"]["name"]
        prompt = f"Describe the spatial relationship between the {a} and the {b} in one short sentence."
        for role, image_id, expected in [
            ("source", r["source_image_id"], rel_word),
            ("target", r["target_image_id"], OPPOSITE[rel_word]),
        ]:
            items.append({
                "pair_id": r["pair_id"], "group": r["group"], "role": role,
                "image_id": image_id, "question": r["question"], "prompt": prompt,
                "expected": expected,
            })

    print(f"{len(selected)} pairs, {len(items)} descriptions")
    images = load_images(sorted({it["image_id"] for it in items}))
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    desc = {}
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        texts = generate_batch(model, proc, images, batch, args.max_new_tokens)
        for it, text in zip(batch, texts):
            first, all_rels = parse_relation(text)
            desc[(it["pair_id"], it["role"])] = {
                "text": text, "first": first, "all_relations": all_rels,
                "correct": it["expected"] in all_rels and OPPOSITE[it["expected"]] not in all_rels,
                "wrong": OPPOSITE[it["expected"]] in all_rels and it["expected"] not in all_rels,
                "none": not all_rels,
                "expected": it["expected"],
            }
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"  {min(start + len(batch), len(items))}/{len(items)}")

    # Cued yes/no margins on the same pairs.
    margins = {}
    for r in selected:
        for role, image_id in [("source", r["source_image_id"]), ("target", r["target_image_id"])]:
            key = (r["pair_id"], role)
            margins[key] = margin(model, prepare(proc, r["question"], images[image_id]), yes_ids, no_ids)

    per_pair = []
    for r in selected:
        ds = desc[(r["pair_id"], "source")]
        dt = desc[(r["pair_id"], "target")]
        ms = margins[(r["pair_id"], "source")]
        mt = margins[(r["pair_id"], "target")]
        per_pair.append({
            "pair_id": r["pair_id"], "group": r["group"],
            "desc_source": ds, "desc_target": dt,
            "m_source": ms, "m_target": mt,
            "yesno_correct": ms > mt,
            "desc_pair_correct": ds["correct"] and dt["correct"],
        })

    def agg(rows_sub):
        return {
            "n_pairs": len(rows_sub),
            "desc_source_acc": float(np.mean([x["desc_source"]["correct"] for x in rows_sub])),
            "desc_target_acc": float(np.mean([x["desc_target"]["correct"] for x in rows_sub])),
            "desc_pair_acc": float(np.mean([x["desc_pair_correct"] for x in rows_sub])),
            "desc_source_none": float(np.mean([x["desc_source"]["none"] for x in rows_sub])),
            "desc_target_none": float(np.mean([x["desc_target"]["none"] for x in rows_sub])),
            "yesno_pair_acc": float(np.mean([x["yesno_correct"] for x in rows_sub])),
        }

    results = {"overall": agg(per_pair), "groups": {}, "per_pair": per_pair}
    for g in groups:
        sub = [x for x in per_pair if x["group"] == g]
        if sub:
            results["groups"][g] = agg(sub)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for name, row in [("overall", results["overall"])] + list(results["groups"].items()):
        print(f"  {name:38s} yes/no={row['yesno_pair_acc']:.3f} "
              f"desc_pair={row['desc_pair_acc']:.3f} "
              f"desc_src={row['desc_source_acc']:.3f} desc_tgt={row['desc_target_acc']:.3f} "
              f"none={row['desc_source_none']:.2f}/{row['desc_target_none']:.2f}")
    print("examples:")
    for x in per_pair[:4]:
        print(" ", x["group"], "S:", x["desc_source"]["text"].strip())
        print("    T:", x["desc_target"]["text"].strip())


if __name__ == "__main__":
    main()
