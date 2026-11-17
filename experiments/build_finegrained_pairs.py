"""Build same-image fine-grained discrimination pairs from GQA scene graphs.

Three task families:
  existence: a present object vs an absent object
  attribute: a true colour/material vs a false one on the same object
  behavior:  a true relation vs a confusable false relation on the same pair

Both questions in a pair use the same image, so image content, object identity,
and spatial layout are held fixed. This isolates the question-conditioned
readout rather than low-level visual differences.
"""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from experiments.build_relation_pairs import VOCAB
from experiments.e2_ablation import unique_objects

ROOT = Path(__file__).resolve().parents[1]
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
VAL_Q = ROOT / "data/raw/gqa/val_balanced_instructions/val-00000-of-00001.parquet"

COLORS = {"white", "black", "blue", "green", "brown", "red", "gray", "yellow",
          "orange", "pink", "purple", "tan", "silver", "gold"}
MATERIALS = {"wood", "metal", "glass", "plastic", "brick", "concrete", "stone",
             "fabric", "leather", "ceramic"}

CONFUSABLE = {
    "cat": ["dog"], "dog": ["cat"],
    "cup": ["bowl", "bottle"], "bowl": ["cup", "plate"], "plate": ["bowl"],
    "bottle": ["cup", "glass"], "glass": ["cup", "bottle"],
    "chair": ["bench", "stool"], "bench": ["chair"], "stool": ["chair"],
    "car": ["bus", "truck"], "bus": ["car", "truck"], "truck": ["car", "bus"],
    "bike": ["motorcycle"], "motorcycle": ["bike"],
    "couch": ["bed", "bench"], "bed": ["couch"],
    "hat": ["cap", "helmet"], "cap": ["hat", "helmet"], "helmet": ["hat", "cap"],
    "jacket": ["coat", "shirt"], "coat": ["jacket"], "shirt": ["jacket"],
    "table": ["desk", "counter"], "desk": ["table"], "counter": ["table"],
    "tree": ["building", "house"], "building": ["house", "tree"], "house": ["building"],
    "door": ["window"], "window": ["door"],
    "fork": ["knife", "spoon"], "knife": ["fork", "spoon"], "spoon": ["fork", "knife"],
    "phone": ["laptop", "camera"], "laptop": ["phone", "monitor"], "monitor": ["laptop"],
    "bag": ["backpack"], "backpack": ["bag"],
    "pillow": ["cushion", "blanket"], "cushion": ["pillow"], "blanket": ["pillow"],
    "sign": ["pole"], "pole": ["sign"],
    "pan": ["pot"], "pot": ["pan"],
    "shoe": ["boot"], "boot": ["shoe"],
    "pants": ["jeans", "shorts"], "jeans": ["pants", "shorts"], "shorts": ["pants", "jeans"],
}

COLOR_CONFUSABLE = {
    "white": ["gray", "black"], "black": ["gray", "brown"], "gray": ["white", "black"],
    "brown": ["tan", "black"], "tan": ["brown"], "blue": ["green"], "green": ["blue"],
    "red": ["orange", "pink"], "orange": ["red", "yellow"], "yellow": ["orange"],
    "pink": ["red", "purple"], "purple": ["pink"], "silver": ["gray", "white"],
    "gold": ["yellow", "silver"],
}

REL_TEMPLATES = {
    "wearing": "Is the {s} wearing the {o}?",
    "holding": "Is the {s} holding the {o}?",
    "sitting on": "Is the {s} sitting on the {o}?",
    "standing on": "Is the {s} standing on the {o}?",
    "riding": "Is the {s} riding the {o}?",
    "carrying": "Is the {s} carrying the {o}?",
    "eating": "Is the {s} eating the {o}?",
    "looking at": "Is the {s} looking at the {o}?",
    "lying on": "Is the {s} lying on the {o}?",
    "walking on": "Is the {s} walking on the {o}?",
    "on": "Is the {s} on the {o}?",
    "in": "Is the {s} in the {o}?",
}

CLEAR_PREDICATE = {
    "sitting on": ["standing on"],
    "standing on": ["sitting on"],
    "riding": ["walking on"],
    "walking on": ["riding"],
    "lying on": ["sitting on"],
}

CONFUSABLE_REL = {
    "wearing": ["holding", "carrying"],
    "holding": ["carrying", "wearing"],
    "carrying": ["holding", "wearing"],
    "sitting on": ["standing on", "lying on"],
    "standing on": ["sitting on", "walking on"],
    "riding": ["walking on", "standing on"],
    "walking on": ["riding", "standing on"],
    "lying on": ["sitting on"],
    "eating": ["holding"],
    "looking at": ["holding"],
    "on": ["in"],
    "in": ["on"],
}


# Behaviour negatives must be plausible but unambiguously absent. Avoid
# hypernym pairs such as pants/jeans or hat/cap, which can both be true.
BEHAVIOR_CONFUSABLE = {
    "hat": ["helmet"], "cap": ["helmet"], "helmet": ["hat"],
    "pants": ["shorts"], "jeans": ["shorts"], "shorts": ["jeans"],
    "shirt": ["jacket"], "jacket": ["coat"], "coat": ["jacket"],
    "shoe": ["boot"], "boot": ["shoe"], "sneaker": ["boot"],
    "bike": ["car"], "motorcycle": ["car"], "car": ["truck"], "truck": ["car"],
    "bus": ["car"], "chair": ["bench"], "bench": ["chair"],
    "table": ["desk"], "desk": ["table"], "counter": ["table"],
    "bed": ["couch"], "couch": ["bed"], "blanket": ["pillow"], "pillow": ["blanket"],
    "cup": ["bowl"], "bowl": ["plate"], "plate": ["bowl"], "bottle": ["cup"],
    "glass": ["cup"], "fork": ["knife"], "knife": ["spoon"], "spoon": ["fork"],
    "phone": ["laptop"], "laptop": ["monitor"], "monitor": ["laptop"],
    "bag": ["backpack"], "backpack": ["bag"], "tree": ["building"],
    "building": ["house"], "house": ["building"], "door": ["window"], "window": ["door"],
    "sign": ["pole"], "pole": ["sign"], "pan": ["pot"], "pot": ["pan"],
}


def choose_behavior_negative(name, absent):
    for cand in BEHAVIOR_CONFUSABLE.get(name, []):
        if cand in absent:
            return cand
    return None


def split_of(image_id):
    return "dev" if int(hashlib.sha1(image_id.encode()).hexdigest(), 16) % 5 == 0 else "test"


def article(name):
    return "an" if name[:1].lower() in "aeiou" else "a"


def choose_negative(name, absent, rng):
    for cand in CONFUSABLE.get(name, []):
        if cand in absent:
            return cand, "confusable"
    if not absent:
        return None, None
    return rng.choice(sorted(absent)), "random"


def choose_color(attrs, rng):
    present = attrs & COLORS
    if not present:
        return None
    pos = rng.choice(sorted(present))
    for cand in COLOR_CONFUSABLE.get(pos, []):
        if cand not in attrs:
            return pos, cand
    absent = COLORS - attrs
    return pos, (rng.choice(sorted(absent)) if absent else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data/derived"))
    ap.add_argument("--max-per-task", type=int, default=200)
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(SG_PATH) as f:
        sg = json.load(f)
    image_set = set(pd.read_parquet(VAL_Q).imageId.astype(str))

    vocab = set(VOCAB)
    out = {s: collections.defaultdict(list) for s in ("dev", "test")}
    predicate = {s: [] for s in ("dev", "test")}
    for image_id, g in sorted(sg.items()):
        if image_id not in image_set:
            continue
        split = split_of(image_id)
        if len(out[split]["existence"]) >= args.max_per_task and \
           len(out[split]["attribute"]) >= args.max_per_task and \
           len(out[split]["behavior"]) >= args.max_per_task:
            continue
        objs = unique_objects(g, vocab)
        if not objs:
            continue
        rng = random.Random(f"{args.seed}:{image_id}")
        names = sorted(objs)
        all_names = {o["name"].strip().lower() for o in g["objects"].values()}
        absent = vocab - all_names
        id_to_name = {objs[n]["object_id"]: n for n in names}

        # Existence: present vs absent.
        picks = rng.sample(names, min(args.per_image, len(names)))
        for name in picks:
            neg, neg_type = choose_negative(name, absent, rng)
            if not neg:
                continue
            out[split]["existence"].append({
                "pair_id": f"existence:{split}:{image_id}:{name}:{neg}",
                "task": "existence", "group": f"{name}|{neg}",
                "image_id": image_id, "split": split,
                "question_pos": f"Is there {article(name)} {name}?",
                "question_neg": f"Is there {article(neg)} {neg}?",
                "label_pos": 1, "label_neg": 0,
                "negative_type": neg_type,
                "positive_name": name, "negative_name": neg,
            })

        # Attribute: true colour/material vs false.
        attr_picks = [n for n in names if objs[n].get("attributes")]
        rng.shuffle(attr_picks)
        for name in attr_picks[: args.per_image]:
            attrs = {a.lower() for a in objs[name].get("attributes", [])}
            colors = choose_color(attrs, rng)
            if colors:
                pos, neg = colors
                out[split]["attribute"].append({
                    "pair_id": f"attribute:color:{split}:{image_id}:{name}:{pos}:{neg}",
                    "task": "attribute", "subtype": "color", "group": f"{name}|{pos}|{neg}",
                    "image_id": image_id, "split": split,
                    "question_pos": f"Is the {name} {pos}?",
                    "question_neg": f"Is the {name} {neg}?",
                    "label_pos": 1, "label_neg": 0,
                    "negative_type": "attribute_swap",
                    "positive_attr": pos, "negative_attr": neg,
                })
                continue
            materials = attrs & MATERIALS
            if materials:
                pos = rng.choice(sorted(materials))
                absent_m = MATERIALS - attrs
                if not absent_m:
                    continue
                neg = rng.choice(sorted(absent_m))
                out[split]["attribute"].append({
                    "pair_id": f"attribute:material:{split}:{image_id}:{name}:{pos}:{neg}",
                    "task": "attribute", "subtype": "material", "group": f"{name}|{pos}|{neg}",
                    "image_id": image_id, "split": split,
                    "question_pos": f"Is the {name} made of {pos}?",
                    "question_neg": f"Is the {name} made of {neg}?",
                    "label_pos": 1, "label_neg": 0,
                    "negative_type": "attribute_swap",
                    "positive_attr": pos, "negative_attr": neg,
                })

        # Behavior: true relation vs the same relation with an absent confusable object.
        behavior = []
        for s_name in names:
            s_obj = objs[s_name]
            for rel in g["objects"][s_obj["object_id"]].get("relations", []):
                rel_name = rel["name"].strip().lower()
                if rel_name not in REL_TEMPLATES or rel["object"] not in id_to_name:
                    continue
                o_name = id_to_name[rel["object"]]
                neg_obj = choose_behavior_negative(o_name, absent)
                if not neg_obj:
                    continue
                neg_type = "confusable"
                behavior.append({
                    "pair_id": f"behavior:{split}:{image_id}:{s_name}:{rel_name}:{o_name}:{neg_obj}",
                    "task": "behavior", "group": f"{s_name}|{rel_name}|{o_name}",
                    "image_id": image_id, "split": split,
                    "question_pos": REL_TEMPLATES[rel_name].format(s=s_name, o=o_name),
                    "question_neg": REL_TEMPLATES[rel_name].format(s=s_name, o=neg_obj),
                    "label_pos": 1, "label_neg": 0,
                    "negative_type": neg_type,
                    "positive_rel": rel_name, "positive_object": o_name,
                    "negative_object": neg_obj,
                })
        rng.shuffle(behavior)
        out[split]["behavior"].extend(behavior[: args.per_image])

        # Predicate-swap behaviour pilot: same subject/object, only the verb changes.
        for s_name in names:
            s_obj = objs[s_name]
            for rel in g["objects"][s_obj["object_id"]].get("relations", []):
                rel_name = rel["name"].strip().lower()
                if rel_name not in CLEAR_PREDICATE or rel["object"] not in id_to_name:
                    continue
                o_name = id_to_name[rel["object"]]
                present = {r["name"].strip().lower() for r in g["objects"][s_obj["object_id"]].get("relations", [])
                           if r["object"] == rel["object"]}
                for neg_rel in CLEAR_PREDICATE[rel_name]:
                    if neg_rel not in present:
                        predicate[split].append({
                            "pair_id": f"behavior_pred:{split}:{image_id}:{s_name}:{rel_name}:{o_name}:{neg_rel}",
                            "task": "behavior_predicate", "group": f"{s_name}|{rel_name}|{o_name}",
                            "image_id": image_id, "split": split,
                            "question_pos": REL_TEMPLATES[rel_name].format(s=s_name, o=o_name),
                            "question_neg": REL_TEMPLATES[neg_rel].format(s=s_name, o=o_name),
                            "label_pos": 1, "label_neg": 0,
                            "negative_type": "predicate_swap",
                            "positive_rel": rel_name, "negative_rel": neg_rel,
                        })
                        break

    out_dir = Path(args.out_dir)
    for split in ("dev", "test"):
        rows = []
        for task in ("existence", "attribute", "behavior"):
            items = out[split][task]
            rng = random.Random(f"{args.seed}:{split}:{task}")
            if len(items) > args.max_per_task:
                items = rng.sample(items, args.max_per_task)
            rows.extend(items)
        path = out_dir / f"finegrained_pairs_{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts = collections.Counter(r["task"] for r in rows)
        print(split, len(rows), dict(counts), path)
        pitems = predicate[split]
        prng = random.Random(f"{args.seed}:{split}:predicate")
        if len(pitems) > args.max_per_task // 2:
            pitems = prng.sample(pitems, args.max_per_task // 2)
        ppath = out_dir / f"finegrained_predicate_{split}.jsonl"
        with open(ppath, "w") as f:
            for r in pitems:
                f.write(json.dumps(r) + "\n")
        print(split, "predicate", len(pitems), ppath)


if __name__ == "__main__":
    main()
