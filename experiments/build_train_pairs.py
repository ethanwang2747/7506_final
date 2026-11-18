"""Build relation pairs from GQA *train* scene graphs for per-group probes.

For a fixed list of relation groups we scan the train scene graphs, keep
images where both objects are unique by name, and label the relation from the
two boxes with the same 2% margin used for the val pairs.

Output records are one (image, question, label) example each:
  {"pair_id": ..., "group": ..., "question": ..., "image_id": ...,
   "label": 0/1, "target": {...}, "reference": {...}, "width": W, "height": H}

Output:
    data/derived/train_pairs_groups.jsonl
"""

import argparse
import collections
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.build_val_pairs import OBJECT_VOCAB, relation_label
from experiments.common import ROOT
from experiments.e2_ablation import unique_objects

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]


def rng_for(key, seed):
    h = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16)
    return random.Random(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/train_pairs_groups.jsonl")
    ap.add_argument("--per-group", type=int, default=600, help="max images kept per group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with zipfile.ZipFile(ROOT / "data/raw/gqa/sceneGraphs.zip") as z:
        with z.open("train_sceneGraphs.json") as f:
            sg = json.load(f)
    print(f"train scene graphs: {len(sg)}")

    vocab = set(OBJECT_VOCAB)
    records = []
    for group in GROUPS:
        a, rel, b = group.split("|")
        q = f"Is the {a} {rel} the {b}?"
        true_imgs, false_imgs = [], []
        for iid, g in sg.items():
            uniq = unique_objects(g, vocab)
            for obj in uniq.values():
                x, y, w, h = obj["bbox"]
                obj["raw"] = {"x": x, "y": y, "w": w, "h": h}
            if a not in uniq or b not in uniq:
                continue
            lab = relation_label(iid, a, rel, b, {iid: uniq}, sg)
            if lab is None:
                continue
            rec = {
                "pair_id": f"train:{group}:{iid}",
                "group": group,
                "question": q,
                "image_id": iid,
                "label": 1 if lab else 0,
                "target": {"name": a, "bbox": uniq[a]["bbox"]},
                "reference": {"name": b, "bbox": uniq[b]["bbox"]},
                "width": g["width"], "height": g["height"],
            }
            (true_imgs if lab else false_imgs).append(rec)
        rng = rng_for(group, args.seed)
        rng.shuffle(true_imgs)
        rng.shuffle(false_imgs)
        n = args.per_group // 2
        picked = true_imgs[:n] + false_imgs[:n]
        rng.shuffle(picked)
        records.extend(picked)
        print(f"{group}: true {len(true_imgs)} false {len(false_imgs)} kept {len(picked)}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"saved {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
