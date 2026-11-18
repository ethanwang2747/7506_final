"""Build a fresh confirmation set from images used in no prior experiment.

Val images cannot supply one: the pair builder already consumed every val image
that has a unique object pair. We therefore use GQA *train* images that are not
in the SFT set and not in any pair file. They are image-disjoint from SFT and
from every prior evaluation, though the base model may have seen them during
pretraining; the write-up states this caveat.

Pairs are formed within a group by matching one positive image with one
negative image, so each record gives one label-1 and one label-0 example.

Output:
    data/derived/spatial_pairs_confirm.jsonl
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.build_val_pairs import OBJECT_VOCAB, relation_label
from experiments.common import ROOT
from experiments.e2_ablation import unique_objects

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]


def used_images():
    used = set()
    for f in glob.glob(str(ROOT / "data/derived/*pairs*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for k in ("source_image_id", "target_image_id"):
                if k in r:
                    used.add(str(r[k]))
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/spatial_pairs_confirm.jsonl")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(ROOT / "data/raw/gqa/train_sceneGraphs.json") as f:
        sg = json.load(f)
    dataset = ds.dataset(str(ROOT / "data/raw/gqa/train_all_images"), format="parquet")
    ids = set()
    for batch in dataset.to_batches(columns=["id"]):
        ids.update(str(x) for x in batch.column("id").to_pylist())
    sft = set()
    with open(ROOT / "data/derived/sft_examples.jsonl") as f:
        for line in f:
            sft.add(json.loads(line)["image_id"])
    free = sorted(ids - used_images() - sft)
    print(f"train images {len(ids)}, sft {len(sft)}, free {len(free)}")

    vocab = set(OBJECT_VOCAB)
    rng = random.Random(args.seed)
    records = []
    for group in GROUPS:
        a, rel, b = group.split("|")
        q = f"Is the {a} {rel} the {b}?"
        pos, neg = [], []
        for iid in free:
            g = sg.get(iid)
            if not g:
                continue
            uniq = unique_objects(g, vocab)
            for obj in uniq.values():
                x, y, w, h = obj["bbox"]
                obj["raw"] = {"x": x, "y": y, "w": w, "h": h}
            if a not in uniq or b not in uniq:
                continue
            lab = relation_label(iid, a, rel, b, {iid: uniq}, sg)
            if lab is None:
                continue
            (pos if lab else neg).append(iid)
        rng.shuffle(pos)
        rng.shuffle(neg)
        n = min(args.per_class, len(pos), len(neg))
        print(f"{group}: pos {len(pos)} neg {len(neg)} using {n}")
        for k in range(n):
            records.append({
                "pair_id": f"confirm:{group}:{k:04d}",
                "group": group, "question": q, "split": "confirm",
                "source_image_id": pos[k], "target_image_id": neg[k],
            })
    out = ROOT / args.out
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"saved {out} ({len(records)} records, {len(records) * 2} examples)")


if __name__ == "__main__":
    main()
