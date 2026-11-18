"""Build the SFT corpus: side questions plus relation questions.

Side questions come from GQA train scene graphs: for each image we take up to
two objects that are unique by name and ask which half of the image they are
in. Relation questions reuse `train_pairs_groups.jsonl`.

Images are exported once to a parquet with id + JPEG bytes so training does not
touch the 22 raw shards.

Outputs:
    data/derived/sft_examples.jsonl
    data/derived/sft_images.parquet
"""

import argparse
import collections
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.build_val_pairs import OBJECT_VOCAB
from experiments.common import ROOT
from experiments.e2_ablation import unique_objects

SHARD_DIR = ROOT / "data/raw/gqa/train_all_images"


def rng_for(key, seed):
    h = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16)
    return random.Random(h)


def side_of(bbox, W, H):
    x, y, w, h = bbox
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    return ("left" if cx < 0.5 else "right", "top" if cy < 0.5 else "bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/train_pairs_groups.jsonl")
    ap.add_argument("--out", default="data/derived/sft_examples.jsonl")
    ap.add_argument("--images-out", default="data/derived/sft_images.parquet")
    ap.add_argument("--objects-per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(ROOT / args.pairs) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    image_ids = sorted({r["image_id"] for r in pairs})
    print(f"relation pairs {len(pairs)}, images {len(image_ids)}")

    with zipfile.ZipFile(ROOT / "data/raw/gqa/sceneGraphs.zip") as z:
        with z.open("train_sceneGraphs.json") as f:
            sg = json.load(f)
    vocab = set(OBJECT_VOCAB)

    side_examples = []
    used_ids = set(image_ids)
    for iid in image_ids:
        g = sg.get(iid)
        if not g:
            continue
        W, H = g["width"], g["height"]
        uniq = unique_objects(g, vocab)
        names = sorted(uniq, key=lambda n: rng_for(f"{iid}:{n}", args.seed).random())
        for name in names[: args.objects_per_image]:
            obj = uniq[name]
            x, y, w, h = obj["bbox"]
            if w <= 1 or h <= 1:
                continue
            hside, vside = side_of(obj["bbox"], W, H)
            side_examples.append({
                "task": "side_h", "image_id": iid, "name": name,
                "question": f"Is the {name} on the left half or the right half of the image?",
                "answer": hside,
            })
            side_examples.append({
                "task": "side_v", "image_id": iid, "name": name,
                "question": f"Is the {name} in the top half or the bottom half of the image?",
                "answer": vside,
            })
        used_ids.add(iid)

    relation_examples = [{
        "task": "relation", "image_id": r["image_id"], "name": "",
        "question": r["question"],
        "answer": "yes" if r["label"] == 1 else "no",
        "group": r["group"],
    } for r in pairs]

    print("side examples", len(side_examples),
          collections.Counter(e["task"] for e in side_examples))
    print("side answer balance",
          {t: collections.Counter(e["answer"] for e in side_examples if e["task"] == t)
           for t in ("side_h", "side_v")})

    examples = side_examples + relation_examples
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"saved {out} ({len(examples)} examples)")

    print(f"exporting {len(used_ids)} images ...")
    ids, blobs = [], []
    for path in sorted(SHARD_DIR.glob("*.parquet")):
        table = pq.read_table(path, columns=["id", "image"])
        for row in table.to_pylist():
            if row["id"] in used_ids:
                ids.append(row["id"])
                blobs.append(row["image"]["bytes"])
        if len(ids) == len(used_ids):
            break
    print(f"exported {len(ids)} images")
    tbl = pa.table({"id": ids, "image": blobs})
    pq.write_table(tbl, ROOT / args.images_out, compression="zstd")
    print(f"saved {args.images_out}")


if __name__ == "__main__":
    main()
