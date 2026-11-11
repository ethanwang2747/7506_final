import argparse, collections, hashlib, json, math, random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
VAL_Q = ROOT / "data/raw/gqa/val_balanced_instructions/val-00000-of-00001.parquet"

BASE = set("""man woman person girl boy child lady guy player table chair cup plate bowl bottle box bag backpack
blanket pillow car bus truck train bike motorcycle skateboard surfboard boat airplane dog cat horse elephant
giraffe bird laptop phone refrigerator desk couch bed bench shelf cabinet helmet hat cap jacket coat glasses
racket umbrella ball frisbee fence tree building house window door lamp mirror toilet sink counter sign pole
stool pan pot knife fork spoon clock book paper towel basket curtain cushion fan television monitor keyboard
mouse camera guitar piano violin drum toy stroller cart wagon tractor""".split())
CLOTHES = set("shirt pants jeans shorts skirt dress suit tie watch shoe shoes socks sandal sandals boot boots glove gloves scarf belt".split())
VOCAB = BASE | CLOTHES
RELATIONS = {"wearing", "on"}


def split_of(image_id):
    h = int(hashlib.sha1(image_id.encode()).hexdigest(), 16)
    return "dev" if h % 5 == 0 else "test"


def token_box(o, W, H):
    x0 = max(0, min(8, math.floor(o["x"] / W * 9)))
    x1 = max(0, min(8, math.ceil((o["x"] + o["w"]) / W * 9) - 1))
    y0 = max(0, min(8, math.floor(o["y"] / H * 9)))
    y1 = max(0, min(8, math.ceil((o["y"] + o["h"]) / H * 9) - 1))
    return [r * 9 + c for r in range(y0, y1 + 1) for c in range(x0, x1 + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data/derived"))
    ap.add_argument("--pairs-per-group", type=int, default=20)
    ap.add_argument("--min-images", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    with open(SG_PATH) as f:
        sg = json.load(f)
    val_q = pd.read_parquet(VAL_Q)
    image_set = set(val_q.imageId.astype(str))

    uniq = {}
    for iid, g in sg.items():
        if iid not in image_set:
            continue
        W, H = g["width"], g["height"]
        by = collections.defaultdict(list)
        for oid, o in g["objects"].items():
            name = o["name"].strip().lower()
            if name not in VOCAB or o["w"] <= 1 or o["h"] <= 1:
                continue
            if o["x"] < 0 or o["y"] < 0 or o["x"] + o["w"] > W or o["y"] + o["h"] > H:
                continue
            if o["w"] * o["h"] < 0.005 * W * H:
                continue
            rec = dict(o); rec["object_id"] = oid; rec["name"] = name; rec["bbox"] = [o["x"], o["y"], o["w"], o["h"]]
            rec["tokens_9"] = token_box(o, W, H)
            if len(rec["tokens_9"]) < 2:
                continue
            by[name].append(rec)
        uniq[iid] = {n: v[0] for n, v in by.items() if len(v) == 1}

    groups = collections.defaultdict(lambda: {"T": set(), "F": set()})
    for iid, objs in uniq.items():
        by_id = {o["object_id"]: (n, o) for n, o in objs.items()}
        edges = set()
        for n, o in objs.items():
            for rel in sg[iid]["objects"][o["object_id"]].get("relations", []):
                if rel["name"] in RELATIONS and rel["object"] in by_id:
                    on, oo = by_id[rel["object"]]
                    edges.add((n, rel["name"], on))
        for n, o in objs.items():
            for m, p in objs.items():
                if n == m:
                    continue
                for rel in RELATIONS:
                    key = (n, rel, m)
                    if key in edges:
                        groups[key]["T"].add(iid)
                    else:
                        groups[key]["F"].add(iid)

    eligible = {k: v for k, v in groups.items() if min(len(v["T"]), len(v["F"])) >= args.min_images}
    print("groups eligible", len(eligible))

    def qtext(a, rel, b):
        return f"Is the {a} wearing the {b}?" if rel == "wearing" else f"Is the {a} on the {b}?"

    def rec(iid, name):
        o = uniq[iid][name]
        return {"object_id": o["object_id"], "name": name, "bbox": o["bbox"], "tokens_9": o["tokens_9"], "attributes": o.get("attributes", [])}

    for split in ["dev", "test"]:
        rows = []
        for (a, rel, b), d in sorted(eligible.items()):
            T = sorted(i for i in d["T"] if split_of(i) == split)
            F = sorted(i for i in d["F"] if split_of(i) == split)
            if len(T) < 2 or len(F) < 2:
                continue
            rng = random.Random(int(hashlib.sha1(f"{split}:{a}:{rel}:{b}".encode()).hexdigest(), 16))
            k = min(args.pairs_per_group, len(T), len(F))
            Ts, Fs = rng.sample(T, k), rng.sample(F, k)
            for i, (s, t) in enumerate(zip(Ts, Fs)):
                rows.append({
                    "pair_id": f"relation:{split}:{a}|{rel}|{b}:{i:04d}",
                    "task": "relation", "group": f"{a}|{rel}|{b}", "split": split,
                    "question": qtext(a, rel, b),
                    "source_image_id": s, "target_image_id": t,
                    "source_gold": "yes", "target_gold": "no",
                    "source_target": rec(s, a), "source_reference": rec(s, b),
                    "target_target": rec(t, a), "target_reference": rec(t, b),
                })
        path = out_dir / f"wearing_on_pairs_{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(split, len(rows), path)


if __name__ == "__main__":
    main()
