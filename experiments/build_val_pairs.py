import argparse, collections, hashlib, json, math, random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
VAL_Q = ROOT / "data/raw/gqa/val_balanced_instructions/val-00000-of-00001.parquet"

OBJECT_VOCAB = set("""
man woman person girl boy child lady guy player table chair cup plate bowl bottle box bag backpack
blanket pillow car bus truck train bike motorcycle skateboard surfboard boat airplane dog cat horse
elephant giraffe bird laptop phone refrigerator desk couch bed bench shelf cabinet helmet hat cap
jacket coat glasses racket umbrella ball frisbee fence tree building house window door lamp mirror
toilet sink counter sign pole stool pan pot knife fork spoon clock book paper towel basket curtain
cushion fan television monitor keyboard mouse camera guitar piano violin drum toy stroller cart wagon
tractor
""".split())

COLORS = set("red blue green black white brown yellow gray silver gold orange pink purple tan beige".split())
MATERIALS = set("wood metal plastic glass leather fabric stone concrete brick".split())
SIZES = set("small large".split())
SHAPES = set("round square rectangular circular oval".split())

RELATIONS = ["to the left of", "to the right of", "above", "below"]
MARGIN = 0.02


def split_of(image_id: str) -> str:
    h = int(hashlib.sha1(image_id.encode()).hexdigest(), 16)
    return "dev" if h % 5 == 0 else "test"


def token_box(o, W, H):
    x0 = max(0, min(8, math.floor(o["x"] / W * 9)))
    x1 = max(0, min(8, math.ceil((o["x"] + o["w"]) / W * 9) - 1))
    y0 = max(0, min(8, math.floor(o["y"] / H * 9)))
    y1 = max(0, min(8, math.ceil((o["y"] + o["h"]) / H * 9) - 1))
    return [r * 9 + c for r in range(y0, y1 + 1) for c in range(x0, x1 + 1)]


def relation_label(iid, a, rel, b, uniq, sg):
    if a not in uniq[iid] or b not in uniq[iid]:
        return None
    W, H = sg[iid]["width"], sg[iid]["height"]
    oa, ob = uniq[iid][a]["raw"], uniq[iid][b]["raw"]
    dx = ((oa["x"] + oa["w"] / 2) - (ob["x"] + ob["w"] / 2)) / W
    dy = ((oa["y"] + oa["h"] / 2) - (ob["y"] + ob["h"] / 2)) / H
    if rel == "to the left of":
        return True if dx < -MARGIN else (False if dx > MARGIN else None)
    if rel == "to the right of":
        return True if dx > MARGIN else (False if dx < -MARGIN else None)
    if rel == "above":
        return True if dy < -MARGIN else (False if dy > MARGIN else None)
    if rel == "below":
        return True if dy > MARGIN else (False if dy < -MARGIN else None)
    return None


def rng_for(key, seed):
    h = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16)
    return random.Random(h)


def sample_pairs(T, F, split, n, seed):
    T = sorted(x for x in T if split_of(x) == split)
    F = sorted(x for x in F if split_of(x) == split)
    if len(T) < 2 or len(F) < 2:
        return []
    k = min(n, len(T), len(F))
    rng = rng_for(f"{split}:{n}:{seed}", seed)
    Ts = rng.sample(T, k)
    Fs = rng.sample(F, k)
    return list(zip(Ts, Fs))


def object_record(iid, name, obj):
    return {
        "object_id": obj["object_id"],
        "name": name,
        "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
        "tokens_9": token_box(obj["raw"], obj["W"], obj["H"]),
        "attributes": obj["attributes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data/derived"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs-per-group", type=int, default=20)
    ap.add_argument("--min-images", type=int, default=10)
    ap.add_argument("--min-category-images", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(SG_PATH) as f:
        sg = json.load(f)
    val_q = pd.read_parquet(VAL_Q)
    image_ids = sorted(set(val_q.imageId.astype(str)))
    image_set = set(image_ids)

    uniq = {}
    cat_counts = collections.Counter()
    obj_attrs = {}
    for iid, g in sg.items():
        if iid not in image_set:
            continue
        W, H = g["width"], g["height"]
        by = collections.defaultdict(list)
        for oid, o in g["objects"].items():
            name = o["name"].strip().lower()
            if name not in OBJECT_VOCAB:
                continue
            if o["w"] <= 1 or o["h"] <= 1:
                continue
            if o["x"] < 0 or o["y"] < 0 or o["x"] + o["w"] > W or o["y"] + o["h"] > H:
                continue
            if o["w"] * o["h"] < 0.005 * W * H:
                continue
            rec = dict(o)
            rec["object_id"] = oid
            rec["raw"] = o
            rec["W"] = W
            rec["H"] = H
            rec["attributes"] = [a.lower() for a in o.get("attributes", [])]
            by[name].append(rec)
        uniq[iid] = {}
        for name, objs in by.items():
            if len(objs) != 1:
                continue
            obj = objs[0]
            if len(token_box(obj["raw"], W, H)) < 2:
                continue
            uniq[iid][name] = obj
            cat_counts[name] += 1
            attrs = {}
            for dim, vocab in [("color", COLORS), ("material", MATERIALS), ("size", SIZES), ("shape", SHAPES)]:
                vals = sorted(set(obj["attributes"]) & vocab)
                if len(vals) == 1:
                    attrs[dim] = vals[0]
            obj_attrs[(iid, name)] = attrs

    categories = {n for n, c in cat_counts.items() if c >= args.min_category_images}
    print(f"images={len(image_ids)} with_objects={sum(1 for x in uniq.values() if x)} "
          f"categories={len(categories)}")

    spatial_groups = collections.defaultdict(lambda: {"T": set(), "F": set()})
    for iid, objs in uniq.items():
        names = [n for n in objs if n in categories]
        for a in names:
            for b in names:
                if a == b:
                    continue
                for rel in RELATIONS:
                    lab = relation_label(iid, a, rel, b, uniq, sg)
                    if lab is True:
                        spatial_groups[(a, rel, b)]["T"].add(iid)
                    elif lab is False:
                        spatial_groups[(a, rel, b)]["F"].add(iid)

    spatial_eligible = {k: v for k, v in spatial_groups.items()
                        if min(len(v["T"]), len(v["F"])) >= args.min_images}
    print(f"spatial groups total={len(spatial_groups)} eligible={len(spatial_eligible)}")

    semantic_groups = collections.defaultdict(lambda: {"P": set(), "N": set()})
    by_cat_dim = collections.defaultdict(lambda: collections.defaultdict(list))
    for (iid, name), attrs in obj_attrs.items():
        if name not in categories:
            continue
        for dim, val in attrs.items():
            by_cat_dim[(name, dim)][val].append(iid)
    for (name, dim), vals in by_cat_dim.items():
        if len(vals) < 2:
            continue
        for val, pos in vals.items():
            neg = set()
            for v2, ids in vals.items():
                if v2 != val:
                    neg.update(ids)
            if min(len(pos), len(neg)) >= args.min_images:
                semantic_groups[(name, dim, val)]["P"].update(pos)
                semantic_groups[(name, dim, val)]["N"].update(neg)
    print(f"semantic groups eligible={len(semantic_groups)}")

    def question_for_spatial(a, rel, b):
        return f"Is the {a} {rel} the {b}?"

    def question_for_semantic(name, dim, val):
        if dim == "material":
            return f"Is the {name} made of {val}?"
        return f"Is the {name} {val}?"

    manifest = {
        "seed": args.seed,
        "split_rule": "sha1(image_id) % 5 == 0 -> dev, else test",
        "source": {
            "scene_graphs": str(SG_PATH.relative_to(ROOT)),
            "val_instructions": str(VAL_Q.relative_to(ROOT)),
            "scene_graph_images": len(sg),
            "val_question_images": len(image_ids),
        },
        "object_vocab": sorted(OBJECT_VOCAB),
        "categories_used": sorted(categories),
        "spatial_groups_total": len(spatial_groups),
        "spatial_groups_eligible": len(spatial_eligible),
        "semantic_groups_eligible": len(semantic_groups),
        "pairs_per_group": args.pairs_per_group,
        "min_images": args.min_images,
        "min_category_images": args.min_category_images,
        "outputs": {},
    }

    for split in ["dev", "test"]:
        spatial_rows = []
        for (a, rel, b), d in sorted(spatial_eligible.items()):
            q = question_for_spatial(a, rel, b)
            for i, (src, tgt) in enumerate(sample_pairs(d["T"], d["F"], split, args.pairs_per_group, args.seed)):
                so, to = uniq[src], uniq[tgt]
                rows = []
                for name, image_id, objs in [(a, src, so), (b, src, so), (a, tgt, to), (b, tgt, to)]:
                    rows.append(object_record(image_id, name, objs[name]))
                spatial_rows.append({
                    "pair_id": f"spatial:{split}:{a}|{rel}|{b}:{i:04d}",
                    "task": "spatial",
                    "group": f"{a}|{rel}|{b}",
                    "split": split,
                    "question": q,
                    "source_image_id": src,
                    "target_image_id": tgt,
                    "source_gold": "yes",
                    "target_gold": "no",
                    "source_target": rows[0],
                    "source_reference": rows[1],
                    "target_target": rows[2],
                    "target_reference": rows[3],
                })
        path = out_dir / f"spatial_pairs_{split}.jsonl"
        with open(path, "w") as f:
            for r in spatial_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["outputs"][f"spatial_{split}"] = {"path": str(path.relative_to(ROOT)), "pairs": len(spatial_rows),
                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"spatial {split}: {len(spatial_rows)} pairs -> {path}")

        semantic_rows = []
        for (name, dim, val), d in sorted(semantic_groups.items()):
            q = question_for_semantic(name, dim, val)
            for i, (src, tgt) in enumerate(sample_pairs(d["P"], d["N"], split, args.pairs_per_group, args.seed)):
                semantic_rows.append({
                    "pair_id": f"semantic:{split}:{name}|{dim}|{val}:{i:04d}",
                    "task": "semantic",
                    "group": f"{name}|{dim}|{val}",
                    "split": split,
                    "question": q,
                    "source_image_id": src,
                    "target_image_id": tgt,
                    "source_gold": "yes",
                    "target_gold": "no",
                    "source_target": object_record(src, name, uniq[src][name]),
                    "target_target": object_record(tgt, name, uniq[tgt][name]),
                })
        path = out_dir / f"semantic_pairs_{split}.jsonl"
        with open(path, "w") as f:
            for r in semantic_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["outputs"][f"semantic_{split}"] = {"path": str(path.relative_to(ROOT)), "pairs": len(semantic_rows),
                                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"semantic {split}: {len(semantic_rows)} pairs -> {path}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
