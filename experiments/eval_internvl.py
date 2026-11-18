"""InternVL2.5-2B comparison on the same 480 examples.

InternVL uses its own image preprocessing (dynamic tiling) and chat API, so this
script mirrors `eval_other_vlm.py` with InternVL-specific loading and parsing.
Only accuracy is reported (no yes/no margin AUC).

Output:
    results/other_vlm/internvl2_5_2b.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records, max_per_group):
    counts, out = {}, []
    for r in records:
        if r["group"] not in GROUPS:
            continue
        n = counts.get(r["group"], 0)
        if max_per_group and n >= max_per_group:
            continue
        counts[r["group"]] = n + 1
        out.append(r)
    return out


def build_transform(size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def closest_ratio(ar, ratios, w, h, size):
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        diff = abs(ar - r[0] / r[1])
        if diff < best_diff:
            best, best_diff = r, diff
        elif diff == best_diff and w * h > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def dynamic_preprocess(img, min_num=1, max_num=12, size=448, use_thumbnail=False):
    w, h = img.size
    ar = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    tw, th = closest_ratio(ar, ratios, w, h, size)
    target_w, target_h = size * tw, size * th
    resized = img.resize((target_w, target_h))
    tiles = []
    for i in range(tw * th):
        box = ((i % (target_w // size)) * size, (i // (target_w // size)) * size,
               ((i % (target_w // size)) + 1) * size,
               ((i // (target_w // size)) + 1) * size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(img.resize((size, size)))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/InternVL2_5-2B")
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--out", default="results/other_vlm/internvl2_5_2b.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      trust_remote_code=True, device_map="cuda").eval()
    transform = build_transform(448)
    rows = select(read_jsonl(ROOT / args.pairs), args.max_per_group)
    with open(ROOT / "data/derived/experiment_config.json") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)
    images = load_images(sorted({r[f"{role}_image_id"] for r in rows
                                 for role in ("source", "target")}))

    def pixels(img):
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True)
        return torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).cuda()

    def ask(pv, question, max_new=4):
        out, _ = model.chat(tokenizer, pv, question,
                            dict(max_new_tokens=max_new, do_sample=False))
        return out.strip()

    rel, side, given = [], [], []
    for k, r in enumerate(rows):
        for role in ("source", "target"):
            iid = r[f"{role}_image_id"]
            img = images[iid]
            pv = pixels(img)
            lab = 1 if role == "source" else 0
            txt = ask(pv, r["question"])
            rel.append((lab, 1 if re.search(r"\byes\b", txt.lower()) else 0))
            g = sg[iid]
            W, H = g["width"], g["height"]
            for obj_key in ("target", "reference"):
                name = r[f"{role}_{obj_key}"]["name"]
                x, y, w, h = r[f"{role}_{obj_key}"]["bbox"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                for axis, q, truth in (
                        ("horizontal", f"Is the {name} on the left half or the right half of the image? Answer with one word.",
                         "left" if cx < 0.5 else "right"),
                        ("vertical", f"Is the {name} in the top half or the bottom half of the image? Answer with one word.",
                         "top" if cy < 0.5 else "bottom")):
                    out = ask(pv, q).lower()
                    pred = ("left" if "left" in out else "right" if "right" in out
                            else "top" if "top" in out else "bottom" if "bottom" in out else None)
                    side.append((axis, truth, pred))
            rel_name = r["group"].split("|")[1]
            axis = "vertical" if rel_name in ("above", "below") else "horizontal"
            tx = (r[f"{role}_target"]["bbox"][0] + r[f"{role}_target"]["bbox"][2] / 2) / W
            ty = (r[f"{role}_target"]["bbox"][1] + r[f"{role}_target"]["bbox"][3] / 2) / H
            rx = (r[f"{role}_reference"]["bbox"][0] + r[f"{role}_reference"]["bbox"][2] / 2) / W
            ry = (r[f"{role}_reference"]["bbox"][1] + r[f"{role}_reference"]["bbox"][3] / 2) / H
            def side_word(v):
                if axis == "horizontal":
                    return "left" if v < 0.5 else "right"
                return "top" if v < 0.5 else "bottom"
            ts = side_word(tx if axis == "horizontal" else ty)
            rs = side_word(rx if axis == "horizontal" else ry)
            flip = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
            for cond in ("true", "swapped"):
                a, b = (ts, rs) if cond == "true" else (flip[ts], flip[rs])
                prompt = (f"The {r[f'{role}_target']['name']} is on the {a} half of the image. "
                          f"The {r[f'{role}_reference']['name']} is on the {b} half. {r['question']} "
                          f"Answer yes or no.")
                out = ask(pv, prompt, 2).lower()
                pred = 1 if re.search(r"\byes\b", out) else 0
                given.append((cond, lab, pred, ts == rs))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(rows)}")

    y = np.array([a for a, _ in rel]); p = np.array([b for _, b in rel])
    out = {"model": args.model, "tag": "internvl2_5_2b", "n": len(y),
           "relation_acc": float((p == y).mean()), "side": {}, "side_given": {}}
    for axis in ("horizontal", "vertical"):
        rows_a = [(t, pr) for ax, t, pr in side if ax == axis]
        out["side"][axis] = {
            "n": len(rows_a),
            "acc": float(np.mean([pr == t for t, pr in rows_a])),
            "pred_dist": {str(k): sum(1 for _, pr in rows_a if pr == k)
                          for k in set(pr for _, pr in rows_a)},
            "truth_dist": {str(k): sum(1 for t, _ in rows_a if t == k)
                           for k in set(t for t, _ in rows_a)},
        }
    for cond in ("true", "swapped"):
        rr = [(lab, pr, same) for c, lab, pr, same in given if c == cond]
        diff = [(lab, pr) for lab, pr, same in rr if not same]
        out["side_given"][cond] = {
            "n": len(rr), "acc": float(np.mean([lab == pr for lab, pr, _ in rr])),
            "diff_half_n": len(diff),
            "diff_half_acc": float(np.mean([lab == pr for lab, pr in diff])) if diff else None,
        }
    print(json.dumps({k: out[k] for k in ("relation_acc", "side", "side_given")}, indent=1))
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
