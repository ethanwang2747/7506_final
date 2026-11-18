"""Run the same behavioral diagnostic on another small VLM.

Stages:
  1. relation margin AUC on the 480 test-core examples
  2. side report: left/right and top/bottom questions, two option orders
  3. relation with true vs swapped sides in the prompt

The script is architecture-agnostic: it uses the processor's chat template and
the model's yes/no logits, the same protocol as the SmolVLM2 experiments.

Output:
    results/other_vlm/<tag>.json
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, build_prompt, load_images, yes_no_ids

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]


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


def batch_margins(model, proc, items, yes_ids, no_ids, batch_size=4):
    out = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [build_prompt(proc, x["question"]) for x in batch]
        images = [[x["image"]] for x in batch]
        inputs = proc(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        out.extend((torch.logsumexp(lp[:, yes_ids], -1)
                    - torch.logsumexp(lp[:, no_ids], -1)).tolist())
    return out


def batch_generate(model, proc, items, batch_size=4, max_new_tokens=4,
                   instruction="Answer with one word."):
    texts = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [build_prompt(proc, x["question"], instruction) for x in batch]
        images = [[x["image"]] for x in batch]
        inputs = proc(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        n = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts.extend(proc.tokenizer.batch_decode(gen[:, n:], skip_special_tokens=True))
    return texts


def side_word(cx):
    return "left" if cx < 0.5 else "right"


def flip(word):
    return {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[word]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=384 * 384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    if hasattr(proc, "image_processor"):
        proc.image_processor.max_pixels = args.max_pixels
        proc.image_processor.min_pixels = 64 * 64
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    yes_ids, no_ids = yes_no_ids(proc)
    print(f"{args.tag}: yes {yes_ids[:3]} no {no_ids[:3]}")

    rows = select(read_jsonl(ROOT / args.pairs), args.max_per_group)
    with open(ROOT / "data/derived/experiment_config.json") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)
    images = load_images(sorted({r[f"{role}_image_id"] for r in rows
                                 for role in ("source", "target")}))

    # 1) relation margins
    items, meta = [], []
    for r in rows:
        for role in ("source", "target"):
            items.append({"question": r["question"], "image": images[r[f"{role}_image_id"]]})
            meta.append({"pair_id": r["pair_id"], "group": r["group"], "role": role,
                         "label": 1 if role == "source" else 0})
    margins = np.array(batch_margins(model, proc, items, yes_ids, no_ids, args.batch_size))
    y = np.array([m["label"] for m in meta])
    out = {"model": args.model, "tag": args.tag, "n": len(y),
           "margin_acc": float(((margins > 0).astype(int) == y).mean()),
           "margin_auc": float(roc_auc_score(y, margins)),
           "per_group": {}, "side": {}, "side_given": {}}
    for g in GROUPS:
        idx = [i for i, m in enumerate(meta) if m["group"] == g]
        if idx:
            out["per_group"][g] = {
                "acc": float(((margins[idx] > 0).astype(int) == y[idx]).mean()),
                "auc": float(roc_auc_score(y[idx], margins[idx])),
            }
    print(f"relation acc {out['margin_acc']:.3f} auc {out['margin_auc']:.3f}")

    # 2) side report
    side_items, side_meta = [], []
    for r in rows:
        for role in ("source", "target"):
            iid = r[f"{role}_image_id"]
            g = sg[iid]
            W, H = g["width"], g["height"]
            for obj_key in ("target", "reference"):
                name = r[f"{role}_{obj_key}"]["name"]
                x, y_, w, h = r[f"{role}_{obj_key}"]["bbox"]
                cx, cy = (x + w / 2) / W, (y_ + h / 2) / H
                for axis, lo, hi, truth in (
                        ("horizontal", "left half", "right half", side_word(cx)),
                        ("vertical", "top half", "bottom half",
                         "top" if cy < 0.5 else "bottom")):
                    for order, (a, b) in enumerate(((lo, hi), (hi, lo))):
                        q = (f"Is the {name} on the {a} or the {b} of the image? "
                             f"Answer with one word.")
                        side_items.append({"question": q, "image": images[iid]})
                        side_meta.append({"axis": axis, "truth": truth, "order": order})
    texts = batch_generate(model, proc, side_items, args.batch_size, 4)
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
    for axis in ("horizontal", "vertical"):
        idx = [i for i, m in enumerate(side_meta) if m["axis"] == axis]
        dist = collections.Counter(preds[i] for i in idx)
        out["side"][axis] = {
            "n": len(idx),
            "acc": float(np.mean([preds[i] == side_meta[i]["truth"] for i in idx])),
            "pred_dist": {str(k): int(v) for k, v in dist.items()},
            "truth_dist": {str(k): int(v) for k, v in
                           collections.Counter(side_meta[i]["truth"] for i in idx).items()},
        }
        for order in (0, 1):
            oidx = [i for i in idx if side_meta[i]["order"] == order]
            out["side"][f"{axis}_order{order}"] = {
                "n": len(oidx),
                "acc": float(np.mean([preds[i] == side_meta[i]["truth"] for i in oidx])),
                "pred_dist": {str(k): int(v) for k, v in
                              collections.Counter(preds[i] for i in oidx).items()},
            }
    print(f"side h {out['side']['horizontal']['acc']:.3f} "
          f"v {out['side']['vertical']['acc']:.3f}")

    # 3) relation with sides given
    given_items, given_meta = [], []
    for r in rows:
        rel = r["group"].split("|")[1]
        axis = "vertical" if rel in ("above", "below") else "horizontal"
        for role in ("source", "target"):
            iid = r[f"{role}_image_id"]
            g = sg[iid]
            W, H = g["width"], g["height"]
            t, ref = r[f"{role}_target"], r[f"{role}_reference"]
            tx = (t["bbox"][0] + t["bbox"][2] / 2) / W
            ty = (t["bbox"][1] + t["bbox"][3] / 2) / H
            rx = (ref["bbox"][0] + ref["bbox"][2] / 2) / W
            ry = (ref["bbox"][1] + ref["bbox"][3] / 2) / H
            ts = side_word(tx if axis == "horizontal" else ty)
            rs = side_word(rx if axis == "horizontal" else ry)
            same = ts == rs
            for cond in ("true", "swapped"):
                a, b = (ts, rs) if cond == "true" else (flip(ts), flip(rs))
                prompt = (f"The {t['name']} is on the {a} half of the image. "
                          f"The {ref['name']} is on the {b} half. {r['question']} "
                          f"Answer yes or no.")
                given_items.append({"question": prompt, "image": images[iid]})
                given_meta.append({"cond": cond, "label": 1 if role == "source" else 0,
                                   "same_half": same})
    texts = batch_generate(model, proc, given_items, args.batch_size, 2,
                           instruction="Answer yes or no.")
    preds = [1 if re.search(r"\byes\b", t.lower()) else (0 if re.search(r"\bno\b", t.lower()) else None)
             for t in texts]
    for cond in ("true", "swapped"):
        idx = [i for i, m in enumerate(given_meta) if m["cond"] == cond and preds[i] is not None]
        yy = np.array([given_meta[i]["label"] for i in idx])
        pp = np.array([preds[i] for i in idx])
        diff = [i for i in idx if not given_meta[i]["same_half"]]
        out["side_given"][cond] = {
            "n": len(idx), "acc": float((pp == yy).mean()),
            "diff_half_n": len(diff),
            "diff_half_acc": float((np.array([preds[i] for i in diff])
                                    == np.array([given_meta[i]["label"] for i in diff])).mean())
            if diff else None,
        }
    print(f"side-given true {out['side_given']['true']['acc']:.3f} "
          f"swapped {out['side_given']['swapped']['acc']:.3f}")

    dst = ROOT / (args.out or f"results/other_vlm/{args.tag}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
