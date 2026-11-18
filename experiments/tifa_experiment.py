"""TIFA/COCO description vs cued QA.

Stage 1: generate a free description for each image, and answer the TIFA
yes/no questions about the same image with the candidate margin.

Stage 2 (separate script) judges whether the generated description covers the
TIFA fact. Comparing the two stages tests whether the model can answer when
cued but omits the fact when describing.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (ROOT, build_prompt, load_model, load_processor,
                                yes_no_ids)
from experiments.e0_validate import margin, prepare

IMAGE_DIR = ROOT / "data/raw/tifa/images"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def image_path(coco_id):
    return IMAGE_DIR / f"COCO_val2014_{int(coco_id):012d}.jpg"


def caption_batch(model, proc, items, images, max_new_tokens, prompt_text):
    prompts = [build_prompt(proc, prompt_text, instruction=None) for _ in items]
    imgs = [[images[it["coco_val_id"]]] for it in items]
    inputs = proc(text=prompts, images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.tokenizer.batch_decode(out[:, n:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data/raw/tifa/sample_100.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/tifa/tifa_100.json"))
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--prompt", default="Describe what is happening in the image in one or two sentences.")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    rows = read_jsonl(args.sample)
    captions = {}
    for r in rows:
        captions[r["id"]] = {"id": r["id"], "coco_val_id": r["coco_val_id"], "caption_ref": r["caption"]}
    items = list(captions.values())
    print(f"{len(rows)} questions, {len(items)} captions/images")

    images = {}
    for it in items:
        from PIL import Image
        images[it["coco_val_id"]] = Image.open(image_path(it["coco_val_id"])).convert("RGB")

    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    yes_ids, no_ids = yes_no_ids(proc)

    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        texts = caption_batch(model, proc, batch, images, args.max_new_tokens, args.prompt)
        for it, text in zip(batch, texts):
            it["description"] = text.strip()
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"  captions {min(start + len(batch), len(items))}/{len(items)} ({time.time() - t0:.0f}s)")

    for i, r in enumerate(rows):
        img = images[r["coco_val_id"]]
        m = margin(model, prepare(proc, r["question"], img), yes_ids, no_ids)
        r["margin"] = m
        r["pred"] = "yes" if m > 0 else "no"
        r["correct"] = r["pred"] == r["answer"]
        r["description"] = captions[r["id"]]["description"]
        if (i + 1) % 100 == 0:
            print(f"  qa {i + 1}/{len(rows)}")

    y = np.array([1 if r["answer"] == "yes" else 0 for r in rows])
    s = np.array([r["margin"] for r in rows])
    overall = {
        "n_questions": len(rows),
        "n_captions": len(items),
        "accuracy": float(np.mean([r["correct"] for r in rows])),
        "auc": float(roc_auc_score(y, s)),
        "yes_rate": float(np.mean([r["pred"] == "yes" for r in rows])),
    }
    by_type = {}
    for typ in sorted(set(r["element_type"] for r in rows)):
        sub = [r for r in rows if r["element_type"] == typ]
        by_type[typ] = {
            "n": len(sub),
            "accuracy": float(np.mean([r["correct"] for r in sub])),
            "yes_rate": float(np.mean([r["pred"] == "yes" for r in sub])),
        }

    out = {
        "config": vars(args),
        "overall": overall,
        "by_element_type": by_type,
        "captions": items,
        "questions": rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    print("  overall", overall)
    for typ, v in by_type.items():
        print(f"  {typ:20s} n={v['n']:3d} acc={v['accuracy']:.3f} yes={v['yes_rate']:.2f}")
    print("caption examples:")
    for it in items[:3]:
        print("  ref:", it["caption_ref"])
        print("  gen:", it["description"])


if __name__ == "__main__":
    main()
