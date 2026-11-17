"""TempCompass small-sample video evaluation.

Samples 30 videos (6 per temporal dimension). For each video:
  - 4 yes/no questions (candidate margin),
  - 2 multiple-choice questions (generated letter),
  - 1 open description (keyword coverage of the reference answer).

The same questions are repeated with a black video to measure language priors.
"""

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin

VIDEO_DIR = ROOT / "data/raw/tempcompass/videos"
DATA_DIR = ROOT / "data/raw/tempcompass"
DIMS = ["action", "direction", "speed", "order", "attribute_change"]


def read_task(name):
    df = pd.read_parquet(DATA_DIR / name / "test-00000-of-00001.parquet")
    return df


def sample_frames(path, n=8):
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            frames.append(f.to_image())
    if not frames:
        raise ValueError(f"no frames: {path}")
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def video_inputs(proc, text, frames):
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": text},
    ]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, videos=[[frames]], return_tensors="pt")
    return {k: v.to("cuda") for k, v in inputs.items()}


def generate(model, proc, inputs, max_new_tokens=32):
    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.tokenizer.decode(out[0, n:], skip_special_tokens=True).strip()


def parse_letter(text):
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


def content_words(text):
    stop = set("a an the is are was were of to in on at and or with from by for as it its this that "
               "video man woman person people someone something".split())
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop and len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/tempcompass/tempcompass_30.json"))
    ap.add_argument("--max-videos-per-dim", type=int, default=6)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    yn = read_task("yes_no")
    mc = read_task("multi-choice")
    cap = read_task("captioning")
    yn_by = {k: v for k, v in yn.groupby("video_id")}
    mc_by = {k: v for k, v in mc.groupby("video_id")}
    cap_by = {k: v for k, v in cap.groupby("video_id")}

    videos = []
    for dim in DIMS:
        ids = sorted(set(cap[cap.dim == dim].video_id) & set(yn.video_id) & set(mc.video_id))
        videos.extend(ids[: args.max_videos_per_dim])
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    videos = list(dict.fromkeys(videos))
    print(f"{len(videos)} videos")

    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    items = []
    for vid in videos:
        yrows = yn_by[vid].drop_duplicates("question").head(4)
        mrows = mc_by[vid].drop_duplicates("question").head(2)
        crow = cap_by[vid].iloc[0]
        for _, r in yrows.iterrows():
            items.append({"video_id": vid, "task": "yes_no", "dim": r["dim"],
                          "prompt": r["question"], "answer": r["answer"]})
        for _, r in mrows.iterrows():
            items.append({"video_id": vid, "task": "multi_choice", "dim": r["dim"],
                          "prompt": r["question"] + "\nAnswer with the letter of the correct option.",
                          "answer": r["answer"]})
        items.append({"video_id": vid, "task": "description", "dim": crow["dim"],
                      "prompt": "Describe what is happening in the video in one short sentence.",
                      "answer": crow["mc_answer"], "reference": crow["mc_answer"]})

    print(f"{len(items)} task items; frames={args.frames}")
    frames_cache = {}
    for vid in videos:
        frames_cache[vid] = sample_frames(VIDEO_DIR / f"{vid}.mp4", args.frames)
    blank = [Image.new("RGB", (384, 384), (0, 0, 0)) for _ in range(args.frames)]

    results = []
    t0 = time.time()
    for i, it in enumerate(items):
        frames = frames_cache[it["video_id"]]
        for mode, fr in [("full", frames), ("blank", blank)]:
            inputs = video_inputs(proc, it["prompt"], fr)
            if it["task"] == "yes_no":
                m = margin(model, inputs, yes_ids, no_ids)
                pred = "yes" if m > 0 else "no"
                correct = pred == it["answer"]
                it[f"{mode}_margin"] = m
                it[f"{mode}_pred"] = pred
                it[f"{mode}_correct"] = correct
            else:
                text = generate(model, proc, inputs)
                it[f"{mode}_text"] = text
                if it["task"] == "multi_choice":
                    letter = parse_letter(text)
                    it[f"{mode}_pred"] = letter
                    it[f"{mode}_correct"] = letter == it["answer"].strip()[0]
                else:
                    ref = content_words(it.get("reference", it["answer"]))
                    gen = content_words(text)
                    it[f"{mode}_coverage"] = len(ref & gen) / len(ref) if ref else 0.0
        results.append(it)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")

    def summarize(sub):
        out = {"n": len(sub)}
        for task in ("yes_no", "multi_choice"):
            rows = [x for x in sub if x["task"] == task]
            if rows:
                out[task] = {
                    "n": len(rows),
                    "full_acc": float(np.mean([x["full_correct"] for x in rows])),
                    "blank_acc": float(np.mean([x["blank_correct"] for x in rows])),
                    "yes_rate": float(np.mean([x["full_pred"] == "yes" for x in rows])) if task == "yes_no" else None,
                }
        desc = [x for x in sub if x["task"] == "description"]
        if desc:
            out["description"] = {
                "n": len(desc),
                "full_coverage": float(np.mean([x["full_coverage"] for x in desc])),
                "blank_coverage": float(np.mean([x["blank_coverage"] for x in desc])),
            }
        return out

    summary = {"overall": summarize(results), "by_dim": {}}
    for dim in DIMS:
        summary["by_dim"][dim] = summarize([x for x in results if x["dim"] == dim])

    out = {"config": vars(args), "summary": summary, "items": results}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    print("overall", json.dumps(summary["overall"], indent=2))
    for dim, s in summary["by_dim"].items():
        print(dim, json.dumps(s))


if __name__ == "__main__":
    main()
