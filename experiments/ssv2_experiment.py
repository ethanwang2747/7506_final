"""Something-Something V2 small-sample action discrimination.

Each video gets a true action template and a false template from another video.
Both questions use the same video; a black-video control measures text priors.
"""

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin

VIDEO_DIR = ROOT / "data/raw/ssv2/videos"


def sample_frames(path, n=8):
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            frames.append(f.to_image())
    if not frames:
        raise ValueError(path)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def video_inputs(proc, text, frames):
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": text},
    ]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, videos=[[frames]], return_tensors="pt")
    return {k: v.to("cuda") for k, v in inputs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data/raw/ssv2/sample_30.tsv"))
    ap.add_argument("--out", default=str(ROOT / "results/ssv2/ssv2_30.json"))
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for line in Path(args.sample).read_text().splitlines():
        vid, template = line.split("\t")
        rows.append({"video_id": vid, "template": template})
    templates = sorted(set(r["template"] for r in rows))
    rng = np.random.default_rng(42)
    for i, r in enumerate(rows):
        choices = [t for t in templates if t != r["template"]]
        r["negative_template"] = choices[int(rng.integers(0, len(choices)))]
        r["question_pos"] = f"Does the video show someone {r['template'][0].lower() + r['template'][1:]}?"
        r["question_neg"] = f"Does the video show someone {r['negative_template'][0].lower() + r['negative_template'][1:]}?"

    print(f"{len(rows)} videos")
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    frames_cache = {r["video_id"]: sample_frames(VIDEO_DIR / f"{r['video_id']}.webm", args.frames) for r in rows}
    blank = [Image.new("RGB", (384, 384), (0, 0, 0)) for _ in range(args.frames)]

    for r in rows:
        for mode, frames in [("full", frames_cache[r["video_id"]]), ("blank", blank)]:
            mpos = margin(model, video_inputs(proc, r["question_pos"], frames), yes_ids, no_ids)
            mneg = margin(model, video_inputs(proc, r["question_neg"], frames), yes_ids, no_ids)
            r[f"{mode}_pos"] = mpos
            r[f"{mode}_neg"] = mneg
            r[f"{mode}_delta"] = mpos - mneg
            r[f"{mode}_correct"] = mpos > mneg

    def agg(sub):
        return {
            "n": len(sub),
            "full_acc": float(np.mean([x["full_correct"] for x in sub])),
            "blank_acc": float(np.mean([x["blank_correct"] for x in sub])),
            "full_yes_pos": float(np.mean([x["full_pos"] > 0 for x in sub])),
            "full_yes_neg": float(np.mean([x["full_neg"] > 0 for x in sub])),
        }

    out = {"config": vars(args), "overall": agg(rows), "by_negative_template": {}, "rows": rows}
    for t in sorted(set(r["negative_template"] for r in rows)):
        sub = [r for r in rows if r["negative_template"] == t]
        if len(sub) >= 2:
            out["by_negative_template"][t] = agg(sub)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    print("overall", out["overall"])
    for r in rows[:4]:
        print("VIDEO", r["video_id"])
        print(" +", r["question_pos"], "|", round(r["full_delta"], 3), round(r["blank_delta"], 3))
        print(" -", r["question_neg"])


if __name__ == "__main__":
    main()
