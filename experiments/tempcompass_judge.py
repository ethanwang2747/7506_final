"""Judge TempCompass open descriptions against the reference option."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

DEFAULT_MODEL = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/tempcompass/tempcompass_30.json"))
    ap.add_argument("--out", default=str(ROOT / "results/tempcompass/tempcompass_30_judge.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    items = [x for x in data["items"] if x["task"] == "description"]
    print(f"{len(items)} descriptions")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    ).eval()

    prompts = []
    for it in items:
        ref = it.get("reference", it["answer"])
        for mode in ("full", "blank"):
            prompts.append(
                "You are judging whether a video description matches a reference event.\n"
                f"Reference: {ref}\n"
                f"Description: {it[mode + '_text']}\n"
                "Does the description match the reference event? Reply only yes or no."
            )
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
             for p in prompts]
    outs = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        n = inputs["input_ids"].shape[1]
        outs.extend(tokenizer.batch_decode(out[:, n:], skip_special_tokens=True))

    for i, it in enumerate(items):
        for j, mode in enumerate(("full", "blank")):
            it[mode + "_judge"] = outs[2 * i + j].strip().lower().startswith("yes")

    full = float(np.mean([x["full_judge"] for x in items]))
    blank = float(np.mean([x["blank_judge"] for x in items]))
    by_dim = {}
    for dim in sorted(set(x["dim"] for x in items)):
        sub = [x for x in items if x["dim"] == dim]
        by_dim[dim] = {
            "n": len(sub),
            "full_coverage": float(np.mean([x["full_judge"] for x in sub])),
            "blank_coverage": float(np.mean([x["blank_judge"] for x in sub])),
        }
    out = {"overall": {"n": len(items), "full_coverage": full, "blank_coverage": blank},
           "by_dim": by_dim, "items": items}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    print("overall", out["overall"])
    for dim, row in by_dim.items():
        print(dim, row)
    for x in items[:3]:
        print("REF", x.get("reference"))
        print("GEN", x["full_text"][:200])


if __name__ == "__main__":
    main()
