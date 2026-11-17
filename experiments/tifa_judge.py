"""Judge whether a generated description covers TIFA facts.

Uses a local text-only instruction model as a factual-coverage judge. For each
TIFA yes/no fact, ask whether the description contains enough information to
answer the question with the reference answer. This is a CLAIR-style text
entailment check, not a visual judge.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

DEFAULT_MODEL = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")


def make_prompt(caption, question, answer):
    return (
        "You are checking whether a caption contains a fact.\n"
        f"Caption: {caption}\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
        "Does the caption contain enough information to answer the question "
        "with the correct answer? Reply only yes or no."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/tifa/tifa_100.json"))
    ap.add_argument("--out", default=str(ROOT / "results/tifa/tifa_100_judge.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    questions = data["questions"]
    if args.limit:
        questions = questions[: args.limit]
    print(f"{len(questions)} questions, judge={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    ).eval()

    prompts = [make_prompt(q["description"], q["question"], q["answer"]) for q in questions]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
             for p in prompts]
    outputs = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        n = inputs["input_ids"].shape[1]
        outputs.extend(tokenizer.batch_decode(out[:, n:], skip_special_tokens=True))
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"  {min(start + len(batch), len(texts))}/{len(texts)}")

    for q, raw in zip(questions, outputs):
        t = raw.strip().lower()
        q["judge_raw"] = raw.strip()
        q["judge_yes"] = t.startswith("yes")
        # keyword coverage as a sanity check
        cap = re.sub(r"[^a-z0-9 ]", " ", q["description"].lower())
        w = re.sub(r"[^a-z0-9 ]", " ", q["element"].lower()).strip()
        q["keyword_yes"] = bool(w) and (w in cap or w.rstrip("s") in cap)

    def summary(rows):
        return {
            "n": len(rows),
            "cued_recall": float(np.mean([r["correct"] for r in rows])),
            "judge_coverage": float(np.mean([r["judge_yes"] for r in rows])),
            "keyword_coverage": float(np.mean([r["keyword_yes"] for r in rows])),
            "gap_cued_minus_judge": float(np.mean([r["correct"] for r in rows]) -
                                          np.mean([r["judge_yes"] for r in rows])),
        }

    out = {
        "config": vars(args),
        "overall": summary(questions),
        "by_element_type": {},
        "questions": questions,
    }
    for typ in sorted(set(q["element_type"] for q in questions)):
        out["by_element_type"][typ] = summary([q for q in questions if q["element_type"] == typ])

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    print("  overall", out["overall"])
    for typ, row in out["by_element_type"].items():
        print(f"  {typ:15s} n={row['n']:3d} cued={row['cued_recall']:.3f} "
              f"judge={row['judge_coverage']:.3f} keyword={row['keyword_coverage']:.3f}")
    omissions = [q for q in questions if q["correct"] and not q["judge_yes"]]
    print(f"omissions (cued yes, caption no): {len(omissions)}")
    for q in omissions[:5]:
        print("  Q:", q["question"])
        print("   cap:", q["description"][:220])


if __name__ == "__main__":
    main()
