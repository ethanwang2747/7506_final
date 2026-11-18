"""LoRA SFT on side questions plus relation questions.

The vision tower is frozen; LoRA is applied to the text attention projections.
Loss is computed on the answer tokens only. The first run is intentionally
small: one epoch over a capped subset, with a held-out dev loss and a JSONL
training log.

Example:
    python experiments/sft_lora.py --max-steps 4000 \
        --out results/sft/adapter_side
"""

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_model, load_processor

PROMPT_SUFFIX = 3  # " answer<end_of_utterance>\n"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(path):
    tbl = pq.read_table(ROOT / path, columns=["id", "image"])
    out = {}
    for row in tbl.to_pylist():
        out[row["id"]] = Image.open(io.BytesIO(row["image"])).convert("RGB")
    return out


def build_inputs(proc, example, image):
    user = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": example["question"]}]}]
    messages = user + [{"role": "assistant",
                        "content": [{"type": "text", "text": example["answer"]}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=False)
    inp = proc(text=text, images=[[image]], return_tensors="pt")
    labels = inp["input_ids"].clone()
    k = PROMPT_SUFFIX
    labels[:, :-k] = -100
    inp = {key: v.to("cuda") for key, v in inp.items()}
    labels = labels.to("cuda")
    return inp, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/derived/sft_examples.jsonl")
    ap.add_argument("--images", default="data/derived/sft_images.parquet")
    ap.add_argument("--out", default="results/sft/adapter_side")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--dev-every", type=int, default=500)
    ap.add_argument("--dev-size", type=int, default=48)
    ap.add_argument("--tasks", default="", help="comma list: side_h,side_v,relation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = read_jsonl(ROOT / args.examples)
    if args.tasks:
        keep = set(args.tasks.split(","))
        examples = [e for e in examples if e["task"] in keep]
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    # hold out a dev split by image id
    dev_images = set()
    for e in examples:
        if len(dev_images) < 60 and rng.random() < 0.02:
            dev_images.add(e["image_id"])
    dev = [e for e in examples if e["image_id"] in dev_images][: args.dev_size]
    train = [e for e in examples if e["image_id"] not in dev_images]
    print(f"train {len(train)} dev {len(dev)} images {len({e['image_id'] for e in train})}")

    images = load_images(args.images)
    print(f"loaded {len(images)} images")

    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = min(args.max_steps, len(train))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / max(1, total)))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log = open(log_path, "w")
    t0 = time.time()
    running, seen = 0.0, 0
    opt.zero_grad()
    for step in range(1, total + 1):
        e = train[step - 1]
        inp, labels = build_inputs(proc, e, images[e["image_id"]])
        out = model(**inp, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()
        running += float(out.loss)
        seen += 1
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if step % args.log_every == 0:
            rec = {"step": step, "loss": running / seen,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "task": e["task"]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step}/{total} loss {rec['loss']:.4f} "
                  f"lr {rec['lr']:.2e} {rec['elapsed_s']}s peak {rec['peak_gb']}GB")
            running, seen = 0.0, 0
        if dev and step % args.dev_every == 0:
            model.eval()
            vals = []
            with torch.no_grad():
                for d in dev:
                    di, dl = build_inputs(proc, d, images[d["image_id"]])
                    vals.append(float(model(**di, labels=dl).loss))
            model.train()
            rec = {"step": step, "dev_loss": sum(vals) / len(vals),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps({"dev": rec}) + "\n")
            log.flush()
            print(f"  dev loss {rec['dev_loss']:.4f}")
        if step % 1000 == 0:
            model.save_pretrained(out_dir / f"step_{step}")
    model.save_pretrained(out_dir)
    log.close()
    print(f"saved adapter to {out_dir}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
