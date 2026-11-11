import io
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
IMAGE_DIR = ROOT / "data/raw/gqa/val_balanced_images"


def load_processor(splitting=False):
    proc = AutoProcessor.from_pretrained(MID)
    proc.image_processor.do_image_splitting = splitting
    proc.tokenizer.padding_side = "left"
    return proc


def load_model(attn="sdpa"):
    model = AutoModelForImageTextToText.from_pretrained(
        MID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=attn
    ).eval()
    model.generation_config.pad_token_id = 2
    return model


def yes_no_ids(proc):
    yes = sorted(set(sum([proc.tokenizer.encode(w, add_special_tokens=False) for w in ["yes", "Yes", " yes", " Yes"]], [])))
    no = sorted(set(sum([proc.tokenizer.encode(w, add_special_tokens=False) for w in ["no", "No", " no", " No"]], [])))
    return yes, no


def load_images(image_ids, image_dir=IMAGE_DIR):
    image_ids = sorted(set(image_ids))
    dataset = ds.dataset(str(image_dir), format="parquet")
    tbl = dataset.to_table(filter=pc.is_in(ds.field("id"), value_set=pa.array(image_ids)), columns=["id", "image"])
    out = {}
    for row in tbl.to_pylist():
        out[row["id"]] = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    missing = set(image_ids) - set(out)
    if missing:
        raise KeyError(f"missing images: {sorted(missing)[:5]} (n={len(missing)})")
    return out


def build_prompt(proc, question, instruction="Answer with a single word, yes or no."):
    text = question if not instruction else f"{question}\n{instruction}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(messages, add_generation_prompt=True)


def batch_margins(model, proc, items, yes_ids, no_ids, batch_size=4):
    margins = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [build_prompt(proc, x["question"]) for x in batch]
        images = [[x["image"]] for x in batch]
        inputs = proc(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[:, -1, :]
        lp = torch.log_softmax(logits.float(), dim=-1)
        sy = torch.logsumexp(lp[:, yes_ids], dim=-1)
        sn = torch.logsumexp(lp[:, no_ids], dim=-1)
        margins.extend((sy - sn).tolist())
    return margins


def batch_generate(model, proc, items, batch_size=4, max_new_tokens=4):
    texts = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [build_prompt(proc, x["question"]) for x in batch]
        images = [[x["image"]] for x in batch]
        inputs = proc(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts.extend(proc.tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return texts
