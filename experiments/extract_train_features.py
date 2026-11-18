"""Extract S1/S6/S12 visual tokens for the generated GQA train pairs.

Images live in 22 parquet shards. Each shard is read once, the needed ids are
decoded, and the model runs one forward per (image, question) example with the
same hooks used for the cached eval features.

Output:
    results/readout/features_train_groups.pt
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare
from experiments.extract_readout_features import box_feat

SHARD_DIR = ROOT / "data/raw/gqa/train_all_images"
STAGES = ["S1", "S6", "S12"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(records):
    want = {r["image_id"] for r in records}
    out = {}
    t0 = time.time()
    for k, path in enumerate(sorted(SHARD_DIR.glob("*.parquet"))):
        ids = pq.read_table(path, columns=["id"])["id"].to_pylist()
        idx = [i for i, x in enumerate(ids) if x in want]
        if not idx:
            continue
        tbl = pq.read_table(path, columns=["id", "image"])
        for i in idx:
            row = tbl.slice(i, 1).to_pylist()[0]
            out[row["id"]] = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        print(f"  shard {k + 1}: {len(idx)} images ({len(out)}/{len(want)} total, "
              f"{time.time() - t0:.0f}s)")
        if len(out) == len(want):
            break
    missing = want - set(out)
    if missing:
        print(f"  missing {len(missing)} images; using local fixed dir for those")
        from experiments.common import load_images as load_fixed
        out.update(load_fixed(sorted(missing)))
    return out


def forward_capture(model, inputs, yes_ids, no_ids):
    cap = {}
    handles = []

    def out_hook(name):
        def f(mod, inp, out):
            t = out.last_hidden_state if hasattr(out, "last_hidden_state") else (
                out[0] if isinstance(out, (tuple, list)) else out
            )
            cap[name] = t.detach().to("cpu", torch.float16)
        return f

    handles.append(model.model.connector.register_forward_hook(out_hook("S1")))
    for i in (6, 12):
        def pre(layer_idx):
            def f(mod, args, kwargs):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cap[f"S{layer_idx}"] = x.detach().to("cpu", torch.float16)
            return f
        handles.append(model.model.text_model.layers[i].register_forward_pre_hook(
            pre(i), with_kwargs=True))
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        margin = float(torch.logsumexp(lp[:, yes_ids], -1) - torch.logsumexp(lp[:, no_ids], -1))
    finally:
        for h in handles:
            h.remove()
    return cap, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/train_pairs_groups.jsonl")
    ap.add_argument("--out", default="results/readout/features_train_groups.pt")
    args = ap.parse_args()

    records = read_jsonl(ROOT / args.pairs)
    print(f"{len(records)} records, {len({r['image_id'] for r in records})} images")
    images = load_images(records)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    print(f"{len(images)} images ready")

    arrays = {k: [] for k in ["S1", "S6", "S12", "margin"]}
    examples = []
    t0 = time.time()
    for i, r in enumerate(records):
        img = images[r["image_id"]]
        inp = prepare(proc, r["question"], img)
        cap, margin = forward_capture(model, inp, yes_ids, no_ids)
        ids = inp["input_ids"][0]
        vpos = (ids == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(vpos[0]), int(vpos[-1]) + 1
        arrays["S1"].append(cap["S1"][0].clone())
        for s in ("S6", "S12"):
            arrays[s].append(cap[s][0][v0:v1].clone())
        arrays["margin"].append(torch.tensor(margin))
        examples.append({
            "key": f"{r['image_id']}|{r['question']}|0",
            "pair_id": r["pair_id"],
            "group": r["group"],
            "image_id": r["image_id"],
            "question": r["question"],
            "label": r["label"],
            "position": box_feat(r["target"], r["reference"], r["width"], r["height"]),
        })
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            print(f"  {i + 1}/{len(records)} ({time.time() - t0:.0f}s)")

    payload = {k: torch.stack(v) for k, v in arrays.items() if k != "margin"}
    payload["margin"] = torch.stack(arrays["margin"])
    payload["examples"] = examples
    payload["keys"] = [e["key"] for e in examples]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
