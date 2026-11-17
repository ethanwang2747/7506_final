"""Cache frozen SmolVLM2 features for lightweight readout experiments.

Each cached example is one (image, question) forward. Features are stored per
stage so that readers can be trained without re-running the backbone.

Examples:
    python experiments/extract_readout_features.py \
        --pairs data/derived/spatial_pairs_dev_strict.jsonl \
        --out results/readout/features_dev_strict.pt

    python experiments/extract_readout_features.py \
        --pairs data/derived/spatial_pairs_test_strict.jsonl \
        --groups "woman|to the left of|man,..." --max-per-group 40 \
        --out results/readout/features_test_core.pt --flip
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

CONFIG = ROOT / "data/derived/experiment_config.json"
STAGES = ["S1", "S6", "S12"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_records(records, groups, max_per_group):
    if groups:
        wanted = set(groups)
        records = [r for r in records if r["group"] in wanted]
    if max_per_group <= 0:
        return records
    counts = {}
    out = []
    for r in records:
        n = counts.get(r["group"], 0)
        if n >= max_per_group:
            continue
        counts[r["group"]] = n + 1
        out.append(r)
    return out


def box_feat(obj, ref, W, H):
    def one(o):
        x, y, w, h = o["bbox"]
        return [((x + w / 2) / W), ((y + h / 2) / H), w / W, h / H]

    a, b = one(obj), one(ref)
    return a + b + [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]


def mirror_box(box, W):
    x, y, w, h = box
    return [W - (x + w), y, w, h]


def pooled_s0(x):
    # x: (729, 1152) in row-major 27x27 order; average each 3x3 block to 9x9.
    d = x.shape[-1]
    return x.reshape(9, 3, 9, 3, d).mean(dim=(1, 3))


def forward_capture(model, inputs, yes_ids, no_ids):
    cap = {}
    handles = []

    def out_hook(name):
        def f(mod, inp, out):
            t = out.last_hidden_state if hasattr(out, "last_hidden_state") else (
                out[0] if isinstance(out, (tuple, list)) else out
            )
            cap[name] = t.detach()

        return f

    handles.append(model.model.vision_model.post_layernorm.register_forward_hook(out_hook("S0")))
    handles.append(model.model.connector.register_forward_hook(out_hook("S1")))

    for i in (0, 6, 12):
        def pre(layer_idx):
            def f(mod, args, kwargs):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cap["E" if layer_idx == 0 else f"S{layer_idx}"] = x.detach()

            return f

        handles.append(model.model.text_model.layers[i].register_forward_pre_hook(pre(i), with_kwargs=True))

    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[:, -1, :]
        lp = torch.log_softmax(logits.float(), dim=-1)
        yes = torch.logsumexp(lp[:, yes_ids], dim=-1) - torch.logsumexp(lp[:, no_ids], dim=-1)
    finally:
        for h in handles:
            h.remove()
    return cap, float(yes.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=None, help="comma-separated group filter")
    ap.add_argument("--max-per-group", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["pairs", "flip"], default="pairs")
    ap.add_argument("--only-flipped", action="store_true", help="in flip mode, skip the original image")
    ap.add_argument("--flip-role", choices=["source", "target"], default="source")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",") if args.groups else None
    records = select_records(read_jsonl(Path(args.pairs)), groups, args.max_per_group)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no records selected")

    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)

    items = []
    examples = []
    seen_keys = set()

    def add_item(key, image_id, question, flip):
        if key not in seen_keys:
            seen_keys.add(key)
            items.append((key, image_id, question, flip))

    for r in records:
        rel = r["group"].split("|")[1]
        if args.mode == "pairs":
            for role, image_id in [("source", r["source_image_id"]), ("target", r["target_image_id"])]:
                key = f"{image_id}|{r['question']}|0"
                add_item(key, image_id, r["question"], False)
                g = sg[image_id]
                obj = r[f"{role}_target"]
                ref = r[f"{role}_reference"]
                examples.append({
                    "key": key,
                    "pair_id": r["pair_id"],
                    "group": r["group"],
                    "split": r.get("split", ""),
                    "role": role,
                    "label": 1 if role == "source" else 0,
                    "is_flip": 0,
                    "position": box_feat(obj, ref, g["width"], g["height"]),
                })
        else:
            role = args.flip_role
            image_id = r[f"{role}_image_id"]
            g = sg[image_id]
            base_label = 1 if role == "source" else 0
            for is_flip in ((1,) if args.only_flipped else (0, 1)):
                key = f"{image_id}|{r['question']}|{is_flip}"
                add_item(key, image_id, r["question"], bool(is_flip))
                obj = r[f"{role}_target"]
                ref = r[f"{role}_reference"]
                if is_flip:
                    obj = {**obj, "bbox": mirror_box(obj["bbox"], g["width"])}
                    ref = {**ref, "bbox": mirror_box(ref["bbox"], g["width"])}
                horizontal = ("left" in rel) or ("right" in rel)
                label = (1 - base_label) if (is_flip and horizontal) else base_label
                examples.append({
                    "key": key,
                    "pair_id": r["pair_id"],
                    "group": r["group"],
                    "split": r.get("split", ""),
                    "role": f"{role}_flipped" if is_flip else role,
                    "label": label,
                    "is_flip": int(is_flip),
                    "position": box_feat(obj, ref, g["width"], g["height"]),
                })

    print(f"{args.mode}: {len(records)} pairs, {len(items)} unique forwards, {len(examples)} examples")
    image_ids = sorted({it[1] for it in items})
    images = load_images(image_ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    arrays = {k: [] for k in ["S0_mean", "S1", "S6", "S12", "E_mean", "E_last",
                              "T6", "T12", "H6", "H12", "margin"]}
    meta = []
    t0 = time.time()
    for i, (key, image_id, question, flip) in enumerate(items):
        img = images[image_id]
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        inp = prepare(proc, question, img)
        cap, margin = forward_capture(model, inp, yes_ids, no_ids)

        ids = inp["input_ids"][0]
        mask = inp["attention_mask"][0].bool()
        img_mask = ids == model.config.image_token_id
        vpos = img_mask.nonzero().flatten()
        v0, v1 = int(vpos[0]), int(vpos[-1]) + 1
        text_mask = mask & ~img_mask
        last = int(mask.nonzero()[-1])

        e = cap["E"][0]
        arrays["S0_mean"].append(pooled_s0(cap["S0"][0]).mean(dim=(0, 1)).to("cpu", torch.float16))
        arrays["S1"].append(cap["S1"][0].to("cpu", torch.float16))
        arrays["E_mean"].append(e[text_mask].mean(dim=0).to("cpu", torch.float16))
        arrays["E_last"].append(e[last].to("cpu", torch.float16))
        for stage in ("6", "12"):
            x = cap[f"S{stage}"][0]
            arrays[f"S{stage}"].append(x[v0:v1].to("cpu", torch.float16))
            arrays[f"T{stage}"].append(x[text_mask].mean(dim=0).to("cpu", torch.float16))
            arrays[f"H{stage}"].append(x[last].to("cpu", torch.float16))
        arrays["margin"].append(torch.tensor(margin))
        meta.append({"key": key, "image_id": image_id, "question": question, "flip": int(flip),
                     "width": sg[image_id]["width"], "height": sg[image_id]["height"]})
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")

    payload = {k: torch.stack(v) for k, v in arrays.items() if v}
    payload["keys"] = [it[0] for it in items]
    payload["meta"] = meta
    payload["examples"] = examples
    payload["stages"] = ["S0_mean", "S1", "S6", "S12"]
    payload["model"] = str(getattr(model.config, "_name_or_path", ""))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
