"""Cache last-token hidden states at selected text layers for the side probe.

For every (image, question) example we store the hidden state at the final
prompt token entering layers 12, 16, 18, 19, 20 and 23, plus the model's own
yes/no margin and the normalized boxes of the target and reference objects.

The side probe then asks: at the layer where the answer becomes readable, is
the absolute side of each object (left/right half, top/bottom half) linearly
decodable? If yes, the model has the ingredients but does not combine them.

Output:
    results/diagnostics/side_features_test_core.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

LAYERS = [12, 16, 18, 19, 20, 23]
GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records, groups, max_per_group):
    counts, out = {}, []
    for r in records:
        if r["group"] not in groups:
            continue
        n = counts.get(r["group"], 0)
        if max_per_group and n >= max_per_group:
            continue
        counts[r["group"]] = n + 1
        out.append(r)
    return out


def side(box, W, H):
    x, y, w, h = box
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    return {"cx": cx, "cy": cy, "left": float(cx < 0.5), "top": float(cy < 0.5),
            "w": w / W, "h": h / H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--out", default="results/diagnostics/side_features_test_core.pt")
    ap.add_argument("--max-per-group", type=int, default=40)
    args = ap.parse_args()

    rows = select(read_jsonl(ROOT / args.pairs), set(GROUPS), args.max_per_group)
    if not rows:
        raise SystemExit("no pairs selected")
    print(f"{len(rows)} pairs, {len({r['group'] for r in rows})} groups")

    with open(ROOT / "data/derived/experiment_config.json") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)

    image_ids = sorted({r[f"{role}_image_id"] for r in rows for role in ("source", "target")})
    images = load_images(image_ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    caps = {}

    def hook(i):
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            caps[i] = x[:, -1, :].detach().to("cpu", torch.float16)
        return f

    handles = [model.model.text_model.layers[i].register_forward_pre_hook(hook(i), with_kwargs=True)
               for i in LAYERS]

    feats = {i: [] for i in LAYERS}
    margins, meta = [], []
    try:
        with torch.no_grad():
            for k, r in enumerate(rows):
                for role in ("source", "target"):
                    iid = r[f"{role}_image_id"]
                    g = sg[iid]
                    W, H = g["width"], g["height"]
                    inp = prepare(proc, r["question"], images[iid])
                    out = model(**inp, use_cache=False)
                    logits = out.logits[:, -1, :].float()
                    lp = torch.log_softmax(logits, dim=-1)
                    margins.append(float(torch.logsumexp(lp[:, yes_ids], -1)
                                         - torch.logsumexp(lp[:, no_ids], -1)))
                    for i in LAYERS:
                        feats[i].append(caps[i][0])
                    tgt = side(r[f"{role}_target"]["bbox"], W, H)
                    ref = side(r[f"{role}_reference"]["bbox"], W, H)
                    meta.append({
                        "pair_id": r["pair_id"], "group": r["group"], "role": role,
                        "image_id": iid, "label": 1 if role == "source" else 0,
                        "target": tgt, "ref": ref,
                    })
                if (k + 1) % 20 == 0 or k + 1 == len(rows):
                    print(f"  {k + 1}/{len(rows)}")
    finally:
        for h in handles:
            h.remove()

    payload = {f"H{i}": torch.stack(feats[i]) for i in LAYERS}
    payload["margin"] = torch.tensor(margins, dtype=torch.float32)
    payload["meta"] = meta
    payload["layers"] = LAYERS
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    print(f"saved {dst} ({dst.stat().st_size / 1e6:.1f} MB, {len(meta)} examples)")


if __name__ == "__main__":
    main()
