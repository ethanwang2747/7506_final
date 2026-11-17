"""Causal attention-head ablation/amplification on spatial pairs."""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare

GROUPS = [
    "woman|to the left of|man", "man|to the right of|woman",
    "man|to the left of|person", "person|to the right of|man",
    "tree|to the left of|building", "chair|to the left of|table",
]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_metrics(per_pair):
    acc = np.mean([p["ms"] > p["mt"] for p in per_pair])
    y, s = [], []
    for p in per_pair:
        y += [1, 0]
        s += [p["ms"], p["mt"]]
    from sklearn.metrics import roc_auc_score
    return {"pair_acc": float(acc), "auc": float(roc_auc_score(y, s)),
            "mean_delta": float(np.mean([p["ms"] - p["mt"] for p in per_pair]))}


def run_condition(model, proc, pairs, images, yes_ids, no_ids, heads_by_layer, scale):
    handles = []
    for layer, heads in heads_by_layer.items():
        o_proj = model.model.text_model.layers[layer].self_attn.o_proj
        head_dim = getattr(model.model.text_model.layers[layer].self_attn, "head_dim", 64)

        def hook(module, args, heads=heads, head_dim=head_dim, scale=scale):
            x = args[0].clone()
            for h in heads:
                x[..., h * head_dim:(h + 1) * head_dim] *= scale
            return (x,)

        handles.append(o_proj.register_forward_pre_hook(hook))
    out = []
    try:
        for r in pairs:
            ms = margin(model, prepare(proc, r["question"], images[r["source_image_id"]]), yes_ids, no_ids)
            mt = margin(model, prepare(proc, r["question"], images[r["target_image_id"]]), yes_ids, no_ids)
            out.append({"pair_id": r["pair_id"], "group": r["group"], "ms": ms, "mt": mt})
    finally:
        for h in handles:
            h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data/derived/spatial_pairs_dev.jsonl"))
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/attention/head_intervention_dev.json"))
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.pairs) if r["group"] in GROUPS]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    pairs = []
    for g in GROUPS:
        pairs.extend(by.get(g, [])[: args.pairs_per_group])
    print(f"{len(pairs)} pairs")

    images = load_images(sorted({r["source_image_id"] for r in pairs} |
                                {r["target_image_id"] for r in pairs}))
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    top = {6: [3, 9], 12: [24, 6]}
    rng = np.random.default_rng(0)
    random_heads = {6: [int(x) for x in rng.choice(32, 2, replace=False)],
                    12: [int(x) for x in rng.choice(32, 2, replace=False)]}
    all6 = {6: list(range(32))}
    conditions = {
        "clean": ({}, 1.0),
        "ablate_top": (top, 0.0),
        "amplify_top": (top, args.scale),
        "ablate_random": (random_heads, 0.0),
        "amplify_random": (random_heads, args.scale),
        "ablate_all_layer6": (all6, 0.0),
    }

    results = {}
    for name, (heads, scale) in conditions.items():
        out = run_condition(model, proc, pairs, images, yes_ids, no_ids, heads, scale)
        results[name] = {"metrics": pair_metrics(out), "per_pair": out}
        print(name, results[name]["metrics"])

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": vars(args), "top_heads": top,
                                "random_heads": random_heads, "results": results}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
