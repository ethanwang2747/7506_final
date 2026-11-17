"""Latent source-target direction steering on spatial pairs."""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

GROUPS = [
    "woman|to the left of|man", "man|to the right of|woman",
    "man|to the left of|person", "person|to the right of|man",
    "tree|to the left of|building", "chair|to the left of|table",
]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def layer_input(layer):
    def hook(module, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        return x[:, -1, :].detach().clone()
    return layer.register_forward_pre_hook(hook, with_kwargs=True)


def margin_from_logits(logits, yes_ids, no_ids):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float((torch.logsumexp(lp[:, yes_ids], -1) - torch.logsumexp(lp[:, no_ids], -1)).item())


def capture_last(model, inp, layer_idx, yes_ids, no_ids):
    box = {}
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        box["h"] = x[:, -1, :].detach().clone()
    h = model.model.text_model.layers[layer_idx].register_forward_pre_hook(f, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model(**inp, use_cache=False)
    finally:
        h.remove()
    return box["h"], margin_from_logits(out.logits[:, -1, :], yes_ids, no_ids)


def margin_with_steer(model, inp, layer_idx, d, alpha, mode, yes_ids, no_ids):
    layer = model.model.text_model.layers[layer_idx]
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        if mode == "target":
            x[:, -1, :] += alpha * d
        elif mode == "source":
            x[:, -1, :] -= alpha * d
        elif mode == "all":
            x[:, -1, :] += alpha * d
    h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        return margin_from_logits(out.logits[:, -1, :], yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-pairs", default=str(ROOT / "data/derived/spatial_pairs_dev.jsonl"))
    ap.add_argument("--test-pairs", default=str(ROOT / "data/derived/spatial_pairs_test.jsonl"))
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "results/latent/steering_layer6.json"))
    args = ap.parse_args()

    def select(path):
        rows = [r for r in read_jsonl(path) if r["group"] in GROUPS]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["group"]].append(r)
        out = []
        for g in GROUPS:
            out.extend(by.get(g, [])[: args.pairs_per_group])
        return out

    dev, test = select(args.dev_pairs), select(args.test_pairs)
    print(f"dev {len(dev)} test {len(test)}")
    images = load_images(sorted({r[k] for r in dev + test for k in ("source_image_id", "target_image_id")}))
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    diffs, norms = [], []
    for r in dev:
        hs, _ = capture_last(model, prepare(proc, r["question"], images[r["source_image_id"]]), args.layer, yes_ids, no_ids)
        ht, _ = capture_last(model, prepare(proc, r["question"], images[r["target_image_id"]]), args.layer, yes_ids, no_ids)
        diffs.append((hs - ht).squeeze(0)); norms.append((hs - ht).norm().item())
    d = torch.stack(diffs).mean(0)
    mean_norm = float(np.mean(norms))
    d = d / d.norm()
    rng = np.random.default_rng(0)
    d_rand = torch.tensor(rng.normal(size=d.shape), dtype=d.dtype, device=d.device)
    d_rand = d_rand / d_rand.norm()
    # shuffled direction
    perm = torch.tensor(rng.permutation(len(diffs)))
    d_shuf = torch.stack(diffs)[perm].mean(0)
    d_shuf = d_shuf / d_shuf.norm()
    print("direction norm mean", mean_norm)

    def metrics(per_pair):
        from sklearn.metrics import roc_auc_score
        y, s = [], []
        for p in per_pair:
            y += [1, 0]; s += [p["ms"], p["mt"]]
        return {"pair_acc": float(np.mean([p["ms"] > p["mt"] for p in per_pair])),
                "auc": float(roc_auc_score(y, s)),
                "mean_delta": float(np.mean([p["ms"] - p["mt"] for p in per_pair])),
                "mean_ms": float(np.mean([p["ms"] for p in per_pair])),
                "mean_mt": float(np.mean([p["mt"] for p in per_pair]))}

    results = {}
    for name, direction, mode, alpha in [
        ("clean", None, None, 0.0),
        ("target_plus", d, "target", args.alpha * mean_norm),
        ("source_minus", d, "source", args.alpha * mean_norm),
        ("all_plus", d, "all", args.alpha * mean_norm),
        ("target_random", d_rand, "target", args.alpha * mean_norm),
        ("target_shuffled", d_shuf, "target", args.alpha * mean_norm),
    ]:
        out = []
        for r in test:
            inp_s = prepare(proc, r["question"], images[r["source_image_id"]])
            inp_t = prepare(proc, r["question"], images[r["target_image_id"]])
            if direction is None:
                ms = margin_from_logits(model(**inp_s, use_cache=False).logits[:, -1, :], yes_ids, no_ids)
                mt = margin_from_logits(model(**inp_t, use_cache=False).logits[:, -1, :], yes_ids, no_ids)
            else:
                ms = margin_with_steer(model, inp_s, args.layer, direction, alpha, mode, yes_ids, no_ids) if mode in ("source", "all") else margin_from_logits(model(**inp_s, use_cache=False).logits[:, -1, :], yes_ids, no_ids)
                mt = margin_with_steer(model, inp_t, args.layer, direction, alpha, mode, yes_ids, no_ids) if mode in ("target", "all") else margin_from_logits(model(**inp_t, use_cache=False).logits[:, -1, :], yes_ids, no_ids)
            out.append({"pair_id": r["pair_id"], "group": r["group"], "ms": ms, "mt": mt})
        results[name] = {"metrics": metrics(out), "per_pair": out}
        print(name, results[name]["metrics"])

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": vars(args), "mean_norm": mean_norm, "results": results}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
