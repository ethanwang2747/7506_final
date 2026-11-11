import argparse, collections, json, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import run_intervention, token_box_9

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors.pt"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def small_region(tokens):
    if len(tokens) <= 4:
        return sorted(tokens)
    center = tokens[len(tokens) // 2]
    r, c = divmod(center, 9)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 9 and 0 <= cc < 9:
                t = rr * 9 + cc
                if t in tokens:
                    out.append(t)
    out = sorted(set(out))
    if len(out) < 2:
        out = sorted(tokens)[:4]
    return out[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"{args.task}_pairs_{args.split}"]) if r["group"] == args.group]
    rows = rows[:args.pairs_per_group]
    print(f"coverage: {args.task} {args.group} {len(rows)} pairs")

    images = load_images([r["source_image_id"] for r in rows])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    means = torch.load(MEANS, map_location="cpu")["means"]

    per_pair = []
    for r in rows:
        obj = r["source_target"]
        tokens = obj.get("tokens_9") or token_box_9(obj, 384, 384)
        if not tokens:
            continue
        regions = {
            "single": [tokens[len(tokens) // 2]],
            "small": small_region(tokens),
            "object": sorted(tokens),
            "full": list(range(81)),
        }
        inp = prepare(proc, r["question"], images[r["source_image_id"]])
        pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0 = int(pos[0])
        m_clean = margin(model, inp, yes_ids, no_ids)
        rec = {"pair_id": r["pair_id"], "group": r["group"], "m_clean": m_clean, "sizes": {k: len(v) for k, v in regions.items()}, "delta": {}}
        for region, toks in regions.items():
            for stage in ["S1", "S6"]:
                m = run_intervention(model, inp, stage, toks, means[stage], v0, yes_ids, no_ids)
                rec["delta"][f"{stage}:{region}"] = m - m_clean
        per_pair.append(rec)

    summary = []
    for region in ["single", "small", "object", "full"]:
        row = {"region": region}
        for stage in ["S1", "S6"]:
            vals = [r["delta"][f"{stage}:{region}"] for r in per_pair]
            row[stage] = {"mean": statistics.mean(vals), "median": statistics.median(vals), "n": len(vals)}
        summary.append(row)

    out = Path(args.out) if args.out else ROOT / f"results/e2/coverage_{args.task}_{args.group.replace('|', '_').replace(' ', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['region']:8s} n_tokens={per_pair[0]['sizes'][r['region']]:2d} "
              f"S1={r['S1']['mean']:+.2f} S6={r['S6']['mean']:+.2f}")


if __name__ == "__main__":
    main()
