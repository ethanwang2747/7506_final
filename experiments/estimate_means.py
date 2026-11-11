import argparse, collections, json
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor
from experiments.e0_validate import capture, prepare

CONFIG = ROOT / "data/derived/experiment_config.json"
STAGES = ["S0", "S1", "S6", "S12", "S18"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--splitting", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    rows = []
    for task in ["spatial", "semantic"]:
        groups = cfg["tasks"][task]["core_groups"] + cfg["tasks"][task]["exploratory_groups"]
        for r in read_jsonl(ROOT / cfg["data"][f"{task}_pairs_dev"]):
            if r["group"] in groups:
                rows.append(r)
    if args.max_pairs:
        rows = rows[:args.max_pairs]
    tasks = {}
    for r in rows:
        tasks.setdefault((r["source_image_id"], r["question"]), {"image_id": r["source_image_id"], "question": r["question"]})
    tasks = list(tasks.values())
    print(f"mean estimation: {len(rows)} pairs, {len(tasks)} unique source items")

    images = load_images([t["image_id"] for t in tasks])
    proc = load_processor(splitting=bool(args.splitting))
    model = load_model(attn="sdpa")

    sums = {s: None for s in STAGES}
    counts = collections.Counter()
    for idx, t in enumerate(tasks):
        inp = prepare(proc, t["question"], images[t["image_id"]])
        pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        cap = capture(model, inp, [6, 12, 18])
        for s in STAGES:
            x = cap[s]
            if s in ("S0", "S1"):
                v = x.reshape(-1, x.shape[-1])
            else:
                v = x[0, v0:v1, :]
            v = v.float()
            if sums[s] is None:
                sums[s] = torch.zeros(v.shape[-1], dtype=torch.float64, device=v.device)
            sums[s] += v.sum(dim=0)
            counts[s] += v.shape[0]
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{len(tasks)}")

    means = {s: (sums[s] / counts[s]).cpu() for s in STAGES}
    out = Path(args.out) if args.out else ROOT / ("data/derived/mean_vectors_split.pt" if args.splitting else "data/derived/mean_vectors.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"means": means, "counts": dict(counts), "stages": STAGES, "n_items": len(tasks)}, out)
    print("saved", out)
    for s in STAGES:
        print(f"  {s}: mean shape {tuple(means[s].shape)}, tokens {counts[s]}")


if __name__ == "__main__":
    main()
