import argparse, collections, json, math, random, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import token_box_9

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors_split.pt"
STAGE = "S1"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def resized_dims(W, H):
    if W >= H:
        W1, H1 = 1536.0, 1536.0 * H / W
    else:
        H1, W1 = 1536.0, 1536.0 * W / H
    if W1 >= H1:
        W2 = math.ceil(W1 / 384) * 384
        H2 = int(W2 / (W1 / H1))
        H2 = math.ceil(H2 / 384) * 384
    else:
        H2 = math.ceil(H1 / 384) * 384
        W2 = int(H2 * (W1 / H1))
        W2 = math.ceil(W2 / 384) * 384
    return int(W2), int(H2)


def crop_tokens(bbox, W, H, W2, H2, rows, cols):
    x, y, w, h = bbox
    x0, y0 = x * W2 / W, y * H2 / H
    x1, y1 = (x + w) * W2 / W, (y + h) * H2 / H
    out = {}
    for r in range(rows):
        for c in range(cols):
            lx0 = max(0.0, x0 - c * 384)
            lx1 = min(384.0, x1 - c * 384)
            ly0 = max(0.0, y0 - r * 384)
            ly1 = min(384.0, y1 - r * 384)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            tx0 = max(0, min(8, math.floor(lx0 / 384 * 9)))
            tx1 = max(0, min(8, math.ceil(lx1 / 384 * 9) - 1))
            ty0 = max(0, min(8, math.floor(ly0 / 384 * 9)))
            ty1 = max(0, min(8, math.ceil(ly1 / 384 * 9) - 1))
            toks = [rr * 9 + cc for rr in range(ty0, ty1 + 1) for cc in range(tx0, tx1 + 1)]
            if toks:
                out[r * cols + c] = toks
    return out


def patch_s1(model, inp, block_tokens, n_blocks, mu, yes_ids, no_ids):
    # block_tokens: dict block_idx -> list of 9x9 token ids
    def f(mod, args, out):
        x = out.clone()
        for b, toks in block_tokens.items():
            x[b, toks, :] = mu.to(dtype=x.dtype, device=x.device)
        return x
    h = model.model.connector.register_forward_hook(f)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def patch_full(model, inp, n_blocks, mu, yes_ids, no_ids):
    def f(mod, args, out):
        x = out.clone()
        x[:, :, :] = mu.to(dtype=x.dtype, device=x.device)
        return x
    h = model.model.connector.register_forward_hook(f)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="woman|to the left of|man,chair|to the left of|table")
    ap.add_argument("--pairs-per-group", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/e4/crop_cumulative.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",")
    rows = [r for r in read_jsonl(ROOT / cfg["data"]["spatial_pairs_test"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"crop cumulative: {len(selected)} pairs")

    means = torch.load(MEANS, map_location="cpu")["means"]
    images = load_images([r["source_image_id"] for r in selected])
    proc = load_processor(splitting=True)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        img = images[r["source_image_id"]]
        W, H = img.size
        W2, H2 = resized_dims(W, H)
        rows_n, cols_n = H2 // 384, W2 // 384
        n_crops = rows_n * cols_n
        n_blocks = n_crops + 1
        crops = crop_tokens(r["source_target"]["bbox"], W, H, W2, H2, rows_n, cols_n)
        global_tokens = token_box_9(r["source_target"], W, H)
        inp = prepare(proc, r["question"], img)
        pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        if (v1 - v0) // 81 != n_blocks:
            print(f"skip {r['pair_id']}: expected {n_blocks} blocks, got {(v1-v0)//81}")
            continue
        m_clean = margin(model, inp, yes_ids, no_ids)
        rec = {"pair_id": r["pair_id"], "group": r["group"], "n_blocks": n_blocks,
               "n_crops": n_crops, "target_crops": sorted(crops), "m_clean": m_clean, "delta": {}}
        # global target
        rec["delta"]["global"] = patch_s1(model, inp, {n_blocks - 1: global_tokens}, n_blocks, means[STAGE], yes_ids, no_ids) - m_clean
        # cumulative crops
        acc = {}
        for k, b in enumerate(sorted(crops)):
            acc[b] = crops[b]
            m = patch_s1(model, inp, acc, n_blocks, means[STAGE], yes_ids, no_ids)
            rec["delta"][f"crop_cum_{k+1}"] = m - m_clean
        # all crops, and global + crops
        rec["delta"]["crops_all"] = patch_s1(model, inp, crops, n_blocks, means[STAGE], yes_ids, no_ids) - m_clean
        all_t = dict(crops); all_t[n_blocks - 1] = global_tokens
        rec["delta"]["global_plus_crops"] = patch_s1(model, inp, all_t, n_blocks, means[STAGE], yes_ids, no_ids) - m_clean
        rec["delta"]["full"] = patch_full(model, inp, n_blocks, means[STAGE], yes_ids, no_ids) - m_clean
        # random crop regions matched to global target token count
        rng = random.Random(f"{r['pair_id']}:crop")
        rand = {}
        for b in sorted(crops):
            pool = [t for t in range(81) if t not in crops[b]]
            n = min(len(crops[b]), len(pool))
            if n:
                rand[b] = sorted(rng.sample(pool, n))
        if rand:
            rec["delta"]["random_crops"] = patch_s1(model, inp, rand, n_blocks, means[STAGE], yes_ids, no_ids) - m_clean
        per_pair.append(rec)
        if (idx + 1) % 2 == 0:
            print(f"  {idx+1}/{len(selected)}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp), "n_blocks_mean": statistics.mean(r["n_blocks"] for r in pp),
               "n_target_crops_mean": statistics.mean(len(r["target_crops"]) for r in pp)}
        for key in ["global", "crops_all", "global_plus_crops", "full", "random_crops"]:
            vals = [r["delta"][key] for r in pp if key in r["delta"]]
            if vals:
                row[key] = {"mean": statistics.mean(vals), "median": statistics.median(vals), "n": len(vals)}
        # cumulative curve: average at each k for pairs that have it
        maxk = max((len(r["target_crops"]) for r in pp), default=0)
        curve = []
        for k in range(1, maxk + 1):
            vals = [r["delta"].get(f"crop_cum_{k}") for r in pp if f"crop_cum_{k}" in r["delta"]]
            curve.append({"k": k, "mean": statistics.mean(vals) if vals else None, "n": len(vals)})
        row["cumulative"] = curve
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']} blocks={r['n_blocks_mean']:.1f} target_crops={r['n_target_crops_mean']:.1f}")
        for k in ["global", "crops_all", "global_plus_crops", "full", "random_crops"]:
            if k in r:
                print(f"    {k}: mean={r[k]['mean']:+.2f} median={r[k]['median']:+.2f}")
        print("    cum:", [(c["k"], round(c["mean"], 2) if c["mean"] is not None else None) for c in r["cumulative"]])


if __name__ == "__main__":
    main()
