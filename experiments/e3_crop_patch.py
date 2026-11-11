import argparse, collections, json, random, statistics
from pathlib import Path

import torch
from PIL import Image

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import token_box_9
from experiments.e3_regional import map_tokens
from experiments.e4_crop_cumulative import crop_tokens, resized_dims

CONFIG = ROOT / "data/derived/experiment_config.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def capture_s1(model, inp):
    cap = {}
    h = model.model.connector.register_forward_hook(lambda m, i, o: cap.__setitem__("S1", o.detach()))
    with torch.no_grad():
        model(**inp, use_cache=False)
    h.remove()
    return cap["S1"]


def patch_blocks(model, inp, block_tokens, src_s1, src_block_tokens, src_block_index, yes_ids, no_ids):
    def f(mod, args, out):
        x = out.clone()
        for b, tgt_toks in block_tokens.items():
            sb = src_block_index.get(b, b)
            src_toks = src_block_tokens.get(sb)
            if not src_toks:
                continue
            idx = map_tokens(src_toks, tgt_toks)
            x[b, tgt_toks, :] = src_s1[sb, idx, :]
        return x
    h = model.model.connector.register_forward_hook(f)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def patch_full(model, inp, src_s1, yes_ids, no_ids):
    h = model.model.connector.register_forward_hook(lambda m, i, o: src_s1)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="woman|to the left of|man,chair|to the left of|table")
    ap.add_argument("--pairs-per-group", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/e3/crop_patch.json"))
    ap.add_argument("--flip-target", action="store_true")
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
    print(f"crop patch: {len(selected)} pairs")

    images = load_images([r["source_image_id"] for r in selected])
    proc = load_processor(splitting=True)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        img = images[r["source_image_id"]]
        flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        W, H = img.size
        W2, H2 = resized_dims(W, H)
        rows_n, cols_n = H2 // 384, W2 // 384
        n_blocks = rows_n * cols_n + 1
        src_crops = crop_tokens(r["source_target"]["bbox"], W, H, W2, H2, rows_n, cols_n)
        tb = r["source_target"]["bbox"]
        tb_f = [W - (tb[0] + tb[2]), tb[1], tb[2], tb[3]]
        tgt_crops = crop_tokens(tb_f, W, H, W2, H2, rows_n, cols_n)
        src_global = token_box_9(r["source_target"], W, H)
        tgt_global = token_box_9({"bbox": tb_f}, W, H)

        src_in = prepare(proc, r["question"], img)
        if args.flip_target:
            src_crops = crop_tokens(r["source_target"]["bbox"], W, H, W2, H2, rows_n, cols_n)
            tb = r["source_target"]["bbox"]
            tb_f = [W - (tb[0] + tb[2]), tb[1], tb[2], tb[3]]
            tgt_crops = crop_tokens(tb_f, W, H, W2, H2, rows_n, cols_n)
            src_global = token_box_9(r["source_target"], W, H)
            tgt_global = token_box_9({"bbox": tb_f}, W, H)
            tgt_image = flip
        else:
            tgt_image = images[r["target_image_id"]]
        tgt_in = prepare(proc, r["question"], tgt_image)
        pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        if (int(pos[-1]) + 1 - int(pos[0])) // 81 != n_blocks:
            continue
        m_s = margin(model, src_in, yes_ids, no_ids)
        m_t = margin(model, tgt_in, yes_ids, no_ids)
        cap = capture_s1(model, src_in)
        denom = m_s - m_t
        rec = {"pair_id": r["pair_id"], "group": r["group"], "n_blocks": n_blocks,
               "target_crops": sorted(tgt_crops), "m_source": m_s, "m_target": m_t, "delta": denom, "R": {}}

        # global target (last block)
        if src_global and tgt_global:
            m = patch_blocks(model, tgt_in, {n_blocks - 1: tgt_global}, cap, {n_blocks - 1: src_global}, {n_blocks - 1: n_blocks - 1}, yes_ids, no_ids)
            rec["R"]["global"] = (m - m_t) / denom if abs(denom) > 1e-8 else None

        # crop cumulative: pair crop by mirrored column
        acc = {}
        for k, b_t in enumerate(sorted(tgt_crops)):
            r_, c_ = divmod(b_t, cols_n)
            b_s = r_ * cols_n + (cols_n - 1 - c_)
            if b_s in src_crops:
                acc[b_t] = tgt_crops[b_t]
                m = patch_blocks(model, tgt_in, acc, cap, src_crops, {bt: (bt // cols_n) * cols_n + (cols_n - 1 - bt % cols_n) for bt in acc}, yes_ids, no_ids)
                rec["R"][f"crop_cum_{k+1}"] = (m - m_t) / denom if abs(denom) > 1e-8 else None
        if acc:
            m = patch_blocks(model, tgt_in, acc, cap, src_crops, {bt: (bt // cols_n) * cols_n + (cols_n - 1 - bt % cols_n) for bt in acc}, yes_ids, no_ids)
            rec["R"]["crops_all"] = (m - m_t) / denom if abs(denom) > 1e-8 else None
        all_t = dict(acc)
        if src_global and tgt_global:
            all_t[n_blocks - 1] = tgt_global
        if all_t:
            src_map = dict(src_crops)
            src_index = {bt: (bt // cols_n) * cols_n + (cols_n - 1 - bt % cols_n) for bt in acc}
            if src_global:
                src_map[n_blocks - 1] = src_global
                src_index[n_blocks - 1] = n_blocks - 1
            m = patch_blocks(model, tgt_in, all_t, cap, src_map, src_index, yes_ids, no_ids)
            rec["R"]["global_plus_crops"] = (m - m_t) / denom if abs(denom) > 1e-8 else None
        m = patch_full(model, tgt_in, cap, yes_ids, no_ids)
        rec["R"]["full"] = (m - m_t) / denom if abs(denom) > 1e-8 else None

        # random crop control matched to target crop tokens
        rng = random.Random(f"{r['pair_id']}:crop_patch")
        rand = {}
        for b_t, toks in tgt_crops.items():
            pool = [t for t in range(81) if t not in toks]
            n = min(len(toks), len(pool))
            if n:
                rand[b_t] = sorted(rng.sample(pool, n))
        if rand:
            src_rand = {}
            src_index = {}
            for b, t in rand.items():
                sb = (b // cols_n) * cols_n + (cols_n - 1 - b % cols_n)
                src_index[b] = sb
                src_rand[sb] = sorted(rng.sample(range(81), len(t)))
            m = patch_blocks(model, tgt_in, rand, cap, src_rand, src_index, yes_ids, no_ids)
            rec["R"]["random_crops"] = (m - m_t) / denom if abs(denom) > 1e-8 else None
        per_pair.append(rec)
        print(f"  {idx+1}/{len(selected)} {r['group']}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp), "n_blocks": statistics.mean(r["n_blocks"] for r in pp)}
        for key in ["global", "crops_all", "global_plus_crops", "full", "random_crops"]:
            vals = [r["R"][key] for r in pp if key in r["R"] and r["R"][key] is not None]
            row[key] = {"median": statistics.median(vals) if vals else None, "n": len(vals)}
        maxk = max((len([k for k in r["R"] if k.startswith("crop_cum_")]) for r in pp), default=0)
        curve = []
        for k in range(1, maxk + 1):
            vals = [r["R"].get(f"crop_cum_{k}") for r in pp if f"crop_cum_{k}" in r["R"]]
            curve.append({"k": k, "median": statistics.median(vals) if vals else None, "n": len(vals)})
        row["cumulative"] = curve
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']} blocks={r['n_blocks']:.1f}")
        for k in ["global", "crops_all", "global_plus_crops", "full", "random_crops"]:
            print(f"    {k}: median={r[k]['median']}")
        print("    cum:", [(c['k'], c['median']) for c in r['cumulative']])


if __name__ == "__main__":
    main()
