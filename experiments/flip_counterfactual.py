import argparse, collections, json, math, random, statistics
from pathlib import Path

import torch
from PIL import Image

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import capture, margin, prepare
from experiments.e2_ablation import pick_background, unique_objects
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors.pt"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
STAGES = ["S1", "S6"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mirror_tokens(tokens, grid=9, axis="horizontal"):
    out = []
    for t in tokens:
        r, c = divmod(t, grid)
        if axis == "horizontal":
            out.append(r * grid + (grid - 1 - c))
        else:
            out.append((grid - 1 - r) * grid + c)
    return sorted(out)


def expand_to_27(tokens):
    out = []
    for t in tokens:
        r, c = divmod(t, 9)
        for dr in range(3):
            for dc in range(3):
                out.append((3 * r + dr) * 27 + 3 * c + dc)
    return out


def replace_tokens(x, idx, mu, offset=0):
    x[:, [offset + i for i in idx], :] = mu.to(dtype=x.dtype, device=x.device)


def patch_s1_region(model, inp, tgt_tokens, src_s1, src_tokens, yes_ids, no_ids):
    def f(mod, args, out):
        x = out.clone()
        x[:, tgt_tokens, :] = src_s1[:, src_tokens, :]
        return x
    h = model.model.connector.register_forward_hook(f)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def patch_s6_region(model, inp, tgt_tokens, src_s6, src_tokens, v0, yes_ids, no_ids):
    layer = model.model.text_model.layers[6]
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        x[:, [v0 + t for t in tgt_tokens], :] = src_s6[:, [v0 + t for t in src_tokens], :]
    h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def patch_full(model, inp, stage, src_tensor, v0, yes_ids, no_ids):
    if stage == "S1":
        h = model.model.connector.register_forward_hook(lambda m, i, o: src_tensor)
        try:
            return margin(model, inp, yes_ids, no_ids)
        finally:
            h.remove()
    layer = model.model.text_model.layers[6]
    span = src_tensor.shape[1] - v0
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        x[:, v0:v0 + span, :] = src_tensor[:, v0:v0 + span, :]
    h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def ablate(model, inp, stage, tokens, mu, v0, yes_ids, no_ids):
    if stage == "S1":
        def f(mod, args, out):
            x = out.clone()
            replace_tokens(x, tokens, mu)
            return x
        h = model.model.connector.register_forward_hook(f)
    else:
        layer = model.model.text_model.layers[6]
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            replace_tokens(x, tokens, mu, offset=v0)
        h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", default="chair|to the left of|table,woman|to the left of|man,man|to the left of|person")
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--axis", choices=["horizontal", "vertical"], default="horizontal")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", default=str(ROOT / "results/flip/flip_counterfactual.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",")
    path = Path(args.pairs) if args.pairs else ROOT / cfg["data"][f"spatial_pairs_{args.split}"]
    rows = [r for r in read_jsonl(path) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        seen = set()
        for r in by_group.get(g, []):
            if r["source_image_id"] in seen:
                continue
            seen.add(r["source_image_id"])
            selected.append(r)
            if len([x for x in selected if x["group"] == g]) >= args.pairs_per_group:
                break
    print(f"flip counterfactual: {len(selected)} source images")

    with open(SG_PATH) as f:
        sg = json.load(f)
    means = torch.load(MEANS, map_location="cpu")["means"]
    images = load_images([r["source_image_id"] for r in selected])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        src_id = r["source_image_id"]
        img = images[src_id]
        flip = img.transpose(Image.FLIP_LEFT_RIGHT if args.axis == "horizontal" else Image.FLIP_TOP_BOTTOM)
        g = sg[src_id]
        W, H = g["width"], g["height"]
        tgt = r["source_target"]["tokens_9"]
        ref = r["source_reference"]["tokens_9"] if "source_reference" in r else []
        tgt_f = mirror_tokens(tgt, axis=args.axis)
        ref_f = mirror_tokens(ref, axis=args.axis)
        objs = unique_objects(g, set(OBJECT_VOCAB))
        used = {r["source_target"]["name"]} | ({r["source_reference"]["name"]} if "source_reference" in r else set())
        cands = sorted([o for n, o in objs.items() if n not in used], key=lambda o: (abs(len(o["tokens_9"]) - len(tgt)), o["name"]))
        irr = cands[0]["tokens_9"] if cands else []
        irr_f = mirror_tokens(irr, axis=args.axis)
        rng = random.Random(f"{r['pair_id']}:flip")
        bg = pick_background(tgt, set(tgt) | set(ref) | set(irr), len(tgt), seed=f"{r['pair_id']}:bg")
        bg_f = mirror_tokens(bg, axis=args.axis) if bg else []
        rand_pool = [i for i in range(81) if i not in set(tgt) | set(ref) | set(irr)]
        n_rand = min(len(tgt), len(rand_pool))
        rand_src = sorted(rng.sample(rand_pool, n_rand)) if n_rand >= 1 else []
        rand_tgt = mirror_tokens(rand_src, axis=args.axis)

        orig_in = prepare(proc, r["question"], img)
        flip_in = prepare(proc, r["question"], flip)
        pos = (orig_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        m_o = margin(model, orig_in, yes_ids, no_ids)
        m_f = margin(model, flip_in, yes_ids, no_ids)
        cap = capture(model, orig_in, [6])

        rec = {
            "pair_id": r["pair_id"], "group": r["group"], "question": r["question"], "image_id": src_id,
            "m_original": m_o, "m_flipped": m_f, "delta": m_o - m_f,
            "clean_rank": m_o > m_f,
            "ablation": {}, "patching": {},
        }
        regions_o = {"target": tgt, "reference": ref, "irrelevant": irr, "background": bg}
        regions_f = {"target": tgt_f, "reference": ref_f, "irrelevant": irr_f, "background": bg_f}
        for stage in STAGES:
            for region in ["target", "reference", "irrelevant", "background"]:
                if not regions_o[region]:
                    continue
                mo = ablate(model, orig_in, stage, regions_o[region], means[stage], v0, yes_ids, no_ids)
                mf = ablate(model, flip_in, stage, regions_f[region], means[stage], v0, yes_ids, no_ids)
                rec["ablation"][f"{stage}:{region}"] = {"orig_delta": mo - m_o, "flip_delta": mf - m_f, "orig_margin": mo, "flip_margin": mf}
            # full patching source -> flipped
            mp = patch_full(model, flip_in, stage, cap[stage], v0, yes_ids, no_ids)
            denom = m_o - m_f
            rec["patching"][f"{stage}:full"] = {"margin": mp, "R": (mp - m_f) / denom if abs(denom) > 1e-8 else None}
            # regional patching source -> flipped
            for region in ["target", "reference", "irrelevant"]:
                if not regions_o[region]:
                    continue
                mp = patch_s1_region(model, flip_in, regions_f[region], cap["S1"], regions_o[region], yes_ids, no_ids) if stage == "S1" else patch_s6_region(model, flip_in, regions_f[region], cap["S6"], regions_o[region], v0, yes_ids, no_ids)
                rec["patching"][f"{stage}:{region}"] = {"margin": mp, "R": (mp - m_f) / denom if abs(denom) > 1e-8 else None}
            if rand_src:
                mp = patch_s1_region(model, flip_in, rand_tgt, cap["S1"], rand_src, yes_ids, no_ids) if stage == "S1" else patch_s6_region(model, flip_in, rand_tgt, cap["S6"], rand_src, v0, yes_ids, no_ids)
                rec["patching"][f"{stage}:random"] = {"margin": mp, "R": (mp - m_f) / denom if abs(denom) > 1e-8 else None}
        per_pair.append(rec)
        if (idx + 1) % 5 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp), "auc_clean": sum(r["clean_rank"] for r in pp) / len(pp),
               "mean_delta": statistics.mean(r["delta"] for r in pp)}
        for stage in STAGES:
            # after target ablation, how well do original and flipped still rank?
            auc_after = sum(r["ablation"][f"{stage}:target"]["orig_margin"] > r["ablation"][f"{stage}:target"]["flip_margin"] for r in pp) / len(pp)
            row[f"{stage}:auc_after_target_ablation"] = auc_after
            for region in ["target", "reference", "irrelevant", "background"]:
                key = f"{stage}:{region}"
                vals = [r["ablation"][key] for r in pp if key in r["ablation"]]
                if vals:
                    row[f"{stage}:{region}"] = {
                        "orig_delta_mean": statistics.mean(v["orig_delta"] for v in vals),
                        "flip_delta_mean": statistics.mean(v["flip_delta"] for v in vals),
                    }
            for key in ["full", "target", "reference", "irrelevant", "random"]:
                k = f"{stage}:{key}"
                vals = [r["patching"][k]["R"] for r in pp if k in r["patching"] and r["patching"][k]["R"] is not None]
                row[f"{stage}:patch_{key}"] = {"median_R": statistics.median(vals) if vals else None, "n": len(vals)}
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']} AUC={r['auc_clean']:.2f} delta={r['mean_delta']:+.2f}")
        for stage in STAGES:
            print(f"    {stage}: auc_after_target_abl={r[f'{stage}:auc_after_target_ablation']:.2f} "
                  f"patch_full={r[f'{stage}:patch_full']['median_R']} patch_target={r[f'{stage}:patch_target']['median_R']} "
                  f"patch_random={r[f'{stage}:patch_random']['median_R']}")


if __name__ == "__main__":
    main()
