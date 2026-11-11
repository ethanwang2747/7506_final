import argparse, collections, json, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import pick_background, token_box_9, unique_objects
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors_split.pt"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
STAGES = ["S1"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def expand_to_27(tokens):
    out = []
    for t in tokens:
        r, c = divmod(t, 9)
        for dr in range(3):
            for dc in range(3):
                out.append((3 * r + dr) * 27 + 3 * c + dc)
    return out


def run(model, inp, stage, region, tokens9, n_blocks, mu, v0, yes_ids, no_ids):
    # Block order in SmolVLMImageProcessor: crops first, global image last.
    global_offset = (n_blocks - 1) * 81
    if stage == "S1":
        def f(mod, args, out):
            x = out.clone()
            if region == "full":
                x[:, :, :] = mu.to(dtype=x.dtype, device=x.device)
            else:
                x[n_blocks - 1, tokens9, :] = mu.to(dtype=x.dtype, device=x.device)
            return x
        h = model.model.connector.register_forward_hook(f)
    else:
        layer = model.model.text_model.layers[int(stage[1:])]
        span = n_blocks * 81
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if region == "full":
                x[:, v0:v0 + span, :] = mu.to(dtype=x.dtype, device=x.device)
            else:
                x[:, [v0 + global_offset + t for t in tokens9], :] = mu.to(dtype=x.dtype, device=x.device)
        h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",") if args.groups else cfg["tasks"][args.task]["core_groups"] + cfg["tasks"][args.task]["exploratory_groups"]
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"{args.task}_pairs_{args.split}"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"splitting E2: {len(selected)} pairs")

    with open(SG_PATH) as f:
        sg = json.load(f)
    means = torch.load(MEANS, map_location="cpu")["means"]
    images = load_images([r["source_image_id"] for r in selected])
    proc = load_processor(splitting=True)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        inp = prepare(proc, r["question"], images[r["source_image_id"]])
        pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        n_blocks = (v1 - v0) // 81
        g = sg[r["source_image_id"]]
        W, H = g["width"], g["height"]
        tgt = token_box_9(r["source_target"], W, H)
        ref = token_box_9(r["source_reference"], W, H) if "source_reference" in r else []
        objs = unique_objects(g, set(OBJECT_VOCAB))
        used = {r["source_target"]["name"]} | ({r["source_reference"]["name"]} if "source_reference" in r else set())
        cands = [o for n, o in objs.items() if n not in used]
        cands.sort(key=lambda o: (abs(len(o["tokens_9"]) - len(tgt)), o["name"]))
        irr = cands[0]["tokens_9"] if cands else []
        bg = pick_background(tgt, set(tgt) | set(ref) | set(irr), len(tgt), seed=f"{r['pair_id']}:bg")
        regions = {"target": tgt, "irrelevant": irr, "background": bg}
        if ref:
            regions["reference"] = ref
        regions["full"] = list(range(81))
        m_clean = margin(model, inp, yes_ids, no_ids)
        rec = {"pair_id": r["pair_id"], "group": r["group"], "n_blocks": n_blocks, "visual_tokens": v1 - v0,
               "m_clean": m_clean, "delta": {}}
        for region, toks in regions.items():
            if not toks:
                continue
            for stage in STAGES:
                m = run(model, inp, stage, region, toks, n_blocks, means[stage], v0, yes_ids, no_ids)
                rec["delta"][f"{stage}:{region}"] = m - m_clean
        per_pair.append(rec)
        if (idx + 1) % 5 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp), "n_blocks_mean": statistics.mean(r["n_blocks"] for r in pp)}
        for stage in STAGES:
            for region in ["target", "reference", "irrelevant", "background", "full"]:
                vals = [r["delta"][f"{stage}:{region}"] for r in pp if f"{stage}:{region}" in r["delta"]]
                if vals:
                    row[f"{stage}:{region}"] = {"mean": statistics.mean(vals), "median": statistics.median(vals), "n": len(vals)}
        summary.append(row)

    out = Path(args.out) if args.out else ROOT / f"results/e2/{args.task}_{args.split}_splitting.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for row in summary:
        print(f"  {row['group']} n={row['n_pairs']} blocks={row['n_blocks_mean']:.1f}")
        for stage in STAGES:
            line = []
            for region in ["target", "reference", "irrelevant", "background", "full"]:
                k = f"{stage}:{region}"
                if k in row:
                    line.append(f"{region}={row[k]['mean']:+.2f}")
            print(f"    {stage}: " + " ".join(line))


if __name__ == "__main__":
    main()
