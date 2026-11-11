import argparse, collections, json, math, random
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors.pt"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
STAGES = ["S0", "S1", "S6", "S12", "S18"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def token_box_9(rec, W, H):
    x0 = max(0, min(8, math.floor(rec["bbox"][0] / W * 9)))
    x1 = max(0, min(8, math.ceil((rec["bbox"][0] + rec["bbox"][2]) / W * 9) - 1))
    y0 = max(0, min(8, math.floor(rec["bbox"][1] / H * 9)))
    y1 = max(0, min(8, math.ceil((rec["bbox"][1] + rec["bbox"][3]) / H * 9) - 1))
    return [r * 9 + c for r in range(y0, y1 + 1) for c in range(x0, x1 + 1)]


def expand_to_27(tokens):
    out = []
    for t in tokens:
        r, c = divmod(t, 9)
        for dr in range(3):
            for dc in range(3):
                out.append((3 * r + dr) * 27 + 3 * c + dc)
    return out


def unique_objects(g, vocab):
    W, H = g["width"], g["height"]
    by = collections.defaultdict(list)
    for oid, o in g["objects"].items():
        name = o["name"].strip().lower()
        if name not in vocab or o["w"] <= 1 or o["h"] <= 1:
            continue
        if o["x"] < 0 or o["y"] < 0 or o["x"] + o["w"] > W or o["y"] + o["h"] > H:
            continue
        rec = dict(o)
        rec["object_id"] = oid
        rec["name"] = name
        rec["bbox"] = [o["x"], o["y"], o["w"], o["h"]]
        rec["tokens_9"] = token_box_9(rec, W, H)
        if o["w"] * o["h"] < 0.005 * W * H or len(rec["tokens_9"]) < 2:
            continue
        by[name].append(rec)
    return {n: v[0] for n, v in by.items() if len(v) == 1}


def pick_background(target_tokens, used, n_tokens, seed):
    candidates = [i for i in range(81) if i not in used]
    if len(candidates) < n_tokens:
        return None
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n_tokens))


def replace_tokens(x, idx, mu, offset=0):
    x[:, [offset + i for i in idx], :] = mu.to(dtype=x.dtype, device=x.device)


def run_intervention(model, inp, stage, tokens9, mu, v0, yes_ids, no_ids):
    if stage == "S0":
        idx = expand_to_27(tokens9)
        def f(mod, args, out):
            x = out.clone()
            replace_tokens(x, idx, mu)
            return x
        h = model.model.vision_model.post_layernorm.register_forward_hook(f)
    elif stage == "S1":
        def f(mod, args, out):
            x = out.clone()
            replace_tokens(x, tokens9, mu)
            return x
        h = model.model.connector.register_forward_hook(f)
    else:
        layer = model.model.text_model.layers[int(stage[1:])]
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            replace_tokens(x, tokens9, mu, offset=v0)
        h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = cfg["tasks"][args.task]["core_groups"] + cfg["tasks"][args.task]["exploratory_groups"]
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"{args.task}_pairs_{args.split}"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"{args.task} {args.split}: {len(selected)} pairs, {len(by_group)} groups")

    with open(SG_PATH) as f:
        sg = json.load(f)
    vocab = set(OBJECT_VOCAB)
    means = torch.load(MEANS, map_location="cpu")["means"]

    images = load_images([r["source_image_id"] for r in selected])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        src = r["source_image_id"]
        g = sg[src]
        W, H = g["width"], g["height"]
        tgt_tokens = token_box_9(r["source_target"], W, H)
        ref_tokens = token_box_9(r["source_reference"], W, H) if "source_reference" in r else []
        objs = unique_objects(g, vocab)
        used_names = {r["source_target"]["name"]}
        if "source_reference" in r:
            used_names.add(r["source_reference"]["name"])
        cands = [o for n, o in objs.items() if n not in used_names]
        cands.sort(key=lambda o: (abs(len(o["tokens_9"]) - len(tgt_tokens)), o["name"]))
        irr_tokens = cands[0]["tokens_9"] if cands else []
        used = set(tgt_tokens) | set(ref_tokens) | set(irr_tokens)
        bg_tokens = pick_background(tgt_tokens, used, len(tgt_tokens), seed=f"{r['pair_id']}:bg")
        regions = {"target": tgt_tokens, "irrelevant": irr_tokens, "background": bg_tokens}
        if ref_tokens:
            regions["reference"] = ref_tokens

        inp = prepare(proc, r["question"], images[src])
        pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0 = int(pos[0])
        m_clean = margin(model, inp, yes_ids, no_ids)
        rec = {
            "pair_id": r["pair_id"], "task": args.task, "group": r["group"], "question": r["question"],
            "source_image_id": src, "m_clean": m_clean,
            "regions": {k: v for k, v in regions.items() if v is not None},
            "delta": {},
        }
        for region, toks in regions.items():
            if toks is None:
                continue
            for stage in STAGES:
                m = run_intervention(model, inp, stage, toks, means[stage], v0, yes_ids, no_ids)
                rec["delta"][f"{stage}:{region}"] = m - m_clean
        for stage in STAGES:
            m = run_intervention(model, inp, stage, list(range(81)), means[stage], v0, yes_ids, no_ids)
            rec["delta"][f"{stage}:full"] = m - m_clean
        per_pair.append(rec)
        if (idx + 1) % 5 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    keys = sorted({k for r in per_pair for k in r["delta"]})
    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "task": args.task, "n_pairs": len(pp), "m_clean_mean": sum(r["m_clean"] for r in pp) / len(pp)}
        for k in keys:
            vals = [r["delta"][k] for r in pp if k in r["delta"]]
            if vals:
                row[k] = {"mean": sum(vals) / len(vals), "frac_neg": sum(v < 0 for v in vals) / len(vals), "n": len(vals)}
        summary.append(row)

    out = Path(args.out) if args.out else ROOT / f"results/e2/{args.task}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for row in summary:
        print(f"  {row['group']} n={row['n_pairs']} m={row['m_clean_mean']:+.2f}")
        for stage in STAGES:
            line = []
            for region in ["target", "reference", "irrelevant", "background", "full"]:
                k = f"{stage}:{region}"
                if k in row:
                    line.append(f"{region}={row[k]['mean']:+.2f}")
            print(f"    {stage}: " + " ".join(line))


if __name__ == "__main__":
    main()
