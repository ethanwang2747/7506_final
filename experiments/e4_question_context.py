import argparse, collections, json, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import pick_background, token_box_9, unique_objects
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
MEANS = ROOT / "data/derived/mean_vectors.pt"
STAGES = ["S0", "S1", "S6"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replace_tokens(x, idx, mu, offset=0):
    x[:, [offset + i for i in idx], :] = mu.to(dtype=x.dtype, device=x.device)


def expand_to_27(tokens):
    out = []
    for t in tokens:
        r, c = divmod(t, 9)
        for dr in range(3):
            for dc in range(3):
                out.append((3 * r + dr) * 27 + 3 * c + dc)
    return out


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
    ap.add_argument("--pairs-per-group", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/e4/question_context.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = cfg["tasks"]["spatial"]["core_groups"]
    rows = [r for r in read_jsonl(ROOT / cfg["data"]["spatial_pairs_dev"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    control_by_group = {}
    for i, g in enumerate(groups):
        other = groups[(i + 1) % len(groups)]
        control_by_group[g] = by_group[other][0]["question"]
    print(f"question-context E4: {len(selected)} pairs")

    with open(SG_PATH) as f:
        sg = json.load(f)
    means = torch.load(MEANS, map_location="cpu")["means"]
    images = load_images([r["source_image_id"] for r in selected])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for r in selected:
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

        rec = {"pair_id": r["pair_id"], "group": r["group"], "own_question": r["question"],
               "control_question": control_by_group[r["group"]], "delta": {}}
        for qkind, question in [("own", r["question"]), ("control", control_by_group[r["group"]])]:
            inp = prepare(proc, question, images[r["source_image_id"]])
            pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
            v0 = int(pos[0])
            m_clean = margin(model, inp, yes_ids, no_ids)
            rec[f"m_clean_{qkind}"] = m_clean
            for region, toks in regions.items():
                if not toks:
                    continue
                for stage in STAGES:
                    m = run_intervention(model, inp, stage, toks, means[stage], v0, yes_ids, no_ids)
                    rec["delta"][f"{qkind}:{stage}:{region}"] = m - m_clean
        per_pair.append(rec)

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        rec = {"group": g, "n_pairs": len(pp), "own_question": pp[0]["own_question"], "control_question": pp[0]["control_question"]}
        for stage in STAGES:
            for qkind in ["own", "control"]:
                tv = [r["delta"][f"{qkind}:{stage}:target"] for r in pp if f"{qkind}:{stage}:target" in r["delta"]]
                cv = []
                for r in pp:
                    c = [r["delta"][f"{qkind}:{stage}:{x}"] for x in ["irrelevant", "background"] if f"{qkind}:{stage}:{x}" in r["delta"]]
                    if c:
                        cv.append(sum(c) / len(c))
                if tv and cv:
                    rec[f"{qkind}:{stage}:target_minus_controls"] = statistics.mean(t - c for t, c in zip(tv, cv))
        summary.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']}")
        for stage in STAGES:
            print(f"    {stage}: own={r.get(f'own:{stage}:target_minus_controls', float('nan')):+.2f} "
                  f"control={r.get(f'control:{stage}:target_minus_controls', float('nan')):+.2f}")


if __name__ == "__main__":
    main()
