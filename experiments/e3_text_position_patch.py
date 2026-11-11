import argparse, collections, json, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare

CONFIG = ROOT / "data/derived/experiment_config.json"
LAYERS = [12, 18]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def capture_layers(model, inp, layers):
    cap = {}
    handles = []
    for i in layers:
        def pre(i):
            def f(mod, args, kwargs):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cap[i] = x.detach()
            return f
        handles.append(model.model.text_model.layers[i].register_forward_pre_hook(pre(i), with_kwargs=True))
    with torch.no_grad():
        model(**inp, use_cache=False)
    for h in handles:
        h.remove()
    return cap


def patch_positions(model, inp, layer_idx, positions, src_hs, yes_ids, no_ids):
    layer = model.model.text_model.layers[layer_idx]
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        x[:, positions, :] = src_hs[:, positions, :]
    h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="woman|to the left of|man,chair|to the left of|table")
    ap.add_argument("--pairs-per-group", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/e3/text_position_patch.json"))
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
    print(f"text-position patch: {len(selected)} pairs")

    images = load_images([x for r in selected for x in (r["source_image_id"], r["target_image_id"])])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for r in selected:
        src_in = prepare(proc, r["question"], images[r["source_image_id"]])
        tgt_in = prepare(proc, r["question"], images[r["target_image_id"]])
        pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        seq = src_in["input_ids"].shape[1]
        text_positions = list(range(0, v0)) + list(range(v1, seq))
        visual_positions = list(range(v0, v1))
        m_s = margin(model, src_in, yes_ids, no_ids)
        m_t = margin(model, tgt_in, yes_ids, no_ids)
        cap = capture_layers(model, src_in, LAYERS)
        rec = {"pair_id": r["pair_id"], "group": r["group"], "m_source": m_s, "m_target": m_t,
               "delta": m_s - m_t, "patch": {}}
        for i in LAYERS:
            for name, positions in [("visual", visual_positions), ("text", text_positions), ("both", list(range(seq)))]:
                m = patch_positions(model, tgt_in, i, positions, cap[i], yes_ids, no_ids)
                rec["patch"][f"S{i}:{name}"] = (m - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
        per_pair.append(rec)

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp)}
        for i in LAYERS:
            for name in ["visual", "text", "both"]:
                vals = [r["patch"][f"S{i}:{name}"] for r in pp if r["patch"][f"S{i}:{name}"] is not None]
                row[f"S{i}:{name}"] = {"median": statistics.median(vals) if vals else None, "n": len(vals)}
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']}")
        for i in LAYERS:
            print(f"    S{i}: visual={r[f'S{i}:visual']['median']:+.2f} text={r[f'S{i}:text']['median']:+.2f} both={r[f'S{i}:both']['median']:+.2f}")


if __name__ == "__main__":
    main()
