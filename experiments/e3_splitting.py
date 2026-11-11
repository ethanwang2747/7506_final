import argparse, collections, json, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare

CONFIG = ROOT / "data/derived/experiment_config.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def capture_s1_s6(model, inp):
    cap = {}
    h1 = model.model.connector.register_forward_hook(lambda m, i, o: cap.__setitem__("S1", o.detach()))
    def pre(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        cap["S6"] = x.detach()
    h2 = model.model.text_model.layers[6].register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        model(**inp, use_cache=False)
    h1.remove(); h2.remove()
    return cap


def patch_s1(model, inp, src_s1, yes_ids, no_ids):
    h = model.model.connector.register_forward_hook(lambda m, i, o: src_s1)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def patch_s6(model, inp, src_s6, v0, span, yes_ids, no_ids):
    layer = model.model.text_model.layers[6]
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        x[:, v0:v0 + span, :] = src_s6[:, v0:v0 + span, :]
    h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/e3/spatial_test_splitting.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",") if args.groups else cfg["tasks"]["spatial"]["core_groups"] + cfg["tasks"]["spatial"]["exploratory_groups"]
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{args.split}"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"splitting E3: {len(selected)} pairs")

    images = load_images([x for r in selected for x in (r["source_image_id"], r["target_image_id"])])
    proc = load_processor(splitting=True)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        src_in = prepare(proc, r["question"], images[r["source_image_id"]])
        tgt_in = prepare(proc, r["question"], images[r["target_image_id"]])
        ps = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        pt = (tgt_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0s, v1s = int(ps[0]), int(ps[-1]) + 1
        v0t, v1t = int(pt[0]), int(pt[-1]) + 1
        ns, nt = (v1s - v0s) // 81, (v1t - v0t) // 81
        if ns != nt:
            continue
        m_s = margin(model, src_in, yes_ids, no_ids)
        m_t = margin(model, tgt_in, yes_ids, no_ids)
        cap = capture_s1_s6(model, src_in)
        m1 = patch_s1(model, tgt_in, cap["S1"], yes_ids, no_ids)
        m6 = patch_s6(model, tgt_in, cap["S6"], v0t, ns * 81, yes_ids, no_ids)
        denom = m_s - m_t
        per_pair.append({
            "pair_id": r["pair_id"], "group": r["group"], "source_image_id": r["source_image_id"],
            "blocks": ns, "m_source": m_s, "m_target": m_t, "delta": denom,
            "R_S1": (m1 - m_t) / denom if abs(denom) > 1e-8 else None,
            "R_S6": (m6 - m_t) / denom if abs(denom) > 1e-8 else None,
        })
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp), "blocks": statistics.mean(r["blocks"] for r in pp)}
        for s in ["S1", "S6"]:
            vals = [r[f"R_{s}"] for r in pp if r[f"R_{s}"] is not None]
            row[s] = {"median": statistics.median(vals) if vals else None,
                      "frac_gt_half": sum(v > 0.5 for v in vals) / len(vals) if vals else None}
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']} blocks={r['blocks']:.1f} "
              f"S1 med={r['S1']['median']:+.2f} S6 med={r['S6']['median']:+.2f}")


if __name__ == "__main__":
    main()
