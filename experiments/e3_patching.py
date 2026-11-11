import argparse, collections, json, statistics
from pathlib import Path

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import capture, margin, patch_lm, patch_output, prepare

CONFIG = ROOT / "data/derived/experiment_config.json"
STAGES = ["S0", "S1", "S6", "S12", "S18"]
LAYERS = [6, 12, 18]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--min-abs-delta", type=float, default=0.5)
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
    print(f"{args.task} {args.split}: {len(selected)} pairs from {len(by_group)} groups")

    images = load_images([x for r in selected for x in (r["source_image_id"], r["target_image_id"])])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        src_in = prepare(proc, r["question"], images[r["source_image_id"]])
        tgt_in = prepare(proc, r["question"], images[r["target_image_id"]])
        pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        m_s = margin(model, src_in, yes_ids, no_ids)
        m_t = margin(model, tgt_in, yes_ids, no_ids)
        src_cap = capture(model, src_in, LAYERS)
        rec = {"pair_id": r["pair_id"], "task": args.task, "group": r["group"], "question": r["question"],
               "source_image_id": r["source_image_id"], "target_image_id": r["target_image_id"],
               "m_source": m_s, "m_target": m_t, "delta": m_s - m_t, "patch": {}}
        for stage in STAGES:
            if stage == "S0":
                m = patch_output(model, tgt_in, model.model.vision_model.post_layernorm, src_cap["S0"], yes_ids, no_ids)
            elif stage == "S1":
                m = patch_output(model, tgt_in, model.model.connector, src_cap["S1"], yes_ids, no_ids)
            else:
                i = int(stage[1:])
                m = patch_lm(model, tgt_in, i, v0, v1, src_cap[stage], yes_ids, no_ids)
            denom = m_s - m_t
            rec["patch"][stage] = {"margin": m, "R": (m - m_t) / denom if abs(denom) > 1e-8 else None}
        per_pair.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        stable = [r for r in pp if abs(r["delta"]) >= args.min_abs_delta]
        row = {"task": args.task, "group": g, "n_pairs": len(pp), "n_stable": len(stable),
               "mean_delta": statistics.mean(r["delta"] for r in pp) if pp else None}
        for s in STAGES:
            vals = [r["patch"][s]["R"] for r in pp if r["patch"][s]["R"] is not None]
            vals_stable = [r["patch"][s]["R"] for r in stable if r["patch"][s]["R"] is not None]
            row[s] = {
                "R_mean": statistics.mean(vals) if vals else None,
                "R_median": statistics.median(vals) if vals else None,
                "R_frac_pos": sum(v > 0 for v in vals) / len(vals) if vals else None,
                "R_frac_gt_half": sum(v > 0.5 for v in vals) / len(vals) if vals else None,
                "R_mean_stable": statistics.mean(vals_stable) if vals_stable else None,
                "R_median_stable": statistics.median(vals_stable) if vals_stable else None,
            }
        summary.append(row)

    out = Path(args.out) if args.out else ROOT / f"results/e3/{args.task}_{args.split}_full.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for row in summary:
        print(f"  {row['group']} n={row['n_pairs']} stable={row['n_stable']} delta={row['mean_delta']:+.2f}")
        for s in STAGES:
            x = row[s]
            if x["R_mean"] is not None:
                print(f"    {s}: R_mean={x['R_mean']:+.2f} median={x['R_median']:+.2f} "
                      f"frac>0={x['R_frac_pos']:.2f} frac>0.5={x['R_frac_gt_half']:.2f} "
                      f"stable_mean={x['R_mean_stable']:+.2f}")


if __name__ == "__main__":
    main()
