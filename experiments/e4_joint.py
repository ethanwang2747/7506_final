import argparse, collections, json, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare
from experiments.e2_ablation import pick_background, run_intervention, token_box_9, unique_objects
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
MEANS = ROOT / "data/derived/mean_vectors.pt"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
STAGES = ["S1", "S6", "S12"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", default="woman|to the left of|man,man|to the right of|woman,chair|to the left of|table")
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/e4/joint_ablation.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",")
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{args.split}"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"joint ablation: {len(selected)} pairs")

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
        if not tgt or not ref:
            continue
        joint = sorted(set(tgt) | set(ref))
        inp = prepare(proc, r["question"], images[r["source_image_id"]])
        pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0 = int(pos[0])
        m_clean = margin(model, inp, yes_ids, no_ids)
        rec = {"pair_id": r["pair_id"], "group": r["group"], "m_clean": m_clean, "sizes": {"target": len(tgt), "reference": len(ref), "joint": len(joint)}, "delta": {}}
        for region, toks in [("target", tgt), ("reference", ref), ("joint", joint)]:
            for stage in STAGES:
                m = run_intervention(model, inp, stage, toks, means[stage], v0, yes_ids, no_ids)
                rec["delta"][f"{stage}:{region}"] = m - m_clean
        for stage in STAGES:
            m = run_intervention(model, inp, stage, list(range(81)), means[stage], v0, yes_ids, no_ids)
            rec["delta"][f"{stage}:full"] = m - m_clean
        per_pair.append(rec)

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp)}
        for stage in STAGES:
            for region in ["target", "reference", "joint", "full"]:
                vals = [r["delta"][f"{stage}:{region}"] for r in pp if f"{stage}:{region}" in r["delta"]]
                if vals:
                    row[f"{stage}:{region}"] = {"mean": statistics.mean(vals), "median": statistics.median(vals), "n": len(vals)}
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']}")
        for stage in STAGES:
            print("    " + stage + ": " + " ".join(
                f"{x}={r[f'{stage}:{x}']['mean']:+.2f}" for x in ["target", "reference", "joint", "full"] if f"{stage}:{x}" in r))


if __name__ == "__main__":
    main()
