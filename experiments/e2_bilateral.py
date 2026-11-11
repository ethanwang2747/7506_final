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
STAGES = ["S1", "S6"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="woman|to the left of|man,chair|to the left of|table,man|to the left of|person")
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/e2/bilateral.json"))
    ap.add_argument("--pairs", default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",")
    pairs_path = Path(args.pairs) if args.pairs else ROOT / cfg["data"]["spatial_pairs_test"]
    rows = [r for r in read_jsonl(pairs_path) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"bilateral E2: {len(selected)} pairs")

    with open(SG_PATH) as f:
        sg = json.load(f)
    means = torch.load(MEANS, map_location="cpu")["means"]
    images = load_images([x for r in selected for x in (r["source_image_id"], r["target_image_id"])])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        sides = {}
        for side, image_id, tkey, rkey in [
            ("source", r["source_image_id"], "source_target", "source_reference"),
            ("target", r["target_image_id"], "target_target", "target_reference"),
        ]:
            g = sg[image_id]
            W, H = g["width"], g["height"]
            tgt = token_box_9(r[tkey], W, H)
            ref = token_box_9(r[rkey], W, H) if rkey in r else []
            objs = unique_objects(g, set(OBJECT_VOCAB))
            used = {r[tkey]["name"]} | ({r[rkey]["name"]} if rkey in r else set())
            cands = sorted([o for n, o in objs.items() if n not in used], key=lambda o: (abs(len(o["tokens_9"]) - len(tgt)), o["name"]))
            irr = cands[0]["tokens_9"] if cands else []
            bg = pick_background(tgt, set(tgt) | set(ref) | set(irr), len(tgt), seed=f"{r['pair_id']}:{side}:bg")
            inp = prepare(proc, r["question"], images[image_id])
            pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
            v0 = int(pos[0])
            m_clean = margin(model, inp, yes_ids, no_ids)
            rec = {"m_clean": m_clean, "delta": {}}
            for region, toks in [("target", tgt), ("reference", ref), ("irrelevant", irr), ("background", bg)]:
                if not toks:
                    continue
                for stage in STAGES:
                    m = run_intervention(model, inp, stage, toks, means[stage], v0, yes_ids, no_ids)
                    rec["delta"][f"{stage}:{region}"] = m - m_clean
            sides[side] = rec
        per_pair.append({"pair_id": r["pair_id"], "group": r["group"], "sides": sides})

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp)}
        # clean AUC
        row["auc_clean"] = sum(r["sides"]["source"]["m_clean"] > r["sides"]["target"]["m_clean"] for r in pp) / len(pp)
        for stage in STAGES:
            for region in ["target", "reference", "irrelevant", "background"]:
                key = f"{stage}:{region}"
                ss = [r["sides"]["source"]["delta"][key] for r in pp if key in r["sides"]["source"]["delta"]]
                tt = [r["sides"]["target"]["delta"][key] for r in pp if key in r["sides"]["target"]["delta"]]
                if ss and tt:
                    row[key] = {"source_delta_mean": statistics.mean(ss), "target_delta_mean": statistics.mean(tt),
                                "source_delta_median": statistics.median(ss), "target_delta_median": statistics.median(tt)}
            # AUC after target ablation
            auc = []
            for r in pp:
                s = r["sides"]["source"]; t = r["sides"]["target"]
                if f"{stage}:target" in s["delta"] and f"{stage}:target" in t["delta"]:
                    auc.append((s["m_clean"] + s["delta"][f"{stage}:target"]) > (t["m_clean"] + t["delta"][f"{stage}:target"]))
            row[f"{stage}:auc_after_target"] = sum(auc) / len(auc) if auc else None
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']} AUC={r['auc_clean']:.2f}")
        for stage in STAGES:
            print(f"    {stage}: target src={r[f'{stage}:target']['source_delta_mean']:+.2f} "
                  f"tgt={r[f'{stage}:target']['target_delta_mean']:+.2f} "
                  f"AUC_after={r[f'{stage}:auc_after_target']:.2f}")


if __name__ == "__main__":
    main()
