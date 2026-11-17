"""Audit text-to-image attention on GQA spatial pairs.

For each layer/head, measure attention mass from text queries to the target and
reference object tokens versus the rest of the visual block. Report head
selectivity and its correlation with pair correctness.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

CONFIG = ROOT / "data/derived/experiment_config.json"
DEFAULT_GROUPS = [
    "woman|to the left of|man", "man|to the right of|woman",
    "man|to the left of|person", "person|to the right of|man",
    "tree|to the left of|building", "chair|to the left of|table",
]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def margin_from_logits(logits, yes_ids, no_ids):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float((torch.logsumexp(lp[:, yes_ids], -1) - torch.logsumexp(lp[:, no_ids], -1)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data/derived/spatial_pairs_dev.jsonl"))
    ap.add_argument("--pairs-per-group", type=int, default=10)
    ap.add_argument("--layers", default="6,12")
    ap.add_argument("--out", default=str(ROOT / "results/attention/audit_dev.json"))
    args = ap.parse_args()

    groups = DEFAULT_GROUPS
    rows = [r for r in read_jsonl(args.pairs) if r["group"] in groups]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by.get(g, [])[: args.pairs_per_group])
    print(f"{len(selected)} pairs")

    images = load_images(sorted({r["source_image_id"] for r in selected} |
                                {r["target_image_id"] for r in selected}))
    proc = load_processor(splitting=False)
    model = load_model(attn="eager")
    yes_ids, no_ids = yes_no_ids(proc)
    layers = [int(x) for x in args.layers.split(",")]

    records = []
    for i, r in enumerate(selected):
        for role, image_id in [("source", r["source_image_id"]), ("target", r["target_image_id"])]:
            inp = prepare(proc, r["question"], images[image_id])
            with torch.no_grad():
                out = model(**inp, output_attentions=True, use_cache=False)
            ids = inp["input_ids"][0]
            mask = inp["attention_mask"][0].bool()
            img_mask = ids == model.config.image_token_id
            vpos = img_mask.nonzero().flatten()
            v0, v1 = int(vpos[0]), int(vpos[-1]) + 1
            last = int(mask.nonzero()[-1])
            obj = r[f"{role}_target"]["tokens_9"]
            ref = r[f"{role}_reference"]["tokens_9"]
            used = set(obj) | set(ref)
            bg = [j for j in range(v1 - v0) if j not in used]
            sel = {}
            for layer in layers:
                a = out.attentions[layer][0].float().cpu()  # (heads, seq, seq)
                row = a[:, last, :]  # answer-position query: (heads, seq)
                m_t = row[:, v0 + np.array(obj)].mean(dim=1) if obj else None
                m_r = row[:, v0 + np.array(ref)].mean(dim=1) if ref else None
                m_b = row[:, v0 + np.array(bg)].mean(dim=1) if bg else None
                if m_t is None or m_r is None:
                    continue
                s = (m_t + m_r) / 2 - (m_b if m_b is not None else 0.0)
                sel[layer] = {
                    "sel": s.numpy(),
                    "target": m_t.numpy(),
                    "reference": m_r.numpy(),
                    "background": m_b.numpy() if m_b is not None else np.zeros_like(s.numpy()),
                }
            records.append({
                "pair_id": r["pair_id"], "group": r["group"], "role": role,
                "margin": margin_from_logits(out.logits[:, -1, :].float().cpu(), yes_ids, no_ids),
                "selectivity": {str(k): v["sel"].tolist() for k, v in sel.items()},
                "mass_target": {str(k): v["target"].tolist() for k, v in sel.items()},
                "mass_reference": {str(k): v["reference"].tolist() for k, v in sel.items()},
                "mass_background": {str(k): v["background"].tolist() for k, v in sel.items()},
            })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(selected)}")

    # pair-level labels and deltas
    by_pair = collections.defaultdict(dict)
    for x in records:
        by_pair[x["pair_id"]][x["role"]] = x
    for pid, d in by_pair.items():
        if "source" in d and "target" in d:
            delta = d["source"]["margin"] - d["target"]["margin"]
            correct = delta > 0
            for role in ("source", "target"):
                d[role]["delta"] = delta
                d[role]["correct"] = correct

    summary = {"layers": {}, "top_heads": [], "pairs": len(selected)}
    for layer in layers:
        heads = len(records[0]["selectivity"][str(layer)]) if records else 0
        table = []
        for h in range(heads):
            sels = np.array([x["selectivity"][str(layer)][h] for x in records if str(layer) in x["selectivity"]])
            corr = np.array([x.get("correct", 0) for x in records if str(layer) in x["selectivity"]], dtype=float)
            deltas = np.array([x.get("delta", 0) for x in records if str(layer) in x["selectivity"]], dtype=float)
            if len(sels) < 5 or len(set(corr)) < 2:
                continue
            mt = np.array([x["mass_target"][str(layer)][h] for x in records if str(layer) in x["mass_target"]])
            mr = np.array([x["mass_reference"][str(layer)][h] for x in records if str(layer) in x["mass_reference"]])
            mb = np.array([x["mass_background"][str(layer)][h] for x in records if str(layer) in x["mass_background"]])
            table.append({
                "layer": layer, "head": h,
                "mean_selectivity": float(sels.mean()),
                "mean_target": float(mt.mean()),
                "mean_reference": float(mr.mean()),
                "mean_background": float(mb.mean()),
                "corr_correct": float(np.corrcoef(sels, corr)[0, 1]),
                "corr_delta": float(np.corrcoef(sels, deltas)[0, 1]),
                "sel_correct": float(sels[corr == 1].mean()) if (corr == 1).any() else None,
                "sel_incorrect": float(sels[corr == 0].mean()) if (corr == 0).any() else None,
            })
        table.sort(key=lambda x: -(x["mean_target"] + x["mean_reference"]))
        summary["layers"][str(layer)] = table
        summary["top_heads"].extend(table[:10])

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"saved {path}")
    for h in summary["top_heads"][:15]:
        print(f"  L{h['layer']}H{h['head']:02d} mass_t={h['mean_target']:.4f} "
              f"mass_r={h['mean_reference']:.4f} bg={h['mean_background']:.4f} "
              f"sel={h['mean_selectivity']:.4f} corr_correct={h['corr_correct']:+.3f} "
              f"corr_delta={h['corr_delta']:+.3f}")


if __name__ == "__main__":
    main()
