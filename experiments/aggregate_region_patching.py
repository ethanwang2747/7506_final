"""Aggregate the same-image flip counterfactual runs into one table.

The raw files in results/flip/ were produced at different times and overlap;
we deduplicate by (pair_id, image_id), then report:

  - flip ranking accuracy of the model's own margin (original vs mirrored),
  - mean margin change when a region is replaced by the layer mean
    (ablation; negative = the region supports "yes"),
  - median fraction of the flip gap recovered by patching a region from the
    original image into the mirrored one (R; 1 = full recovery).

All intervals are bootstrap over pairs.

Output:
    results/diagnostics/region_patching_aggregate.json
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT


def boot_mean(vals, n=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 5:
        return None
    rng = np.random.default_rng(seed)
    draws = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def boot_median(vals, n=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 5:
        return None
    rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    recs = {}
    for f in glob.glob(str(ROOT / "results/flip/*.json")):
        for r in json.load(open(f))["per_pair"]:
            recs[(r["pair_id"], r["image_id"])] = r
    rows = list(recs.values())

    out = {
        "n_pairs": len(rows),
        "n_groups": len({r["group"] for r in rows}),
        "flip_rank_acc": float(np.mean([r["clean_rank"] for r in rows])),
        "mean_gap": float(np.mean([r["delta"] for r in rows])),
        "by_relation": {},
        "ablation": {},
        "patching": {},
    }
    for rel in ["to the left of", "to the right of", "above", "below"]:
        sub = [r for r in rows if r["group"].split("|")[1] == rel]
        if sub:
            out["by_relation"][rel] = {
                "n": len(sub),
                "rank_acc": float(np.mean([r["clean_rank"] for r in sub])),
                "mean_gap": float(np.mean([r["delta"] for r in sub])),
            }

    for stage in ("S1", "S6"):
        for region in ("target", "reference", "irrelevant", "background"):
            key = f"{stage}:{region}"
            o = [r["ablation"][key]["orig_delta"] for r in rows if key in r["ablation"]]
            f = [r["ablation"][key]["flip_delta"] for r in rows if key in r["ablation"]]
            if not o:
                continue
            out["ablation"][key] = {
                "n": len(o),
                "orig_mean": float(np.mean(o)), "orig_ci": boot_mean(o),
                "flip_mean": float(np.mean(f)), "flip_ci": boot_mean(f),
            }
        for region in ("target", "reference", "irrelevant", "random"):
            key = f"{stage}:{region}"
            v = [r["patching"][key]["R"] for r in rows
                 if key in r["patching"] and r["patching"][key]["R"] is not None]
            if v:
                out["patching"][key] = {
                    "n": len(v),
                    "median_R": float(np.median(v)),
                    "median_ci": boot_median(v),
                    "mean_R_trimmed": float(np.mean(np.clip(v, -1, 2))),
                }

    dst = ROOT / "results/diagnostics/region_patching_aggregate.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
