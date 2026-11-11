import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with open(path) as f:
        return json.load(f)


def boot(values, stat=np.mean, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return None
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(vals), len(vals))
        out.append(float(stat(vals[idx])))
    out = np.array(out)
    return {"mean": float(np.mean(out)), "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], "n": len(vals)}


def bilateral(paths):
    per = []
    seen = set()
    for path in paths:
        for r in load(path)["per_pair"]:
            if r["group"] in seen:
                continue
            per.append(r)
        seen |= {r["group"] for r in load(path)["per_pair"]}
    rows = []
    for g in sorted({r["group"] for r in per}):
        pp = [r for r in per if r["group"] == g]
        for stage in ["S1", "S6"]:
            clean_auc, after_auc, ds, dt, diff = [], [], [], [], []
            for r in pp:
                s, t = r["sides"]["source"], r["sides"]["target"]
                if f"{stage}:target" not in s["delta"] or f"{stage}:target" not in t["delta"]:
                    continue
                clean_auc.append(s["m_clean"] > t["m_clean"])
                ds.append(s["delta"][f"{stage}:target"])
                dt.append(t["delta"][f"{stage}:target"])
                after_auc.append((s["m_clean"] + s["delta"][f"{stage}:target"]) > (t["m_clean"] + t["delta"][f"{stage}:target"]))
                diff.append(s["delta"][f"{stage}:target"] - t["delta"][f"{stage}:target"])
            if not clean_auc:
                continue
            rows.append({
                "group": g, "stage": stage,
                "clean_auc": float(np.mean(clean_auc)), "after_auc": float(np.mean(after_auc)),
                "after_auc_boot": boot(after_auc),
                "source_delta": boot(ds), "target_delta": boot(dt), "paired_delta": boot(diff),
            })
    return rows


def flip(paths):
    per = []
    for path in paths:
        per += load(path)["per_pair"]
    rows = []
    for g in sorted({r["group"] for r in per}):
        pp = [r for r in per if r["group"] == g]
        clean_auc = [r["m_original"] > r["m_flipped"] for r in pp]
        stable = [r for r in pp if abs(r["delta"]) >= 0.5]
        row = {"group": g, "n": len(pp), "n_stable": len(stable), "clean_auc": float(np.mean(clean_auc)), "clean_auc_boot": boot(clean_auc)}
        for stage in ["S1", "S6"]:
            vals = [r["patching"][f"{stage}:target"]["R"] for r in stable if r["patching"][f"{stage}:target"]["R"] is not None]
            after = []
            for r in pp:
                if f"{stage}:target" in r["ablation"]:
                    after.append(r["ablation"][f"{stage}:target"]["orig_margin"] > r["ablation"][f"{stage}:target"]["flip_margin"])
            row[f"{stage}:target_R"] = boot(vals, stat=np.median)
            row[f"{stage}:after_auc"] = float(np.mean(after)) if after else None
        rows.append(row)
    return rows


def patch_ci(paths, key):
    per = []
    for path in paths:
        per += load(path)["per_pair"]
    rows = []
    for g in sorted({r["group"] for r in per}):
        pp = [r for r in per if r["group"] == g]
        row = {"group": g, "n": len(pp)}
        keys = sorted({k for r in pp for k in r[key] if r[key][k] is not None})
        stable = [r for r in pp if abs(r.get("delta", 0)) >= 0.5] if "delta" in pp[0] else pp
        for k in keys:
            vals = [r[key][k] for r in stable if k in r[key] and r[key][k] is not None]
            row[k] = boot(vals, stat=np.median)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/stats/extra_summary.json"))
    args = ap.parse_args()
    out = {}
    out["bilateral_spatial"] = bilateral([ROOT / "results/e2/bilateral_key40.json", ROOT / "results/e2/bilateral_20.json"])
    out["bilateral_semantic"] = bilateral([ROOT / "results/e2/bilateral_semantic_40.json"])
    out["flip"] = flip([ROOT / "results/flip/flip_counterfactual_40.json"])
    out["crop_patch"] = patch_ci([ROOT / "results/e3/crop_patch.json", ROOT / "results/e3/crop_patch_more.json", ROOT / "results/e3/crop_patch_extra.json"], "R")
    out["text_patch"] = patch_ci([ROOT / "results/e3/text_position_patch.json"], "patch")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("saved", args.out)
    for name, rows in out.items():
        print("\n", name)
        for r in rows:
            print(" ", r.get("group"), {k: v for k, v in r.items() if k in ("clean_auc", "after_auc", "S1:target_R", "S6:target_R")})


if __name__ == "__main__":
    main()
