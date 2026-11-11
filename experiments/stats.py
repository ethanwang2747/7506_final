import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["S0", "S1", "S6", "S12", "S18"]


def cluster_bootstrap(values_by_cluster, n=2000, seed=42, stat=np.mean):
    clusters = list(values_by_cluster)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(len(clusters), size=len(clusters), replace=True)
        vals = []
        for i in pick:
            vals.extend(values_by_cluster[clusters[i]])
        if vals:
            out.append(float(stat(np.array(vals))))
    out = np.array(out)
    return {
        "mean": float(np.mean(out)),
        "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
        "n_clusters": len(clusters),
    }


def paired_cluster_bootstrap(pairs, n=2000, seed=42):
    # pairs: list of (cluster, value_a, value_b)
    by_cluster = defaultdict(list)
    for c, a, b in pairs:
        by_cluster[c].append(a - b)
    return cluster_bootstrap(by_cluster, n=n, seed=seed)


def summarize_e2(path, n_boot=2000):
    data = json.load(open(path))
    per = data["per_pair"]
    rows = []
    groups = sorted({r["group"] for r in per})
    for g in groups:
        pp = [r for r in per if r["group"] == g]
        for s in STAGES:
            for region in ["target", "reference", "irrelevant", "background", "full"]:
                key = f"{s}:{region}"
                vals = [(r.get("source_image_id", r["pair_id"]), r["delta"][key], r["m_clean"]) for r in pp if key in r["delta"]]
                if not vals:
                    continue
                by_cluster = defaultdict(list)
                for img, v, _ in vals:
                    by_cluster[img].append(v)
                ci = cluster_bootstrap(by_cluster, n=n_boot)
                acc = sum((m + d) > 0 for _, d, m in vals) / len(vals)
                flip = sum(((m + d) > 0) != (m > 0) for _, d, m in vals) / len(vals)
                rows.append({"group": g, "stage": s, "region": region, "n": len(vals), "margin_acc": acc, "flip_rate": flip, **ci})
            # target vs matched controls (irrelevant + background)
            tvals = [(r.get("source_image_id", r["pair_id"]), r["delta"][f"{s}:target"]) for r in pp if f"{s}:target" in r["delta"]]
            cvals = []
            for r in pp:
                c = [r["delta"][f"{s}:{x}"] for x in ["irrelevant", "background"] if f"{s}:{x}" in r["delta"]]
                if c and f"{s}:target" in r["delta"]:
                    cvals.append((r.get("source_image_id", r["pair_id"]), sum(c) / len(c)))
            if tvals and cvals:
                d = {img: (t - c) for (img, t), (img2, c) in zip(tvals, cvals) if img == img2}
                by_cluster = defaultdict(list)
                for img, v in d.items():
                    by_cluster[img].append(v)
                ci = cluster_bootstrap(by_cluster, n=n_boot)
                rows.append({"group": g, "stage": s, "region": "target_minus_controls", "n": len(d), **ci})
    return rows


def summarize_e3(path, n_boot=2000, min_abs_delta=0.5):
    data = json.load(open(path))
    per = data["per_pair"]
    rows = []
    for g in sorted({r["group"] for r in per}):
        pp = [r for r in per if r["group"] == g]
        stable = [r for r in pp if abs(r["delta"]) >= min_abs_delta]
        for s in STAGES:
            vals = [(r.get("source_image_id", r["pair_id"]), r["patch"][s]["R"]) for r in pp if r["patch"][s]["R"] is not None]
            vals_s = [(r.get("source_image_id", r["pair_id"]), r["patch"][s]["R"]) for r in stable if r["patch"][s]["R"] is not None]
            if not vals:
                continue
            by_cluster = defaultdict(list)
            for img, v in vals:
                by_cluster[img].append(v)
            ci = cluster_bootstrap(by_cluster, n=n_boot, stat=np.median)
            by_cluster_s = defaultdict(list)
            for img, v in vals_s:
                by_cluster_s[img].append(v)
            ci_s = cluster_bootstrap(by_cluster_s, n=n_boot, stat=np.median) if vals_s else None
            rows.append({
                "group": g, "stage": s, "n": len(vals), "n_stable": len(vals_s),
                "median_R": ci["mean"], "ci95": ci["ci95"],
                "frac_pos": sum(v > 0 for _, v in vals) / len(vals),
                "frac_gt_half": sum(v > 0.5 for _, v in vals) / len(vals),
                "stable_median_R": ci_s["mean"] if ci_s else None,
                "stable_ci95": ci_s["ci95"] if ci_s else None,
            })
    return rows


def summarize_e1(path, n_boot=2000):
    data = json.load(open(path))
    per = data["per_pair"]
    rows = []
    for g in sorted({r["group"] for r in per}):
        pp = [r for r in per if r["group"] == g]
        by_cluster = defaultdict(list)
        for r in pp:
            by_cluster[r.get("source_image_id", r["pair_id"])].append(r)
        clusters = list(by_cluster)
        rng = np.random.default_rng(42)
        aucs, deltas = [], []
        for _ in range(n_boot):
            pick = rng.choice(len(clusters), size=len(clusters), replace=True)
            src, tgt, ds = [], [], []
            for i in pick:
                for r in by_cluster[clusters[i]]:
                    src.append(r["m_source"])
                    tgt.append(r["m_target"])
                    ds.append(r["delta"])
            if src and tgt:
                aucs.append(sum(1 for s in src for t in tgt if s > t) / (len(src) * len(tgt)))
                deltas.append(float(np.mean(ds)))
        rows.append({
            "group": g, "n_pairs": len(pp),
            "auc": float(np.mean(aucs)), "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "mean_delta": float(np.mean(deltas)), "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=None)
    ap.add_argument("--e2", default=None)
    ap.add_argument("--e3", default=None)
    ap.add_argument("--out", default=str(ROOT / "results/stats/summary.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = {}
    if args.e1:
        out["e1"] = summarize_e1(args.e1, n_boot=args.n_boot)
    if args.e2:
        out["e2"] = summarize_e2(args.e2, n_boot=args.n_boot)
    if args.e3:
        out["e3"] = summarize_e3(args.e3, n_boot=args.n_boot)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    for r in out.get("e1", []):
        print(f"E1 {r['group']:40s} AUC={r['auc']:.3f} CI[{r['auc_ci95'][0]:.3f},{r['auc_ci95'][1]:.3f}] delta={r['mean_delta']:+.2f}")
    for r in out.get("e2", []):
        if r["region"] == "target_minus_controls":
            print(f"E2 {r['group']:40s} {r['stage']:3s} target-control={r['mean']:+.2f} "
                  f"CI[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] n={r['n']}")
    for r in out.get("e3", []):
        print(f"E3 {r['group']:40s} {r['stage']:3s} medianR={r['median_R']:+.2f} "
              f"CI[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] frac>0.5={r['frac_gt_half']:.2f} n={r['n']}")


if __name__ == "__main__":
    main()
