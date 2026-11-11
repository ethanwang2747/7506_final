import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
STAGES = ["S0", "S1", "S6", "S12", "S18"]


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def short(g):
    return g.replace("|", " / ")


def plot_e2(task, ax):
    d = load(RES / f"stats/{task}_test_summary.json")
    if not d:
        return
    rows = [r for r in d["e2"] if r["region"] == "target_minus_controls"]
    groups = sorted({r["group"] for r in rows})
    for g in groups:
        vals, errs = [], []
        for s in STAGES:
            r = next((x for x in rows if x["group"] == g and x["stage"] == s), None)
            vals.append(r["mean"] if r else np.nan)
            errs.append((r["ci95"][1] - r["ci95"][0]) / 2 if r else np.nan)
        ax.errorbar(range(len(STAGES)), vals, yerr=errs, marker="o", capsize=3, label=short(g))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(STAGES)), STAGES)
    ax.set_ylabel(r"$\Delta m$ target $-$ controls")
    ax.set_title(f"{task}: target region ablation")
    ax.legend(fontsize=6, ncol=2)


def plot_e3(task, ax):
    d = load(RES / f"stats/{task}_test_summary.json")
    if not d:
        return
    rows = d["e3"]
    groups = sorted({r["group"] for r in rows})
    for g in groups:
        vals, errs = [], []
        for s in STAGES:
            r = next((x for x in rows if x["group"] == g and x["stage"] == s), None)
            vals.append(r["median_R"] if r else np.nan)
            errs.append((r["ci95"][1] - r["ci95"][0]) / 2 if r else np.nan)
        ax.errorbar(range(len(STAGES)), vals, yerr=errs, marker="o", capsize=3, label=short(g))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xticks(range(len(STAGES)), STAGES)
    ax.set_ylabel(r"median recovery $R_s$")
    ax.set_title(f"{task}: full visual-block patching")
    ax.legend(fontsize=6, ncol=2)


def plot_e1(task, ax):
    d = load(RES / f"stats/{task}_test_e1_summary.json")
    if not d:
        return
    rows = d["e1"]
    labels = [short(r["group"]) for r in rows]
    vals = [r["auc"] for r in rows]
    lo = [r["auc"] - r["auc_ci95"][0] for r in rows]
    hi = [r["auc_ci95"][1] - r["auc"] for r in rows]
    ax.barh(range(len(rows)), vals, xerr=[lo, hi], capsize=3)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test AUC (95% CI)")
    ax.set_title(f"{task}: clean behavior")


def main():
    for task in ["spatial", "semantic"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_e1(task, axes[0])
        plot_e2(task, axes[1])
        plot_e3(task, axes[2])
        fig.tight_layout()
        out = FIG / f"{task}_summary.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
