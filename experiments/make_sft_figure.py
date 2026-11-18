"""Figure for the SFT 2x2: side-only vs relation-only vs mixed."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

D = ROOT / "results/diagnostics"
S = ROOT / "results/sft"

ADAPTERS = [
    ("base", "side_report.json", "eval_relation_base.json", "#9aa0a6"),
    ("side-only", "side_report_sideonly.json", "eval_relation_side_only.json", "#1a73e8"),
    ("relation-only", "side_report_relationonly.json", "eval_relation_relation_only.json", "#e8710a"),
    ("mixed", "side_report_sft.json", "eval_relation_sft.json", "#188038"),
]
if (S / "eval_relation_full.json").exists() and (D / "side_report_full.json").exists():
    ADAPTERS.append(("full", "side_report_full.json", "eval_relation_full.json", "#9334e6"))


def main():
    side = {}
    rel = {}
    for name, sf, rf, _ in ADAPTERS:
        side[name] = json.loads((D / sf).read_text())
        rel[name] = json.loads((S / rf).read_text())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax = axes[0, 0]
    x = np.arange(2)
    w = 0.2
    for k, (name, _, _, color) in enumerate(ADAPTERS):
        vals = [side[name]["by_axis"]["horizontal"]["acc"],
                side[name]["by_axis"]["vertical"]["acc"]]
        ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=color)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["horizontal", "vertical"])
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("side-report accuracy")
    ax.set_title("(A) Asking each object's side")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x = np.arange(2)
    for k, (name, _, _, color) in enumerate(ADAPTERS):
        vals = [rel[name]["acc"], rel[name]["auc"]]
        ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=color)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["accuracy", "AUC"])
    ax.set_ylim(0.4, 0.85)
    ax.set_title("(B) Relation task, question only")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    comp_b = json.loads((D / "side_given_relation.json").read_text())
    comp_s = json.loads((D / "side_given_relation_sft.json").read_text())
    x = np.arange(2)
    b = [comp_b["conditions"]["true"]["diff_half_acc"],
         comp_b["conditions"]["swapped"]["diff_half_acc"]]
    s = [comp_s["conditions"]["true"]["diff_half_acc"],
         comp_s["conditions"]["swapped"]["diff_half_acc"]]
    ax.bar(x - 0.18, b, 0.36, label="base", color="#9aa0a6")
    ax.bar(x + 0.18, s, 0.36, label="mixed", color="#188038")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["true sides", "swapped sides"])
    ax.set_ylim(0, 1.05)
    ax.set_title("(C) Composition (different halves)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    base_pg = json.loads((S / "eval_relation_base.json").read_text())["per_group"]
    cross = json.loads((S / "eval_relation_crossgroup.json").read_text())["per_group"]
    hold = {"chair|to the left of|table", "tree|to the left of|building"}
    names = list(base_pg)
    vals = [cross[g]["auc"] - base_pg[g]["auc"] for g in names]
    colors = ["#d93025" if g in hold else "#188038" for g in names]
    ax.barh(np.arange(len(names)), vals, color=colors)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([g.replace("|", " ") for g in names], fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("delta AUC vs base (cross-group adapter)")
    ax.set_title("(D) Cross-group: red = held-out groups")
    fig.suptitle("Teaching the missing readout improves relations; a full epoch is strongest",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "results/figures/sft_result.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
