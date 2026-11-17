"""Build the two summary figures used in the project report.

  results/figures/diagnostics_dashboard.png
  results/figures/diagnostics_chain.png

All numbers are read from the JSON outputs in results/diagnostics/ so the
figure cannot drift from the experiment results.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

DIAG = ROOT / "results/diagnostics"
FIG = ROOT / "results/figures"
STAGES = ["S1", "S6", "S12"]


def load(name):
    return json.loads((DIAG / name).read_text())


def geometry_scores():
    def load_payload(path):
        p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
        return p

    train = load_payload("results/readout/features_dev_core.pt")
    test = load_payload("results/readout/features_test_core.pt")
    by = {}
    for e in train["examples"]:
        by.setdefault(e["group"], []).append(e)
    scores = np.full(len(test["examples"]), np.nan)
    for g, ex in by.items():
        idx = [i for i, e in enumerate(test["examples"]) if e["group"] == g]
        if len(idx) < 5 or len({e["label"] for e in ex}) < 2:
            continue
        X = np.array([e["position"] for e in ex], dtype=np.float64)
        y = np.array([e["label"] for e in ex])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
        Xte = np.array([test["examples"][i]["position"] for i in idx], dtype=np.float64)
        scores[idx] = clf.decision_function(Xte)
    ok = ~np.isnan(scores)
    y = np.array([e["label"] for e in test["examples"]])[ok]
    margin = test["margin"].numpy()[ok]
    return scores[ok], margin, y


def panel_geometry(ax):
    geo, margin, y = geometry_scores()
    ax.scatter(geo[y == 1], margin[y == 1], s=14, alpha=0.55, c="#1f77b4",
               label="source (relation true)")
    ax.scatter(geo[y == 0], margin[y == 0], s=14, alpha=0.55, c="#d62728",
               label="target (relation false)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    auc_geo = roc_auc_score(y, geo)
    auc_model = roc_auc_score(y, margin)
    rho = spearmanr(geo, margin).correlation
    ax.set_xlabel("geometry-only score (oracle boxes, per-group)")
    ax.set_ylabel("model yes/no margin")
    ax.set_title(f"(A) The model ignores the geometry that defines the label\n"
                 f"position AUC {auc_geo:.2f} vs model AUC {auc_model:.2f}, "
                 f"Spearman {rho:.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


def panel_tokens(ax):
    d = load("per_token_probe.json")
    x = np.arange(len(STAGES))
    name = [d["name_pointing"][s]["top1_hit"] for s in STAGES]
    ans = [d["answer_to the left of"][s]["acc_peak"] for s in STAGES]
    ans_pool = [d["answer_to the left of"][s]["pooled_acc"] for s in STAGES]
    corr = [d["correct"][s]["acc_peak"] for s in STAGES]
    ax.plot(x, name, "o-", lw=2, color="#2ca02c", label="object identity: name pointing")
    ax.plot(x, corr, "s--", lw=1.4, color="#9467bd", label="model correctness (best token)")
    ax.plot(x, ans, "^-", lw=1.4, color="#ff7f0e", label="relation answer (best token)")
    ax.plot(x, ans_pool, "v:", lw=1.2, color="#8c564b", label="relation answer (all-token mean)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axhline(0.205, color="#2ca02c", lw=0.6, ls=":")
    ax.text(2.02, 0.21, "chance (name)", fontsize=7, color="#2ca02c")
    ax.text(2.02, 0.46, "chance (answer)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylim(0.15, 0.75)
    ax.set_ylabel("accuracy / top-1 hit")
    ax.set_title("(B) Identity is local, the relation is not\n"
                 "9x9 per-cell probes, grouped CV by image", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def panel_lens(ax):
    core = load("logit_lens_core.json")["summary"]
    strict = load("logit_lens_test.json")["summary"]
    lc = [d["auc"] for d in core["per_layer"]]
    ls = [d["auc"] for d in strict["per_layer"]]
    ax.plot(range(len(lc)), lc, "o-", ms=3, color="#1f77b4", label="test core (80 pairs)")
    ax.plot(range(len(ls)), ls, "s--", ms=3, color="#ff7f0e", label="strict sample (240 pairs)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvspan(17.5, 19.5, color="#ffe08a", alpha=0.5)
    ax.set_xlabel("text layer (input to layer)")
    ax.set_ylabel("yes/no AUC of the lens")
    ax.set_title("(C) The decision appears only in the last layers\n"
                 "final norm + LM head on the last prompt token", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


def panel_regions(ax):
    d = load("region_patching_aggregate.json")
    regions = ["target", "reference", "irrelevant", "background"]
    abl = [d["ablation"][f"S1:{r}"]["orig_mean"] for r in regions]
    err = [[d["ablation"][f"S1:{r}"]["orig_mean"] - d["ablation"][f"S1:{r}"]["orig_ci"][0] for r in regions],
           [d["ablation"][f"S1:{r}"]["orig_ci"][1] - d["ablation"][f"S1:{r}"]["orig_mean"] for r in regions]]
    x = np.arange(len(regions))
    ax.bar(x - 0.2, abl, width=0.38, color="#d62728", alpha=0.8,
           label="ablation: margin change")
    ax.errorbar(x - 0.2, abl, yerr=err, fmt="none", ecolor="k", lw=1, capsize=3)
    pr = ["target", "reference", "irrelevant", "random"]
    patch = [d["patching"][f"S1:{r}"]["median_R"] for r in pr]
    perr = [[d["patching"][f"S1:{r}"]["median_R"] - d["patching"][f"S1:{r}"]["median_ci"][0] for r in pr],
            [d["patching"][f"S1:{r}"]["median_ci"][1] - d["patching"][f"S1:{r}"]["median_R"] for r in pr]]
    ax.bar(x + 0.2, patch, width=0.38, color="#1f77b4", alpha=0.8,
           label="patching: fraction of flip gap")
    ax.errorbar(x + 0.2, patch, yerr=perr, fmt="none", ecolor="k", lw=1, capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["target", "reference", "irrelevant", "background /\nrandom"])
    ax.set_ylabel("effect (margin units / fraction)")
    ax.set_title("(D) The target region is used, but only partly\n"
                 "S1; 376 same-image pairs; 95% bootstrap CI", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")


def dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    panel_geometry(axes[0, 0])
    panel_tokens(axes[0, 1])
    panel_lens(axes[1, 0])
    panel_regions(axes[1, 1])
    fig.suptitle("Where the spatial answer is lost: diagnostic dashboard", fontsize=13)
    fig.tight_layout()
    out = FIG / "diagnostics_dashboard.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def chain():
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = [
        ("Image\npixels", "input", "grey"),
        ("Visual tokens\n(S1, S6, S12)", "object identity decodable\nname pointing 0.54-0.62", "#2ca02c"),
        ("Region selection", "target region causally used\nablation -0.38, patch R 0.33", "#f0ad4e"),
        ("Composition", "no token or pooled reader\nanswers the relation", "#d9534f"),
        ("Late decision\n(layers 18-19)", "readable only at the end\nAUC 0.50 -> 0.65", "#f0ad4e"),
        ("Answer", "AUC 0.60\nrho with geometry 0.18", "#d9534f"),
    ]
    w, h, gap = 2.0, 2.1, 0.32
    y = 1.3
    for i, (title, sub, color) in enumerate(boxes):
        x = 0.25 + i * (w + gap)
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                             linewidth=1.2, edgecolor=color,
                             facecolor=color + "22" if color != "grey" else "#eeeeee")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x + w / 2, y + 0.62, sub, ha="center", va="center", fontsize=8.0)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, y + h / 2),
                                         (x + w + gap - 0.03, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, lw=1.2))
    ax.text(0.25, 3.85, "The chain: where the information is still usable and where it is lost",
            fontsize=12, weight="bold")
    ax.text(0.25, 0.55,
            "green = available and decodable    amber = partly used / late    red = not readable / not used",
            fontsize=9)
    out = FIG / "diagnostics_chain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    dashboard()
    chain()
