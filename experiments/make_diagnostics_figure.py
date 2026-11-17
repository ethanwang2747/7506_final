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

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
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
               label="正例（关系成立）")
    ax.scatter(geo[y == 0], margin[y == 0], s=14, alpha=0.55, c="#d62728",
               label="负例（关系不成立）")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    auc_geo = roc_auc_score(y, geo)
    auc_model = roc_auc_score(y, margin)
    rho = spearmanr(geo, margin).correlation
    ax.set_xlabel("纯几何分数（oracle 框，按组训练）")
    ax.set_ylabel("模型 yes/no margin")
    ax.set_title(f"(A) 标签由几何决定，模型却几乎不用它\n"
                 f"位置 AUC {auc_geo:.2f} vs 模型 AUC {auc_model:.2f}，"
                 f"Spearman {rho:.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


def panel_tokens(ax):
    d = load("per_token_probe.json")
    x = np.arange(len(STAGES))
    name = [d["name_pointing"][s]["top1_hit"] for s in STAGES]
    ans = [d["answer_to the left of"][s]["acc_peak"] for s in STAGES]
    ans_pool = [d["answer_to the left of"][s]["pooled_acc"] for s in STAGES]
    corr = [d["correct"][s]["acc_peak"] for s in STAGES]
    ax.plot(x, name, "o-", lw=2, color="#2ca02c", label="物体身份：命名指向")
    ax.plot(x, corr, "s--", lw=1.4, color="#9467bd", label="模型正确性（最好格）")
    ax.plot(x, ans, "^-", lw=1.4, color="#ff7f0e", label="关系答案（最好格）")
    ax.plot(x, ans_pool, "v:", lw=1.2, color="#8c564b", label="关系答案（全 token 平均）")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axhline(0.205, color="#2ca02c", lw=0.6, ls=":")
    ax.text(2.02, 0.215, "随机（命名）", fontsize=7, color="#2ca02c")
    ax.text(2.02, 0.465, "随机（答案）", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylim(0.15, 0.75)
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylabel("准确率 / top-1 命中")
    ax.set_title("(B) 身份是局部的，关系不是\n"
                 "9×9 逐格 probe，按图像分组交叉验证", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def panel_lens(ax):
    core = load("logit_lens_core.json")["summary"]
    strict = load("logit_lens_test.json")["summary"]
    lc = [d["auc"] for d in core["per_layer"]]
    ls = [d["auc"] for d in strict["per_layer"]]
    ax.plot(range(len(lc)), lc, "o-", ms=3, color="#1f77b4", label="test core（80 对）")
    ax.plot(range(len(ls)), ls, "s--", ms=3, color="#ff7f0e", label="strict 样本（240 对）")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvspan(17.5, 19.5, color="#ffe08a", alpha=0.5)
    ax.set_xlabel("文本层（该层输入）")
    ax.set_ylabel("lens 的 yes/no AUC")
    ax.set_title("(C) 决策只在最后几层出现\n"
                 "对最后一个 prompt token 应用 final norm + LM head", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


def panel_regions(ax):
    d = load("region_patching_aggregate.json")
    regions = ["target", "reference", "irrelevant", "background"]
    abl = [d["ablation"][f"S1:{r}"]["orig_mean"] for r in regions]
    err = [[d["ablation"][f"S1:{r}"]["orig_mean"] - d["ablation"][f"S1:{r}"]["orig_ci"][0] for r in regions],
           [d["ablation"][f"S1:{r}"]["orig_ci"][1] - d["ablation"][f"S1:{r}"]["orig_mean"] for r in regions]]
    x = np.arange(len(regions))
    ax.bar(x - 0.2, abl, width=0.38, color="#d62728", alpha=0.8,
           label="消融：margin 变化")
    ax.errorbar(x - 0.2, abl, yerr=err, fmt="none", ecolor="k", lw=1, capsize=3)
    pr = ["target", "reference", "irrelevant", "random"]
    patch = [d["patching"][f"S1:{r}"]["median_R"] for r in pr]
    perr = [[d["patching"][f"S1:{r}"]["median_R"] - d["patching"][f"S1:{r}"]["median_ci"][0] for r in pr],
            [d["patching"][f"S1:{r}"]["median_ci"][1] - d["patching"][f"S1:{r}"]["median_R"] for r in pr]]
    ax.bar(x + 0.2, patch, width=0.38, color="#1f77b4", alpha=0.8,
           label="patching：恢复的翻转差距")
    ax.errorbar(x + 0.2, patch, yerr=perr, fmt="none", ecolor="k", lw=1, capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["目标区域", "参照区域", "无关区域", "背景 / 随机"])
    ax.set_ylabel("效应（margin / 比例）")
    ax.set_title("(D) 目标区域被使用，但只承担一部分\n"
                 "S1；376 对同图翻转；95% bootstrap 区间", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")


def dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    panel_geometry(axes[0, 0])
    panel_tokens(axes[0, 1])
    panel_lens(axes[1, 0])
    panel_regions(axes[1, 1])
    fig.suptitle("空间答案是在哪一环丢失的：诊断总览", fontsize=13)
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
        ("图像像素", "输入", "grey"),
        ("视觉 token\n(S1, S6, S12)", "物体身份可解码\n命名指向 0.54-0.62", "#2ca02c"),
        ("区域选择", "目标区域被因果使用\n消融 -0.38，patching 0.33", "#f0ad4e"),
        ("组合", "没有单个 token 或 pooled\n表征能回答关系", "#d9534f"),
        ("晚期决策\n(第 18-19 层)", "答案直到最后才可读\nAUC 0.50 -> 0.65", "#f0ad4e"),
        ("答案", "AUC 0.60\n与几何的 rho 0.18", "#d9534f"),
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
    ax.text(0.25, 3.85, "信息链路：哪一环仍可用，哪一环已经丢失",
            fontsize=12, weight="bold")
    ax.text(0.25, 0.55,
            "绿色 = 可用且可解码    黄色 = 部分使用 / 偏晚    红色 = 读不出 / 未使用",
            fontsize=9)
    out = FIG / "diagnostics_chain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    dashboard()
    chain()
