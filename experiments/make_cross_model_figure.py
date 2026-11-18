"""Chinese figure: SmolVLM2-2.2B vs Qwen2-VL-2B on the same pairs."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    smol = {
        "rel": json.loads((ROOT / "results/sft/eval_relation_base.json").read_text()),
        "side": json.loads((ROOT / "results/diagnostics/side_report.json").read_text()),
        "given": json.loads((ROOT / "results/diagnostics/side_given_relation.json").read_text()),
    }
    qwen = json.loads((ROOT / "results/other_vlm/qwen2vl_2b.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    colors = ["#9aa0a6", "#1a73e8"]
    labels = ["SmolVLM2-2.2B", "Qwen2-VL-2B"]

    ax = axes[0]
    vals = [smol["rel"]["auc"], qwen["margin_auc"]]
    ax.bar([0, 1], vals, color=colors)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 0.9); ax.set_ylabel("AUC")
    ax.set_title("（A）关系判断（同一批 480 样本）")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    x = np.arange(2); w = 0.35
    sv = [smol["side"]["by_axis"]["horizontal"]["acc"], smol["side"]["by_axis"]["vertical"]["acc"]]
    qv = [qwen["side"]["horizontal"]["acc"], qwen["side"]["vertical"]["acc"]]
    ax.bar(x - w / 2, sv, w, color=colors[0], label=labels[0])
    ax.bar(x + w / 2, qv, w, color=colors[1], label=labels[1])
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(["水平", "竖直"])
    ax.set_ylim(0.4, 0.9); ax.set_ylabel("方位报告准确率")
    ax.set_title("（B）直接询问方位")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(sv, qv)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    ax = axes[2]
    st = smol["given"]["conditions"]["true"]["diff_half_acc"]
    ss = smol["given"]["conditions"]["swapped"]["diff_half_acc"]
    qt = qwen["side_given"]["true"]["diff_half_acc"]
    qs = qwen["side_given"]["swapped"]["diff_half_acc"]
    x = np.arange(2)
    ax.bar(x - w / 2, [st, ss], w, color=colors[0], label=labels[0])
    ax.bar(x + w / 2, [qt, qs], w, color=colors[1], label=labels[1])
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(["真方位", "反方位"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("关系准确率（不同半区）")
    ax.set_title("（C）给方位后的组合")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip([st, ss], [qt, qs])):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    fig.suptitle("同尺寸对照：SmolVLM2 的空间读出弱点是模型特异的，不是小 VLM 的共性", fontsize=11)
    fig.tight_layout()
    out = ROOT / "results/figures/cross_model_cn.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
