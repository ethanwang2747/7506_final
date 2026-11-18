"""Chinese figure: visual vs decision side decodability, two models."""
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
    smol_side = json.loads((ROOT / "results/diagnostics/side_probe.json").read_text())
    qwen = json.loads((ROOT / "results/other_vlm/qwen2vl_2b_probe.json").read_text())

    # SmolVLM2: visual target side from the pooled-region probe (0.949 at S1),
    # decision side at L19.
    smol = {
        "visual_t": 0.949, "visual_r": 0.943,
        "dec_t": smol_side["probes"]["target_left"]["19"]["auc"],
        "dec_r": smol_side["probes"]["ref_left"]["19"]["auc"],
    }
    qw = {
        "visual_t": qwen["probes"]["visual_target_left"],
        "visual_r": qwen["probes"]["visual_reference_left"],
        "dec_t": qwen["probes"]["decision_target_left_L20"],
        "dec_r": qwen["probes"]["decision_reference_left_L20"],
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(4)
    w = 0.36
    smol_v = [smol["visual_t"], smol["visual_r"], smol["dec_t"], smol["dec_r"]]
    qw_v = [qw["visual_t"], qw["visual_r"], qw["dec_t"], qw["dec_r"]]
    ax.bar(x - w / 2, smol_v, w, label="SmolVLM2-2.2B", color="#9aa0a6")
    ax.bar(x + w / 2, qw_v, w, label="Qwen2-VL-2B", color="#1a73e8")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["视觉：目标侧", "视觉：参照侧", "决策：目标侧", "决策：参照侧"])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("水平侧别 AUC")
    ax.set_title("视觉 token 都有位置，差别在决策层是否保留")
    ax.legend(fontsize=9)
    for i, (a, b) in enumerate(zip(smol_v, qw_v)):
        ax.text(i - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = ROOT / "results/figures/cross_model_probe_cn.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
