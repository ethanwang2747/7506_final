import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from experiments.common import ROOT, load_images

CONFIG = ROOT / "data/derived/experiment_config.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def draw(ax, img, obj, ref, title, grid=9):
    ax.imshow(img)
    W, H = img.size
    for rec, color, label in [(obj, "tab:red", "target"), (ref, "tab:blue", "reference")]:
        x, y, w, h = rec["bbox"]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=2))
        ax.text(x, max(0, y - 4), label, color=color, fontsize=7)
    for i in range(1, grid):
        ax.axvline(W * i / grid, color="white", lw=0.4, alpha=0.5)
        ax.axhline(H * i / grid, color="white", lw=0.4, alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    cfg = json.loads(CONFIG.read_text())
    rows = [r for r in read_jsonl(ROOT / cfg["data"]["spatial_pairs_test"]) if r["group"] == "chair|to the left of|table"]
    r = rows[0]
    imgs = load_images([r["source_image_id"]])
    img = imgs[r["source_image_id"]]
    flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    W, H = img.size
    obj = r["source_target"]
    ref = r["source_reference"]
    obj_f = dict(obj)
    ref_f = dict(ref)
    obj_f["bbox"] = [W - (obj["bbox"][0] + obj["bbox"][2]), obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]]
    ref_f["bbox"] = [W - (ref["bbox"][0] + ref["bbox"][2]), ref["bbox"][1], ref["bbox"][2], ref["bbox"][3]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    draw(axes[0], img, obj, ref, "original image + 9x9 grid", 9)
    draw(axes[1], flip, obj_f, ref_f, "horizontal flip + 9x9 grid", 9)
    draw(axes[2], img, obj, ref, "original image + 27x27 grid", 27)
    axes[3].imshow(img)
    axes[3].set_title(r["question"], fontsize=7)
    axes[3].axis("off")
    fig.tight_layout()
    out = ROOT / "results/figures/data_validity.png"
    fig.savefig(out, dpi=180)
    print("saved", out)


if __name__ == "__main__":
    main()
