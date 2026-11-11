import argparse, json, random
from pathlib import Path

from PIL import Image, ImageDraw

from experiments.common import ROOT, load_model, load_processor, yes_no_ids
from experiments.e0_validate import margin, prepare

OUT = ROOT / "results/control"


def make_image(bg, positions, size=384, sq=70):
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    for color, (cx, cy) in positions:
        d.rectangle([cx - sq // 2, cy - sq // 2, cx + sq // 2, cy + sq // 2], fill=color)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--out", default=str(OUT / "synthetic_shapes.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    rng = random.Random(0)

    results = []
    for rel in ["above", "below", "to the left of", "to the right of"]:
        for i in range(args.pairs):
            bg = (230, 230, 230)
            # random base positions with a clear relation
            cx, cy = 192, 192
            delta = 95
            if rel == "above":
                pos_yes = [((220, 40, 40), (cx, cy - delta)), ((40, 80, 220), (cx, cy + delta))]
            elif rel == "below":
                pos_yes = [((220, 40, 40), (cx, cy + delta)), ((40, 80, 220), (cx, cy - delta))]
            elif rel == "to the left of":
                pos_yes = [((220, 40, 40), (cx - delta, cy)), ((40, 80, 220), (cx + delta, cy))]
            else:
                pos_yes = [((220, 40, 40), (cx + delta, cy)), ((40, 80, 220), (cx - delta, cy))]
            pos_no = [pos_yes[1], pos_yes[0]]
            q = f"Is the red square {rel} the blue square?"
            for gold, pos in [("yes", pos_yes), ("no", pos_no)]:
                img = make_image(bg, pos)
                inp = prepare(proc, q, img)
                m = margin(model, inp, yes_ids, no_ids)
                results.append({"relation": rel, "pair": i, "gold": gold, "margin": m})
    summary = []
    for rel in ["above", "below", "to the left of", "to the right of"]:
        rs = [r for r in results if r["relation"] == rel]
        yes = [r["margin"] for r in rs if r["gold"] == "yes"]
        no = [r["margin"] for r in rs if r["gold"] == "no"]
        auc = sum(1 for a in yes for b in no if a > b) / (len(yes) * len(no))
        acc = sum((r["margin"] > 0) == (r["gold"] == "yes") for r in rs) / len(rs)
        summary.append({"relation": rel, "n_pairs": len(yes), "auc": auc, "acc": acc,
                        "mean_yes": sum(yes) / len(yes), "mean_no": sum(no) / len(no)})
    Path(args.out).write_text(json.dumps({"summary": summary, "per_sample": results}, indent=2))
    print("saved", args.out)
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
