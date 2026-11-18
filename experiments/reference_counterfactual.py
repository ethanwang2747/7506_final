"""Reference-only counterfactual: keep the target fixed, move the reference.

For each source image (relation true) we create:
  original     unchanged
  ref_hflip    reference bbox cut and pasted at the mirrored x
  ref_vflip    reference bbox cut and pasted at the mirrored y (control)
  ref_gray     reference bbox painted gray

If the model compares the two objects, ref_hflip should lower or flip the
margin; the vertical move is a control that does not change a left/right
relation.

Output: results/diagnostics/reference_counterfactual.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, batch_margins, load_images, load_model, load_processor, yes_no_ids

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
ap.add_argument("--adapter", default=None)
ap.add_argument("--tag", default="base")
ap.add_argument("--max-per-group", type=int, default=40)
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows = [json.loads(l) for l in open(ROOT / args.pairs)]
rows = [r for r in rows if r["group"] in GROUPS]
counts, sel = {}, []
for r in rows:
    n = counts.get(r["group"], 0)
    if n >= args.max_per_group:
        continue
    counts[r["group"]] = n + 1
    sel.append(r)
sg = json.load(open(ROOT / "data/raw/gqa/val_sceneGraphs.json"))
images = load_images(sorted({r["source_image_id"] for r in sel}))
proc = load_processor(splitting=False)
model = load_model(attn="sdpa")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
yes, no = yes_no_ids(proc)

items, meta = [], []
for r in sel:
    iid = r["source_image_id"]
    img = images[iid]
    g = sg[iid]; W, H = g["width"], g["height"]
    x, y, w, h = r["source_reference"]["bbox"]
    box = (int(x), int(y), int(x + w), int(y + h))
    crop = img.crop(box)
    conds = {"original": img}
    mh = img.copy(); mh.paste(crop, (int(W - x - w), int(y))); conds["ref_hflip"] = mh
    mv = img.copy(); mv.paste(crop, (int(x), int(H - y - h))); conds["ref_vflip"] = mv
    mg = img.copy(); ImageDraw.Draw(mg).rectangle(box, fill=(128, 128, 128)); conds["ref_gray"] = mg
    for cond, im in conds.items():
        items.append({"question": r["question"], "image": im})
        meta.append({"pair_id": r["pair_id"], "group": r["group"], "cond": cond})

m = np.array(batch_margins(model, proc, items, yes, no, 8))
by = {}
for x, s in zip(meta, m):
    by.setdefault(x["pair_id"], {})[x["cond"]] = float(s)
out = {"adapter": args.adapter, "n_pairs": len(by), "conditions": {}, "per_pair": by}
for cond in ("ref_hflip", "ref_vflip", "ref_gray"):
    deltas = [v[cond] - v["original"] for v in by.values()]
    flips = [1.0 if (v[cond] > 0) != (v["original"] > 0) else 0.0 for v in by.values()]
    out["conditions"][cond] = {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "frac_flip": float(np.mean(flips)),
        "frac_drop_0.5": float(np.mean([d < -0.5 for d in deltas])),
    }
    print(cond, {k: round(v, 3) for k, v in out["conditions"][cond].items()})
# pair preference: clean margin > moved margin
imgs = {m["pair_id"]: m.get("image_id") for m in meta}
for cond in ("ref_hflip", "ref_vflip", "ref_gray"):
    prefs = [1.0 if v["original"] > v[cond] else 0.0 for v in by.values()]
    out["conditions"][cond]["pair_pref"] = float(np.mean(prefs))
    rng = np.random.default_rng(0)
    vals = np.array(prefs)
    boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
    out["conditions"][cond]["pair_pref_ci"] = [float(np.percentile(boot, 2.5)),
                                               float(np.percentile(boot, 97.5))]
    print(cond, "pair_pref", round(out["conditions"][cond]["pair_pref"], 3),
          out["conditions"][cond]["pair_pref_ci"])
dst = ROOT / (args.out or f"results/diagnostics/reference_counterfactual_{args.tag}.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(f"saved {dst}")
