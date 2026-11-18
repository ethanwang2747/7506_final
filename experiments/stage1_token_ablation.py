"""Stage 1: visual token ablation on the reference-counterfactual task.

For each pair we ablate target / reference / background / random visual tokens
(replaced by the stage mean vector) and measure whether the model still prefers
the clean image over the reference-moved image.

Output: results/diagnostics/stage1_token_ablation_<stage>_<tag>.json
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
ap.add_argument("--stage", default="S1", choices=["S1", "S6", "S12"])
ap.add_argument("--max-per-group", type=int, default=20)
ap.add_argument("--adapter", default=None)
ap.add_argument("--tag", default="base")
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
means = torch.load(ROOT / "data/derived/mean_vectors.pt", map_location="cpu")["means"]
images = load_images(sorted({r["source_image_id"] for r in sel}))
proc = load_processor(splitting=False)
model = load_model(attn="sdpa")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
yes, no = yes_no_ids(proc)
mean = means[args.stage].float()

def margin_ablate(img, question, idx):
    inp = prepare(proc, question, img)
    ids = inp["input_ids"][0]
    vpos = (ids == model.config.image_token_id).nonzero().flatten()
    v0 = int(vpos[0])
    handles = []
    if idx:
        if args.stage == "S1":
            def f(mod, a, out):
                x = out.clone()
                x[:, idx, :] = mean.to(device=x.device, dtype=x.dtype)
                return x
            handles.append(model.model.connector.register_forward_hook(f))
        else:
            layer = model.model.text_model.layers[int(args.stage[1:])]
            def f(mod, a, k):
                x = k.get("hidden_states", a[0] if a else None)
                x[:, [v0 + i for i in idx], :] = mean.to(device=x.device, dtype=x.dtype)
            handles.append(layer.register_forward_pre_hook(f, with_kwargs=True))
    try:
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
        return float(torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1))
    finally:
        for h in handles:
            h.remove()

out = {"stage": args.stage, "adapter": args.adapter, "n_pairs": len(sel), "conditions": {}}
per_pair = []
for k, r in enumerate(sel):
    iid = r["source_image_id"]
    img = images[iid]
    g = sg[iid]; W, H = g["width"], g["height"]
    x, y, w, h = r["source_reference"]["bbox"]
    crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
    moved = img.copy(); moved.paste(crop, (int(W - x - w), int(y)))
    tgt = list(r["source_target"]["tokens_9"])
    ref = list(r["source_reference"]["tokens_9"])
    rng = random.Random(f"{r['pair_id']}:{args.stage}")
    pool = [i for i in range(81) if i not in set(tgt) | set(ref)]
    bg = rng.sample(pool, min(len(tgt), len(pool)))
    rand = rng.sample(range(81), len(tgt))
    conds = {"original": [], "target": tgt, "reference": ref, "background": bg, "random": rand}
    rec = {"pair_id": r["pair_id"]}
    for cond, idx in conds.items():
        mc = margin_ablate(img, r["question"], idx)
        mm = margin_ablate(moved, r["question"], idx)
        rec[cond] = {"clean": mc, "moved": mm}
    per_pair.append(rec)
    if (k + 1) % 20 == 0:
        print(f"  {k + 1}/{len(sel)}")

for cond in ["original", "target", "reference", "background", "random"]:
    prefs = [1.0 if p[cond]["clean"] > p[cond]["moved"] else 0.0 for p in per_pair]
    deltas = [p[cond]["moved"] - p[cond]["clean"] for p in per_pair]
    flips = [1.0 if (p[cond]["clean"] > 0) != (p[cond]["moved"] > 0) else 0.0 for p in per_pair]
    out["conditions"][cond] = {
        "pair_pref": float(np.mean(prefs)),
        "mean_delta": float(np.mean(deltas)),
        "frac_flip": float(np.mean(flips)),
    }
    print(cond, {k: round(v, 3) for k, v in out["conditions"][cond].items()})
out["per_pair"] = per_pair
dst = ROOT / (args.out or f"results/diagnostics/stage1_token_ablation_{args.stage}_{args.tag}.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(f"saved {dst}")
