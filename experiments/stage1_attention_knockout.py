"""Stage 1: attention knockout from text tokens to target/reference visual tokens.

Blocks attention edges (text query -> selected visual key) in chosen decoder
layers, on the reference-counterfactual task. Random-token control included.

Output: results/diagnostics/stage1_attention_knockout_<tag>.json
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]
GROUPS_LAYERS = {"early": list(range(0, 8)), "mid": list(range(8, 16)),
                 "late": list(range(16, 24)), "all": list(range(24))}

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
ap.add_argument("--max-per-group", type=int, default=10)
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
images = load_images(sorted({r["source_image_id"] for r in sel}))
proc = load_processor(splitting=False)
model = load_model(attn="sdpa")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
yes, no = yes_no_ids(proc)

def margin_knockout(img, question, layers, idx):
    inp = prepare(proc, question, img)
    ids = inp["input_ids"][0]
    vpos = (ids == model.config.image_token_id).nonzero().flatten()
    v0, v1 = int(vpos[0]), int(vpos[-1]) + 1
    q = ids.shape[0]
    neg = torch.finfo(torch.bfloat16).min
    mask = torch.triu(torch.full((q, q), neg, device="cuda", dtype=torch.bfloat16), 1)
    inp["attention_mask"] = mask.unsqueeze(0).unsqueeze(0)
    handles = []
    for li in layers:
        def f(mod, a, k, li=li):
            m = k.get("attention_mask")
            if m is None:
                return
            m = m.clone()
            m[:, :, v1:, [v0 + i for i in idx]] = neg
            k["attention_mask"] = m
        handles.append(model.model.text_model.layers[li].self_attn.register_forward_pre_hook(
            f, with_kwargs=True))
    try:
        with torch.no_grad():
            out = model(**inp, use_cache=False)
        lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
        return float(torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1))
    finally:
        for h in handles:
            h.remove()

per_pair = []
for k, r in enumerate(sel):
    iid = r["source_image_id"]; img = images[iid]
    g = sg[iid]; W, H = g["width"], g["height"]
    x, y, w, h = r["source_reference"]["bbox"]
    crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
    moved = img.copy(); moved.paste(crop, (int(W - x - w), int(y)))
    tgt = list(r["source_target"]["tokens_9"]); ref = list(r["source_reference"]["tokens_9"])
    rng = random.Random(f"{r['pair_id']}:ko")
    rand = rng.sample(range(81), len(tgt))
    rec = {"pair_id": r["pair_id"]}
    for gname, layers in GROUPS_LAYERS.items():
        for cname, idx in (("target", tgt), ("reference", ref), ("random", rand)):
            mc = margin_knockout(img, r["question"], layers, idx)
            mm = margin_knockout(moved, r["question"], layers, idx)
            rec[f"{gname}:{cname}"] = {"clean": mc, "moved": mm}
    per_pair.append(rec)
    if (k + 1) % 10 == 0:
        print(f"  {k + 1}/{len(sel)}")

out = {"adapter": args.adapter, "n_pairs": len(sel), "conditions": {}, "per_pair": per_pair}
for gname in GROUPS_LAYERS:
    for cname in ("target", "reference", "random"):
        key = f"{gname}:{cname}"
        prefs = [1.0 if p[key]["clean"] > p[key]["moved"] else 0.0 for p in per_pair]
        deltas = [p[key]["moved"] - p[key]["clean"] for p in per_pair]
        out["conditions"][key] = {"pair_pref": float(np.mean(prefs)),
                                  "mean_delta": float(np.mean(deltas))}
        print(key, {k: round(v, 3) for k, v in out["conditions"][key].items()})
dst = ROOT / (args.out or f"results/diagnostics/stage1_attention_knockout_{args.tag}.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(f"saved {dst}")
