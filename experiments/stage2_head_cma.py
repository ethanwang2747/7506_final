"""Stage 2b: causal mediation by patching attention-head outputs.

clean = original image (relation true); corrupted = reference moved to the
opposite side (relation false). We capture the clean o_proj input at the last
prompt token, then inject it into the corrupted run:
  --mode layer : replace all heads of one layer
  --mode head  : replace one head in selected layers

Recovery R = (m_patched - m_corrupt) / (m_clean - m_corrupt).

Output: results/diagnostics/stage2_cma_<mode>.json
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

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
ap.add_argument("--max-per-group", type=int, default=10)
ap.add_argument("--mode", choices=["layer", "head"], default="layer")
ap.add_argument("--layers", default="", help="comma list for head mode")
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
yes, no = yes_no_ids(proc)
n_layers = len(model.model.text_model.layers)
head_dim = model.config.text_config.head_dim if hasattr(model.config, "text_config") else 128
n_heads = model.config.text_config.num_attention_heads if hasattr(model.config, "text_config") else 16

def margin(inp):
    with torch.no_grad():
        out = model(**inp, use_cache=False)
    lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
    return float(torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1))

def capture_clean(inp):
    caps = {}
    handles = []
    for i in range(n_layers):
        def f(mod, a, i=i):
            x = a[0] if a else None
            caps[i] = x[:, -1, :].detach().clone()
        handles.append(model.model.text_model.layers[i].self_attn.o_proj.register_forward_pre_hook(f))
    try:
        m = margin(inp)
    finally:
        for h in handles:
            h.remove()
    return caps, m

def run_patched(inp, caps, patch):
    handles = []
    for i, spec in patch.items():
        def f(mod, a, i=i, spec=spec):
            x = a[0].clone()
            if spec == "all":
                x[:, -1, :] = caps[i]
            else:
                h = spec
                x[:, -1, h * head_dim:(h + 1) * head_dim] = caps[i][:, h * head_dim:(h + 1) * head_dim]
            return (x,) + tuple(a[1:])
        handles.append(model.model.text_model.layers[i].self_attn.o_proj.register_forward_pre_hook(f))
    try:
        return margin(inp)
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
    inp_c = prepare(proc, r["question"], img)
    inp_m = prepare(proc, r["question"], moved)
    caps, m_clean = capture_clean(inp_c)
    m_corr = margin(inp_m)
    rec = {"pair_id": r["pair_id"], "m_clean": m_clean, "m_corr": m_corr, "patched": {}}
    if args.mode == "layer":
        for i in range(n_layers):
            rec["patched"][str(i)] = run_patched(inp_m, caps, {i: "all"})
    else:
        layers = [int(v) for v in args.layers.split(",") if v != ""]
        for i in layers:
            for h in range(n_heads):
                rec["patched"][f"{i}:{h}"] = run_patched(inp_m, caps, {i: h})
    per_pair.append(rec)
    if (k + 1) % 10 == 0:
        print(f"  {k + 1}/{len(sel)}")

out = {"mode": args.mode, "n_pairs": len(sel), "n_layers": n_layers,
       "n_heads": n_heads, "head_dim": head_dim, "per_pair": per_pair, "recovery": {}}
keys = per_pair[0]["patched"].keys() if per_pair else []
for key in keys:
    vals = []
    for p in per_pair:
        denom = p["m_clean"] - p["m_corr"]
        if abs(denom) > 1e-6:
            vals.append((p["patched"][key] - p["m_corr"]) / denom)
    out["recovery"][key] = {"mean_R": float(np.mean(vals)) if vals else None,
                            "median_R": float(np.median(vals)) if vals else None, "n": len(vals)}
    print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["recovery"][key].items()})
dst = ROOT / (args.out or f"results/diagnostics/stage2_cma_{args.mode}.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(f"saved {dst}")
