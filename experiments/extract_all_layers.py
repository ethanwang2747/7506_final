"""Extract last-token hidden states at every decoder layer.

Used by the tuned lens and head-level analyses. Saves one payload per split.

Usage:
  python experiments/extract_all_layers.py --pairs data/derived/spatial_pairs_test.jsonl \
      --groups "chair|to the left of|table,..." --max-per-group 40 --out results/diagnostics/all_layers_test.pt
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--groups", default=None)
ap.add_argument("--max-per-group", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.pairs if Path(args.pairs).is_absolute() else ROOT / args.pairs)]
if args.groups:
    gs = set(args.groups.split(","))
    rows = [r for r in rows if r["group"] in gs]
if args.max_per_group:
    counts, sel = {}, []
    for r in rows:
        n = counts.get(r["group"], 0)
        if n >= args.max_per_group:
            continue
        counts[r["group"]] = n + 1
        sel.append(r)
    rows = sel
images = load_images(sorted({r[f"{role}_image_id"] for r in rows for role in ("source", "target")}))
proc = load_processor(splitting=False)
model = load_model(attn="sdpa")
yes, no = yes_no_ids(proc)
n_layers = len(model.model.text_model.layers)
cap = {}
handles = []
for i in range(n_layers):
    def f(mod, a, k, i=i):
        x = k.get("hidden_states", a[0] if a else None)
        cap[i] = x[:, -1, :].detach().to("cpu", torch.float16)
    handles.append(model.model.text_model.layers[i].register_forward_pre_hook(f, with_kwargs=True))
feats, margins, meta = [], [], []
try:
    with torch.no_grad():
        for k, r in enumerate(rows):
            for role in ("source", "target"):
                inp = prepare(proc, r["question"], images[r[f"{role}_image_id"]])
                out = model(**inp, use_cache=False)
                feats.append(torch.stack([cap[i][0] for i in range(n_layers)]))
                lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                margins.append(float(torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)))
                meta.append({"pair_id": r["pair_id"], "group": r["group"], "role": role,
                             "label": 1 if role == "source" else 0,
                             "image_id": r[f"{role}_image_id"]})
            if (k + 1) % 40 == 0:
                print(f"  {k + 1}/{len(rows)}")
finally:
    for h in handles:
        h.remove()
payload = {"H": torch.stack(feats), "margin": torch.tensor(margins), "meta": meta,
           "n_layers": n_layers}
dst = ROOT / args.out
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, dst)
print(f"saved {dst} ({dst.stat().st_size/1e6:.1f} MB, {len(meta)} examples)")
