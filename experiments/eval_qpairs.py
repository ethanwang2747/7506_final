"""Evaluate same-image question pairs (position-controlled).

For each image we ask the true relation and its opposite. A model that reads the
relation should assign a higher yes/no margin to the true question; a model that
only reads the target's absolute position cannot separate them.

Output: results/sft/eval_qpairs_<tag>.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, batch_margins, load_images, load_model, load_processor, yes_no_ids

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/derived/spatial_qpairs_test.jsonl")
ap.add_argument("--adapter", default=None)
ap.add_argument("--tag", default="base")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows = [json.loads(l) for l in open(ROOT / args.pairs)]
images = load_images(sorted({r["image_id"] for r in rows}))
proc = load_processor(splitting=False)
model = load_model(attn="sdpa")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
yes, no = yes_no_ids(proc)
items = [{"question": r["question"], "image": images[r["image_id"]]} for r in rows]
m = np.array(batch_margins(model, proc, items, yes, no, args.batch_size))
y = np.array([r["label"] for r in rows])

# pair-level: true question should have a higher margin than its opposite
by = {}
for r, s in zip(rows, m):
    key = r["pair_id"].rsplit(":", 1)[0]
    by.setdefault(key, {})[r["pair_id"].rsplit(":", 1)[1]] = s
deltas = [v["true"] - v["opp"] for v in by.values() if "true" in v and "opp" in v]
out = {"adapter": args.adapter, "n": len(y),
       "acc": float(((m > 0).astype(int) == y).mean()),
       "auc": float(roc_auc_score(y, m)),
       "pair_pref_acc": float(np.mean([d > 0 for d in deltas])),
       "mean_delta": float(np.mean(deltas)),
       "per_group": {}}
for g in sorted({r["group"] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r["group"] == g]
    out["per_group"][g] = {"n": len(idx),
                           "acc": float(((m[idx] > 0).astype(int) == y[idx]).mean()),
                           "auc": float(roc_auc_score(y[idx], m[idx]))}
dst = ROOT / (args.out or f"results/sft/eval_qpairs_{args.tag}.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("n", "acc", "auc", "pair_pref_acc", "mean_delta")}, indent=1))
for g, v in out["per_group"].items():
    print(f"  {g:32s} acc {v['acc']:.3f} auc {v['auc']:.3f}")
print(f"saved {dst}")
