"""Layer-wise logit lens: at which layer does the yes/no answer become readable?

We take the hidden state at the final prompt token after each text layer,
apply the model's own final norm and LM head, and read the yes/no margin.
This is a diagnostic: it shows when the answer becomes linearly decodable
from the residual stream, not whether that layer causes the answer.

Reported per layer: accuracy on balanced source/target pairs, AUC against the
label, correlation with the final margin, and the fraction of examples whose
lens answer already equals the final answer.

Example:
    python experiments/logit_lens.py \
        --pairs data/derived/spatial_pairs_test_strict.jsonl \
        --max-per-group 5 --batch-size 8 \
        --out results/diagnostics/logit_lens_test.json
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (ROOT, build_prompt, load_images, load_model, load_processor,
                                yes_no_ids)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records, groups, max_per_group, max_pairs=0, seed=0):
    if groups:
        records = [r for r in records if r["group"] in set(groups)]
    counts, out = {}, []
    for r in records:
        n = counts.get(r["group"], 0)
        if max_per_group and n >= max_per_group:
            continue
        counts[r["group"]] = n + 1
        out.append(r)
    if max_pairs and len(out) > max_pairs:
        by_rel = collections.defaultdict(list)
        for r in out:
            by_rel[r["group"].split("|")[1]].append(r)
        rng = random.Random(seed)
        per = max(1, max_pairs // max(1, len(by_rel)))
        picked = []
        for rel, rs in by_rel.items():
            rng.shuffle(rs)
            picked.extend(rs[:per])
        out = picked[:max_pairs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test_strict.jsonl")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--max-per-group", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="results/diagnostics/logit_lens_test.json")
    args = ap.parse_args()

    pair_path = Path(args.pairs) if Path(args.pairs).is_absolute() else ROOT / args.pairs
    rows = select(read_jsonl(pair_path), args.groups.split(",") if args.groups else None,
                  args.max_per_group, args.max_pairs, args.seed)
    if not rows:
        raise SystemExit("no pairs selected")

    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    yes_ids, no_ids = yes_no_ids(proc)
    core = model.base_model.model if hasattr(model, "base_model") else model
    n_layers = len(core.model.text_model.layers)
    print(f"{len(rows)} pairs, {n_layers} layers, batch {args.batch_size}")

    items = []
    for r in rows:
        for role in ("source", "target"):
            items.append({"pair_id": r["pair_id"], "group": r["group"], "role": role,
                          "image_id": r[f"{role}_image_id"], "question": r["question"]})
    images = load_images(sorted({it["image_id"] for it in items}))

    caps = {}

    def hook(i):
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            caps[i] = x[:, -1, :].detach()
        return f

    handles = [core.model.text_model.layers[i].register_forward_pre_hook(hook(i), with_kwargs=True)
               for i in range(n_layers)]

    lens, finals = [], []
    try:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            prompts = [build_prompt(proc, it["question"]) for it in batch]
            imgs = [[images[it["image_id"]]] for it in batch]
            inputs = proc(text=prompts, images=imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, use_cache=False)
                logits = out.logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                finals.append((torch.logsumexp(lp[:, yes_ids], -1)
                               - torch.logsumexp(lp[:, no_ids], -1)).detach().cpu())
                row = []
                for i in range(n_layers):
                    h = core.model.text_model.norm(caps[i].to(model.dtype))
                    lp_i = torch.log_softmax(core.lm_head(h).float(), dim=-1)
                    row.append((torch.logsumexp(lp_i[:, yes_ids], -1)
                                - torch.logsumexp(lp_i[:, no_ids], -1)).detach().cpu())
                lens.append(torch.stack(row, dim=1))  # (batch, layers)
            if (start // args.batch_size + 1) % 5 == 0:
                print(f"  {start + len(batch)}/{len(items)}")
    finally:
        for h in handles:
            h.remove()

    L = torch.cat(lens).numpy()
    f = torch.cat(finals).numpy()
    y = np.array([1 if it["role"] == "source" else 0 for it in items])

    per_layer = []
    for i in range(n_layers):
        m = L[:, i]
        acc = float(((m > 0).astype(int) == y).mean())
        auc = float(roc_auc_score(y, m))
        corr = float(np.corrcoef(m, f)[0, 1]) if np.std(m) > 0 else None
        agree = float(((m > 0) == (f > 0)).mean())
        per_layer.append({"layer": i, "acc": acc, "auc": auc,
                          "corr_with_final": corr, "agree_with_final": agree,
                          "mean_margin": float(m.mean())})
        cs = f"{corr:.3f}" if corr is not None else "nan"
        print(f"  L{i:02d} acc {acc:.3f} auc {auc:.3f} corr {cs} agree {agree:.3f}")

    best = max(per_layer, key=lambda d: d["auc"])
    summary = {
        "pairs": len(rows),
        "examples": int(len(y)),
        "n_layers": n_layers,
        "final_acc": float(((f > 0).astype(int) == y).mean()),
        "final_auc": float(roc_auc_score(y, f)),
        "best_layer": best["layer"],
        "best_layer_auc": best["auc"],
        "best_layer_acc": best["acc"],
        "per_layer": per_layer,
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "per_example": [{"pair_id": it["pair_id"], "role": it["role"], "image_id": it["image_id"],
                         "lens": [round(float(v), 4) for v in L[j]], "final": round(float(f[j]), 4)}
                        for j, it in enumerate(items)],
    }, indent=1))
    print(f"saved {out}")
    print(f"final acc {summary['final_acc']:.3f} auc {summary['final_auc']:.3f}; "
          f"best layer {summary['best_layer']} auc {summary['best_layer_auc']:.3f}")


if __name__ == "__main__":
    main()
