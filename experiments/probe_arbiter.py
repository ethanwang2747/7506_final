"""Arbitrate the visual side probe: is 0.95 position leakage from the oracle box?

Conditions:
  original      cached S1/S6/S12 features
  gray          uniform gray image
  shuffled      image split into a 16x16 grid and permuted
  target_gray   target box painted gray

For each condition we pool the target/reference tokens with the same GQA boxes
and probe the object's horizontal side with grouped CV. If gray/shuffled stay
high, the probe reads the location of the pooled tokens rather than object
content.

Output:
    results/diagnostics/probe_arbiter.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, load_images, load_model, load_processor
from experiments.e0_validate import prepare
from experiments.readout_experiment import REGION_PATHS, attach_regions, read_jsonl

STAGES = ["S1", "S6", "S12"]


def load_payload():
    p = torch.load(ROOT / "results/readout/features_test_core.pt",
                   map_location="cpu", weights_only=False)
    p["k2i"] = {k: i for i, k in enumerate(p["keys"])}
    recs = {}
    for rp in REGION_PATHS:
        for r in read_jsonl(rp):
            recs[(r["pair_id"], "source")] = (r["source_target"], r["source_reference"])
            recs[(r["pair_id"], "target")] = (r["target_target"], r["target_reference"])
    attach_regions(p["examples"], recs)
    return p


def shuffle_patches(img, grid=16, seed=0):
    w, h = img.size
    pw, ph = w // grid, h // grid
    tiles = []
    for r in range(grid):
        for c in range(grid):
            tiles.append(img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tiles))
    out = Image.new("RGB", (pw * grid, ph * grid))
    for i, idx in enumerate(perm):
        out.paste(tiles[idx], ((i % grid) * pw, (i // grid) * ph))
    return out


def capture(model, proc, img, question):
    cap = {}
    handles = []
    handles.append(model.model.connector.register_forward_hook(
        lambda m, i, o: cap.__setitem__("S1", o.detach().to("cpu", torch.float16))))
    for i in (6, 12):
        def pre(idx):
            def f(mod, args, kwargs):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cap[f"S{idx}"] = x.detach().to("cpu", torch.float16)
            return f
        handles.append(model.model.text_model.layers[i].register_forward_pre_hook(
            pre(i), with_kwargs=True))
    try:
        with torch.no_grad():
            model(**prepare(proc, question, img), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    ids = prepare(proc, question, img)["input_ids"][0]
    vpos = (ids == model.config.image_token_id).nonzero().flatten()
    v0, v1 = int(vpos[0]), int(vpos[-1]) + 1
    return {"S1": cap["S1"][0],
            "S6": cap["S6"][0][v0:v1],
            "S12": cap["S12"][0][v0:v1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diagnostics/probe_arbiter.json")
    args = ap.parse_args()

    p = load_payload()
    ex = p["examples"]
    meta = {m["key"]: m for m in p["meta"]}
    groups = np.array([str(meta[e["key"]]["image_id"]) for e in ex])
    folds = list(GroupKFold(5).split(np.zeros(len(ex)), groups=groups))
    sides_t = np.array([1 if e["position"][0] < 0.5 else 0 for e in ex])
    sides_r = np.array([1 if e["position"][4] < 0.5 else 0 for e in ex])
    images = load_images(sorted({str(meta[e["key"]]["image_id"]) for e in ex}))
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")

    feats = {c: {s: {"t": [], "r": []} for s in STAGES}
             for c in ("original", "gray", "shuffled", "target_gray")}
    for s in STAGES:
        for e in ex:
            x = p[s][p["k2i"][e["key"]]].float()
            feats["original"][s]["t"].append(x[e["region_target"]].mean(0).numpy())
            feats["original"][s]["r"].append(x[e["region_ref"]].mean(0).numpy())

    for i, e in enumerate(ex):
        iid = str(meta[e["key"]]["image_id"])
        img = images[iid]
        g = meta[e["key"]]
        W, H = g["width"], g["height"]
        # target bbox from position (normalized cx,cy,w,h)
        cx, cy, bw, bh = e["position"][:4]
        box = ((cx - bw / 2) * W, (cy - bh / 2) * H,
               (cx + bw / 2) * W, (cy + bh / 2) * H)
        conds = {
            "gray": Image.new("RGB", img.size, (128, 128, 128)),
            "shuffled": shuffle_patches(img),
            "target_gray": img.copy(),
        }
        draw = ImageDraw.Draw(conds["target_gray"])
        draw.rectangle(box, fill=(128, 128, 128))
        for cond, im in conds.items():
            cap = capture(model, proc, im, meta[e["key"]]["question"])
            for s in STAGES:
                x = cap[s].float()
                feats[cond][s]["t"].append(x[e["region_target"]].mean(0).numpy())
                feats[cond][s]["r"].append(x[e["region_ref"]].mean(0).numpy())
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(ex)}")

    def probe(X, y):
        sc = np.full(len(y), np.nan)
        for tr, te in folds:
            m = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
            clf.fit(m.transform(X[tr]), y[tr])
            sc[te] = clf.decision_function(m.transform(X[te]))
        return float(roc_auc_score(y, sc))

    out = {"n": len(ex), "results": {}}
    for cond in feats:
        out["results"][cond] = {}
        for s in STAGES:
            Xt = np.stack(feats[cond][s]["t"])
            Xr = np.stack(feats[cond][s]["r"])
            out["results"][cond][s] = {
                "target_left": probe(Xt, sides_t),
                "reference_left": probe(Xr, sides_r),
            }
        print(cond, {s: round(out["results"][cond][s]["target_left"], 3) for s in STAGES})
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
