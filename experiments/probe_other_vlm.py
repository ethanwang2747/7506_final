"""Mechanistic side probe for Qwen2-VL-2B: visual tokens vs decision state.

For each of the 480 test-core examples we pool the visual tokens inside the
target/reference box and probe the object's horizontal side; we also probe the
last-token hidden state at selected decoder layers and read a logit-lens margin.

Output:
    results/other_vlm/qwen2vl_2b_probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT, build_prompt, load_images, yes_no_ids

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]
LAYERS = [8, 12, 16, 20, 24, 27]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select(records, max_per_group):
    counts, out = {}, []
    for r in records:
        if r["group"] not in GROUPS:
            continue
        n = counts.get(r["group"], 0)
        if max_per_group and n >= max_per_group:
            continue
        counts[r["group"]] = n + 1
        out.append(r)
    return out


def box_cells(cx, cy, w, h, grid_h, grid_w):
    c0 = max(0, int((cx - w / 2) * grid_w))
    c1 = min(grid_w - 1, int((cx + w / 2) * grid_w))
    r0 = max(0, int((cy - h / 2) * grid_h))
    r1 = min(grid_h - 1, int((cy + h / 2) * grid_h))
    return [r * grid_w + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2-VL-2B-Instruct")
    ap.add_argument("--pairs", default="data/derived/spatial_pairs_test.jsonl")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--max-pixels", type=int, default=384 * 384)
    ap.add_argument("--out", default="results/other_vlm/qwen2vl_2b_probe.json")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    proc.image_processor.max_pixels = args.max_pixels
    proc.image_processor.min_pixels = 64 * 64
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    yes_ids, no_ids = yes_no_ids(proc)

    rows = select(read_jsonl(ROOT / args.pairs), args.max_per_group)
    with open(ROOT / "data/derived/experiment_config.json") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["data"]["scene_graphs"]) as f:
        sg = json.load(f)
    images = load_images(sorted({r[f"{role}_image_id"] for r in rows
                                 for role in ("source", "target")}))

    cap = {}
    handles = []
    for i in LAYERS:
        def make_hook(idx):
            def f(mod, a, k):
                x = k.get("hidden_states", a[0] if a else None)
                cap[f"L{idx}"] = x[:, -1, :].detach().cpu().float()
            return f
        handles.append(model.model.language_model.layers[i].register_forward_pre_hook(
            make_hook(i), with_kwargs=True))
    def vis_hook(m, i, o):
        v = o.last_hidden_state if hasattr(o, "last_hidden_state") else (
            o[0] if isinstance(o, (tuple, list)) else o)
        cap["vis"] = v.detach()
    vis_handle = model.model.visual.register_forward_hook(vis_hook)

    vis_t, vis_r = [], []
    dec = {i: [] for i in LAYERS}
    lens = {i: [] for i in LAYERS}
    y, groups, sides_t, sides_r = [], [], [], []
    try:
        with torch.no_grad():
            for k, r in enumerate(rows):
                for role in ("source", "target"):
                    iid = r[f"{role}_image_id"]
                    inp = proc(text=[build_prompt(proc, r["question"])],
                               images=[images[iid]], return_tensors="pt")
                    inp = {kk: v.to("cuda") for kk, v in inp.items()}
                    model(**inp, use_cache=False)
                    thw = inp["image_grid_thw"][0].tolist()
                    grid_h, grid_w = thw[1] // 2, thw[2] // 2
                    v = cap["vis"]
                    if isinstance(v, (tuple, list)):
                        v = v[0]
                    v = v.reshape(-1, v.shape[-1]).float()
                    g = sg[iid]
                    W, H = g["width"], g["height"]
                    tb = r[f"{role}_target"]["bbox"]
                    rb = r[f"{role}_reference"]["bbox"]
                    tp = ((tb[0] + tb[2] / 2) / W, (tb[1] + tb[3] / 2) / H, tb[2] / W, tb[3] / H)
                    rp = ((rb[0] + rb[2] / 2) / W, (rb[1] + rb[3] / 2) / H, rb[2] / W, rb[3] / H)
                    ti = [i for i in box_cells(*tp, grid_h, grid_w) if i < v.shape[0]]
                    ri = [i for i in box_cells(*rp, grid_h, grid_w) if i < v.shape[0]]
                    if not ti or not ri:
                        continue
                    vis_t.append(v[ti].mean(0).cpu().numpy())
                    vis_r.append(v[ri].mean(0).cpu().numpy())
                    for i in LAYERS:
                        dec[i].append(cap[f"L{i}"][0].numpy())
                        h = cap[f"L{i}"].to("cuda").to(model.dtype)
                        lp = torch.log_softmax(model.lm_head(model.model.language_model.norm(h)).float(), -1)
                        lens[i].append(float(torch.logsumexp(lp[:, yes_ids], -1)
                                             - torch.logsumexp(lp[:, no_ids], -1)))
                    y.append(1 if role == "source" else 0)
                    groups.append(iid)
                    sides_t.append(1 if tp[0] < 0.5 else 0)
                    sides_r.append(1 if rp[0] < 0.5 else 0)
                if (k + 1) % 40 == 0:
                    print(f"  {k + 1}/{len(rows)}")
    finally:
        for h in handles:
            h.remove()
        vis_handle.remove()

    y = np.array(y)
    groups = np.array(groups)
    sides_t, sides_r = np.array(sides_t), np.array(sides_r)
    folds = list(GroupKFold(5).split(np.zeros(len(y)), groups=groups))

    def probe(X, yy):
        sc = np.full(len(yy), np.nan)
        for tr, te in folds:
            m = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
            clf.fit(m.transform(X[tr]), yy[tr])
            sc[te] = clf.decision_function(m.transform(X[te]))
        return float(roc_auc_score(yy, sc))

    out = {"model": args.model, "n": len(y), "layers": LAYERS, "probes": {}, "lens": {}}
    out["probes"]["visual_target_left"] = probe(np.stack(vis_t), sides_t)
    out["probes"]["visual_reference_left"] = probe(np.stack(vis_r), sides_r)
    for i in LAYERS:
        X = np.stack(dec[i])
        out["probes"][f"decision_target_left_L{i}"] = probe(X, sides_t)
        out["probes"][f"decision_reference_left_L{i}"] = probe(X, sides_r)
        out["lens"][f"L{i}"] = float(roc_auc_score(y, np.array(lens[i])))
    print(json.dumps(out, indent=1))
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
