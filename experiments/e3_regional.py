import argparse, collections, json, math, random, statistics
from pathlib import Path

import torch

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import capture, margin, prepare
from experiments.e2_ablation import unique_objects
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
STAGES = ["S1", "S6"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rc(t):
    return divmod(t, 9)


def map_tokens(src_tokens, tgt_tokens):
    if not src_tokens or not tgt_tokens:
        return None
    srs = [rc(t)[0] for t in src_tokens]
    scs = [rc(t)[1] for t in src_tokens]
    trs = [rc(t)[0] for t in tgt_tokens]
    tcs = [rc(t)[1] for t in tgt_tokens]
    sr0, sr1, sc0, sc1 = min(srs), max(srs), min(scs), max(scs)
    tr0, tr1, tc0, tc1 = min(trs), max(trs), min(tcs), max(tcs)
    out = []
    for r, c in zip(trs, tcs):
        fr = 0.0 if tr1 == tr0 else (r - tr0) / (tr1 - tr0)
        fc = 0.0 if tc1 == tc0 else (c - tc0) / (tc1 - tc0)
        rs = round(sr0 + fr * (sr1 - sr0))
        cs = round(sc0 + fc * (sc1 - sc0))
        out.append(rs * 9 + cs)
    return out


def patch_s1(model, inp, tgt_tokens, src_s1, src_idx, yes_ids, no_ids):
    def f(mod, args, out):
        x = out.clone()
        for t, s in zip(tgt_tokens, src_idx):
            x[:, t, :] = src_s1[:, s, :]
        return x
    h = model.model.connector.register_forward_hook(f)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def patch_s6(model, inp, tgt_tokens, src_s6, src_idx, v0, yes_ids, no_ids):
    layer = model.model.text_model.layers[6]
    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        for t, s in zip(tgt_tokens, src_idx):
            x[:, v0 + t, :] = src_s6[:, v0 + s, :]
    h = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inp, yes_ids, no_ids)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs-per-group", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results/e3/spatial_test_regional.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = cfg["tasks"]["spatial"]["core_groups"] + cfg["tasks"]["spatial"]["exploratory_groups"]
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{args.split}"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"regional E3: {len(selected)} pairs")

    with open(ROOT / "data/raw/gqa/val_sceneGraphs.json") as f:
        sg = json.load(f)
    images = load_images([x for r in selected for x in (r["source_image_id"], r["target_image_id"])])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for idx, r in enumerate(selected):
        st = r["source_target"]["tokens_9"]
        tt = r["target_target"]["tokens_9"]
        sr = r["source_reference"]["tokens_9"] if "source_reference" in r else []
        tr = r["target_reference"]["tokens_9"] if "target_reference" in r else []
        if not st or not tt:
            continue
        src_in = prepare(proc, r["question"], images[r["source_image_id"]])
        tgt_in = prepare(proc, r["question"], images[r["target_image_id"]])
        pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0, v1 = int(pos[0]), int(pos[-1]) + 1
        m_s = margin(model, src_in, yes_ids, no_ids)
        m_t = margin(model, tgt_in, yes_ids, no_ids)
        cap = capture(model, src_in, [6])
        src_s1 = cap["S1"]
        src_s6 = cap["S6"]

        rec = {"pair_id": r["pair_id"], "group": r["group"], "m_source": m_s, "m_target": m_t,
               "delta": m_s - m_t, "patch": {}}
        used_s = {r["source_target"]["name"]} | ({r["source_reference"]["name"]} if "source_reference" in r else set())
        used_t = {r["target_target"]["name"]} | ({r["target_reference"]["name"]} if "target_reference" in r else set())
        objs_s = unique_objects(sg[r["source_image_id"]], set(OBJECT_VOCAB))
        objs_t = unique_objects(sg[r["target_image_id"]], set(OBJECT_VOCAB))
        cand_s = sorted([o for n, o in objs_s.items() if n not in used_s], key=lambda o: (abs(len(o["tokens_9"]) - len(st)), o["name"]))
        cand_t = sorted([o for n, o in objs_t.items() if n not in used_t], key=lambda o: (abs(len(o["tokens_9"]) - len(tt)), o["name"]))
        si = cand_s[0]["tokens_9"] if cand_s else []
        ti = cand_t[0]["tokens_9"] if cand_t else []
        for region, src_toks, tgt_toks in [("target", st, tt), ("reference", sr, tr), ("irrelevant", si, ti)]:
            if not src_toks or not tgt_toks:
                continue
            src_idx = map_tokens(src_toks, tgt_toks)
            m1 = patch_s1(model, tgt_in, tgt_toks, src_s1, src_idx, yes_ids, no_ids)
            m6 = patch_s6(model, tgt_in, tgt_toks, src_s6, src_idx, v0, yes_ids, no_ids)
            rec["patch"][f"S1:{region}"] = (m1 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
            rec["patch"][f"S6:{region}"] = (m6 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
        # random control matched to target token count, outside object regions in both images
        rng = random.Random(f"{r['pair_id']}:regional")
        cand_t = [i for i in range(81) if i not in set(tt) | set(tr) | set(ti)]
        cand_s = [i for i in range(81) if i not in set(st) | set(sr) | set(si)]
        n_rand = min(len(tt), len(cand_t), len(cand_s))
        if n_rand >= 1:
            rnd_t = sorted(rng.sample(cand_t, n_rand))
            rnd_s = sorted(rng.sample(cand_s, n_rand))
            m1 = patch_s1(model, tgt_in, rnd_t, src_s1, rnd_s, yes_ids, no_ids)
            m6 = patch_s6(model, tgt_in, rnd_t, src_s6, rnd_s, v0, yes_ids, no_ids)
            rec["patch"]["S1:random"] = (m1 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
            rec["patch"]["S6:random"] = (m6 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
        per_pair.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp)}
        for key in ["S1:target", "S1:reference", "S1:irrelevant", "S1:random", "S6:target", "S6:reference", "S6:irrelevant", "S6:random"]:
            vals = [r["patch"][key] for r in pp if key in r["patch"] and r["patch"][key] is not None]
            row[key] = {"median": statistics.median(vals) if vals else None,
                        "mean": statistics.mean(vals) if vals else None,
                        "frac_gt_half": sum(v > 0.5 for v in vals) / len(vals) if vals else None,
                        "n": len(vals)}
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']}")
        for key in ["S1:target", "S1:reference", "S1:irrelevant", "S1:random", "S6:target", "S6:reference", "S6:irrelevant", "S6:random"]:
            x = r[key]
            if x["median"] is not None:
                print(f"    {key}: median={x['median']:+.2f} frac>0.5={x['frac_gt_half']:.2f} n={x['n']}")


if __name__ == "__main__":
    main()
