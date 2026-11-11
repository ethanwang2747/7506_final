import argparse, collections, json, random, statistics
from pathlib import Path

from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import capture, margin, prepare
from experiments.e2_ablation import unique_objects
from experiments.e3_regional import map_tokens, patch_s1, patch_s6
from experiments.build_val_pairs import OBJECT_VOCAB

CONFIG = ROOT / "data/derived/experiment_config.json"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--pairs-per-group", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results/e3/semantic_test_regional.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",") if args.groups else cfg["tasks"]["semantic"]["core_groups"]
    rows = [r for r in read_jsonl(ROOT / cfg["data"][f"semantic_pairs_{args.split}"]) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"semantic regional E3: {len(selected)} pairs")

    with open(SG_PATH) as f:
        sg = json.load(f)
    images = load_images([x for r in selected for x in (r["source_image_id"], r["target_image_id"])])
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    per_pair = []
    for r in selected:
        st = r["source_target"]["tokens_9"]
        tt = r["target_target"]["tokens_9"]
        if not st or not tt:
            continue
        used_s = {r["source_target"]["name"]}
        used_t = {r["target_target"]["name"]}
        objs_s = unique_objects(sg[r["source_image_id"]], set(OBJECT_VOCAB))
        objs_t = unique_objects(sg[r["target_image_id"]], set(OBJECT_VOCAB))
        cand_s = sorted([o for n, o in objs_s.items() if n not in used_s], key=lambda o: (abs(len(o["tokens_9"]) - len(st)), o["name"]))
        cand_t = sorted([o for n, o in objs_t.items() if n not in used_t], key=lambda o: (abs(len(o["tokens_9"]) - len(tt)), o["name"]))
        si = cand_s[0]["tokens_9"] if cand_s else []
        ti = cand_t[0]["tokens_9"] if cand_t else []

        src_in = prepare(proc, r["question"], images[r["source_image_id"]])
        tgt_in = prepare(proc, r["question"], images[r["target_image_id"]])
        pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
        v0 = int(pos[0])
        m_s = margin(model, src_in, yes_ids, no_ids)
        m_t = margin(model, tgt_in, yes_ids, no_ids)
        cap = capture(model, src_in, [6])
        rec = {"pair_id": r["pair_id"], "group": r["group"], "m_source": m_s, "m_target": m_t, "delta": m_s - m_t, "patch": {}}
        for region, src_toks, tgt_toks in [("target", st, tt), ("irrelevant", si, ti)]:
            if not src_toks or not tgt_toks:
                continue
            idx = map_tokens(src_toks, tgt_toks)
            m1 = patch_s1(model, tgt_in, tgt_toks, cap["S1"], idx, yes_ids, no_ids)
            m6 = patch_s6(model, tgt_in, tgt_toks, cap["S6"], idx, v0, yes_ids, no_ids)
            rec["patch"][f"S1:{region}"] = (m1 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
            rec["patch"][f"S6:{region}"] = (m6 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
        rng = random.Random(f"{r['pair_id']}:regional")
        cand_t = [i for i in range(81) if i not in set(tt) | set(ti)]
        cand_s = [i for i in range(81) if i not in set(st) | set(si)]
        n_rand = min(len(tt), len(cand_t), len(cand_s))
        if n_rand >= 1:
            rnd_t = sorted(rng.sample(cand_t, n_rand))
            rnd_s = sorted(rng.sample(cand_s, n_rand))
            m1 = patch_s1(model, tgt_in, rnd_t, cap["S1"], rnd_s, yes_ids, no_ids)
            m6 = patch_s6(model, tgt_in, rnd_t, cap["S6"], rnd_s, v0, yes_ids, no_ids)
            rec["patch"]["S1:random"] = (m1 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
            rec["patch"]["S6:random"] = (m6 - m_t) / (m_s - m_t) if abs(m_s - m_t) > 1e-8 else None
        per_pair.append(rec)

    summary = []
    for g in groups:
        pp = [r for r in per_pair if r["group"] == g]
        if not pp:
            continue
        row = {"group": g, "n_pairs": len(pp)}
        for key in ["S1:target", "S1:irrelevant", "S1:random", "S6:target", "S6:irrelevant", "S6:random"]:
            vals = [r["patch"][key] for r in pp if key in r["patch"] and r["patch"][key] is not None]
            row[key] = {"median": statistics.median(vals) if vals else None, "n": len(vals)}
        summary.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for r in summary:
        print(f"  {r['group']} n={r['n_pairs']}")
        for key in ["S1:target", "S1:irrelevant", "S1:random", "S6:target", "S6:irrelevant", "S6:random"]:
            x = r[key]
            if x["median"] is not None:
                print(f"    {key}: median={x['median']:+.2f} n={x['n']}")


if __name__ == "__main__":
    main()
