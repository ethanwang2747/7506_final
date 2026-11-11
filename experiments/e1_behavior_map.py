import argparse, collections, json, re, statistics
from pathlib import Path

import torch
from PIL import Image

from experiments.common import ROOT, batch_generate, batch_margins, build_prompt, load_images, load_model, load_processor, yes_no_ids

CONFIG = ROOT / "data/derived/experiment_config.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_yes_no(text):
    m = re.search(r"\b(yes|no)\b", text.strip().lower())
    return m.group(1) if m else text.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pairs-per-group", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--splitting", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = args.groups.split(",") if args.groups else cfg["tasks"][args.task]["core_groups"] + cfg["tasks"][args.task]["exploratory_groups"]
    path = Path(args.pairs) if args.pairs else ROOT / cfg["data"][f"{args.task}_pairs_{args.split}"]
    rows = [r for r in read_jsonl(path) if r["group"] in groups]
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    selected = []
    for g in groups:
        selected.extend(by_group.get(g, [])[:args.pairs_per_group])
    print(f"{args.task} {args.split}: selected {len(selected)} pairs from {len(by_group)} groups")

    # control question comes from a different group
    all_questions = sorted({r["question"] for r in rows})
    control_question = next(q for q in all_questions if q != selected[0]["question"])

    tasks = {}
    pair_meta = []
    for idx, r in enumerate(selected):
        g = r["group"]
        peers = by_group[g]
        peer = peers[(peers.index(r) + 1) % len(peers)]
        mismatch_image_id = peer["target_image_id"]
        if mismatch_image_id == r["source_image_id"]:
            mismatch_image_id = peer["source_image_id"]
        pair_meta.append({"pair": r, "mismatch_image_id": mismatch_image_id})

        specs = [
            ("source_clean", r["source_image_id"], r["question"]),
            ("target_clean", r["target_image_id"], r["question"]),
            ("source_gray", None, r["question"]),
            ("target_gray", None, r["question"]),
            ("source_mismatch_question", r["source_image_id"], control_question),
            ("source_mismatch_image", mismatch_image_id, r["question"]),
        ]
        for kind, image_id, question in specs:
            key = (kind, image_id, question)
            tasks.setdefault(key, {"kind": kind, "image_id": image_id, "question": question})

    image_ids = {t["image_id"] for t in tasks.values() if t["image_id"] is not None}
    images = load_images(image_ids)
    images[None] = Image.new("RGB", (384, 384), (127, 127, 127))
    for t in tasks.values():
        t["image"] = images[t["image_id"]]

    proc = load_processor(splitting=bool(args.splitting))
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)
    batch_size = 1 if args.splitting else args.batch_size
    task_list = list(tasks.values())
    margins = batch_margins(model, proc, task_list, yes_ids, no_ids, batch_size=batch_size)
    margin_by_key = {(t["kind"], t["image_id"], t["question"]): m for t, m in zip(task_list, margins)}
    print(f"forwarded {len(task_list)} unique controls")

    gen_tasks = [t for t in task_list if t["kind"] in ("source_clean", "target_clean")]
    texts = batch_generate(model, proc, gen_tasks, batch_size=batch_size)
    gen_by_key = {(t["kind"], t["image_id"], t["question"]): parse_yes_no(s) for t, s in zip(gen_tasks, texts)}

    per_pair = []
    for meta in pair_meta:
        r = meta["pair"]
        ms = margin_by_key[("source_clean", r["source_image_id"], r["question"])]
        mt = margin_by_key[("target_clean", r["target_image_id"], r["question"])]
        mg = margin_by_key[("source_gray", None, r["question"])]
        mgt = margin_by_key[("target_gray", None, r["question"])]
        mq = margin_by_key[("source_mismatch_question", r["source_image_id"], control_question)]
        mi = margin_by_key[("source_mismatch_image", meta["mismatch_image_id"], r["question"])]
        gs = gen_by_key[("source_clean", r["source_image_id"], r["question"])]
        gt = gen_by_key[("target_clean", r["target_image_id"], r["question"])]
        per_pair.append({
            "pair_id": r["pair_id"],
            "task": args.task,
            "group": r["group"],
            "question": r["question"],
            "source_image_id": r["source_image_id"],
            "target_image_id": r["target_image_id"],
            "m_source": ms,
            "m_target": mt,
            "delta": ms - mt,
            "rank_correct": ms > mt,
            "m_gray": mg,
            "m_gray_target": mgt,
            "gray_rank_correct": mg > mgt,
            "m_mismatch_question": mq,
            "m_mismatch_image": mi,
            "delta_image": ms - mi,
            "delta_question": ms - mq,
            "delta_gray": ms - mg,
            "gen_source": gs,
            "gen_target": gt,
            "gen_source_ok": gs == "yes",
            "gen_target_ok": gt == "no",
            "margin_source_ok": ms > 0,
            "margin_target_ok": mt <= 0,
            "gen_margin_agree_source": (gs == "yes") == (ms > 0),
            "gen_margin_agree_target": (gt == "no") == (mt <= 0),
        })

    summaries = []
    for g in groups:
        pp = [x for x in per_pair if x["group"] == g]
        if not pp:
            continue
        src = [x["m_source"] for x in pp]
        tgt = [x["m_target"] for x in pp]
        deltas = [x["delta"] for x in pp]
        summaries.append({
            "task": args.task,
            "group": g,
            "question": pp[0]["question"],
            "n_pairs": len(pp),
            "auc": sum(1 for s in src for t in tgt if s > t) / (len(src) * len(tgt)),
            "pair_acc": sum(x["rank_correct"] for x in pp) / len(pp),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "frac_delta_ge_0.5": sum(d >= 0.5 for d in deltas) / len(deltas),
            "mean_delta_image": statistics.mean(x["delta_image"] for x in pp),
            "mean_delta_question": statistics.mean(x["delta_question"] for x in pp),
            "mean_delta_gray": statistics.mean(x["delta_gray"] for x in pp),
            "mean_gray_margin": statistics.mean(x["m_gray"] for x in pp),
            "frac_gray_positive": sum(x["m_gray"] > 0 for x in pp) / len(pp),
            "gen_source_acc": sum(x["gen_source_ok"] for x in pp) / len(pp),
            "gen_target_acc": sum(x["gen_target_ok"] for x in pp) / len(pp),
            "margin_source_acc": sum(x["margin_source_ok"] for x in pp) / len(pp),
            "margin_target_acc": sum(x["margin_target_ok"] for x in pp) / len(pp),
            "gen_margin_agree": sum(x["gen_margin_agree_source"] and x["gen_margin_agree_target"] for x in pp) / len(pp),
        })

    out = Path(args.out) if args.out else ROOT / f"results/e1/{args.task}_{args.split}_split{args.splitting}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summaries, "per_pair": per_pair}, indent=2))
    print(f"saved {out}")
    for s in summaries:
        print(f"  AUC={s['auc']:.3f} delta={s['mean_delta']:+.2f} img={s['mean_delta_image']:+.2f} "
              f"q={s['mean_delta_question']:+.2f} gray={s['mean_delta_gray']:+.2f} gen={s['gen_source_acc']:.2f}/{s['gen_target_acc']:.2f}  {s['group']}")


if __name__ == "__main__":
    main()
