import argparse, collections, json, statistics
from pathlib import Path

from experiments.common import load_images, load_model, load_processor, batch_margins, yes_no_ids

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--max-groups", type=int, default=100)
    ap.add_argument("--pairs-per-group", type=int, default=6)
    ap.add_argument("--pair-offset", type=int, default=0)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--min-pairs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--splitting", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.pairs) if args.pairs else ROOT / f"data/derived/{args.task}_pairs_{args.split}.jsonl"
    rows = read_jsonl(path)
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    if args.groups:
        wanted = set(args.groups.split(","))
        by_group = {g: rs for g, rs in by_group.items() if g in wanted}
    groups = sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    groups = [(g, rs) for g, rs in groups if len(rs) >= args.min_pairs][:args.max_groups]
    print(f"{args.task} {args.split}: {len(rows)} pairs, {len(by_group)} groups, screening {len(groups)} groups")

    selected = []
    for group, rs in groups:
        selected.extend(rs[args.pair_offset:args.pair_offset + args.pairs_per_group])
    image_ids = set()
    for r in selected:
        image_ids.add(r["source_image_id"])
        image_ids.add(r["target_image_id"])
    images = load_images(image_ids)
    print(f"selected {len(selected)} pairs, {len(images)} images")

    proc = load_processor(splitting=bool(args.splitting))
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    # dedupe by (image_id, question, gold)
    task_map = {}
    for r in selected:
        for role in ["source", "target"]:
            key = (r[f"{role}_image_id"], r["question"], r[f"{role}_gold"])
            task_map.setdefault(key, {"image_id": key[0], "question": key[1], "gold": key[2]})
    tasks = list(task_map.values())
    for t in tasks:
        t["image"] = images[t["image_id"]]
    batch_size = 1 if args.splitting else args.batch_size
    margins = batch_margins(model, proc, tasks, yes_ids, no_ids, batch_size=batch_size)
    margin_by_key = {(t["image_id"], t["question"], t["gold"]): m for t, m in zip(tasks, margins)}
    print(f"forwarded {len(tasks)} unique (image, question) items")

    per_pair = []
    for r in selected:
        ms = margin_by_key[(r["source_image_id"], r["question"], r["source_gold"])]
        mt = margin_by_key[(r["target_image_id"], r["question"], r["target_gold"])]
        per_pair.append({
            "pair_id": r["pair_id"],
            "task": r["task"],
            "group": r["group"],
            "question": r["question"],
            "source_image_id": r["source_image_id"],
            "target_image_id": r["target_image_id"],
            "m_source": ms,
            "m_target": mt,
            "delta": ms - mt,
            "rank_correct": ms > mt,
        })

    summaries = []
    for group, rs in groups:
        pp = [x for x in per_pair if x["group"] == group]
        if len(pp) < args.min_pairs:
            continue
        src = [x["m_source"] for x in pp]
        tgt = [x["m_target"] for x in pp]
        a = auc(src, tgt)
        deltas = [x["delta"] for x in pp]
        summaries.append({
            "task": args.task,
            "group": group,
            "question": pp[0]["question"],
            "n_pairs": len(pp),
            "n_source_images": len(set(x["source_image_id"] for x in pp)),
            "n_target_images": len(set(x["target_image_id"] for x in pp)),
            "auc": a,
            "pair_acc": sum(x["rank_correct"] for x in pp) / len(pp),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "bal_acc_at_0": 0.5 * (sum(x > 0 for x in src) / len(src) + sum(x <= 0 for x in tgt) / len(tgt)),
            "source_yes_rate": sum(x > 0 for x in src) / len(src),
            "target_no_rate": sum(x <= 0 for x in tgt) / len(tgt),
        })

    summaries.sort(key=lambda x: (x["auc"] if x["auc"] is not None else 0, x["n_pairs"]), reverse=True)
    out = Path(args.out) if args.out else ROOT / f"results/behavior_gate/{args.task}_{args.split}_split{args.splitting}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summaries, "per_pair": per_pair}, f, indent=2)

    print(f"saved {out}")
    print("top groups:")
    for s in summaries[:25]:
        print(f"  AUC={s['auc']:.3f} pair_acc={s['pair_acc']:.2f} delta={s['mean_delta']:+.2f} "
              f"n={s['n_pairs']:2d} bal0={s['bal_acc_at_0']:.2f}  {s['group']}")


if __name__ == "__main__":
    main()
