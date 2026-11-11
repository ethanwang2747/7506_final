import argparse, collections, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.common import ROOT, load_images, load_model, load_processor
from experiments.e0_validate import capture, prepare

CONFIG = ROOT / "data/derived/experiment_config.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-train", default="dev")
    ap.add_argument("--split-test", default="test")
    ap.add_argument("--out", default=str(ROOT / "results/e4/probe.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = cfg["tasks"]["spatial"]["core_groups"] + cfg["tasks"]["spatial"]["exploratory_groups"]
    rows = {}
    for split in [args.split_train, args.split_test]:
        rows[split] = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{split}"]) if r["group"] in groups]

    all_ids = set()
    for split_rows in rows.values():
        for r in split_rows:
            all_ids.add(r["source_image_id"])
            all_ids.add(r["target_image_id"])
    images = load_images(all_ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")

    feats = {}
    for split, split_rows in rows.items():
        for r in split_rows:
            for role, image_id, y in [("source", r["source_image_id"], 1), ("target", r["target_image_id"], 0)]:
                key = (image_id, r["question"])
                if key in feats:
                    continue
                inp = prepare(proc, r["question"], images[image_id])
                cap = capture(model, inp, [])
                feats[key] = {
                    "S0": cap["S0"].mean(dim=1).float().cpu().numpy().ravel(),
                    "S1": cap["S1"].mean(dim=1).float().cpu().numpy().ravel(),
                }
        print(f"extracted {len(feats)} unique features so far ({split})")

    results = []
    for group in groups:
        for stage in ["S0", "S1"]:
            Xtr, ytr, Xte, yte = [], [], [], []
            for split, X, y in [(args.split_train, Xtr, ytr), (args.split_test, Xte, yte)]:
                for r in rows[split]:
                    if r["group"] != group:
                        continue
                    X.append(feats[(r["source_image_id"], r["question"])][stage])
                    y.append(1)
                    X.append(feats[(r["target_image_id"], r["question"])][stage])
                    y.append(0)
            Xtr, Xte = np.array(Xtr), np.array(Xte)
            ytr, yte = np.array(ytr), np.array(yte)
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                continue
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            clf.fit(scaler.transform(Xtr), ytr)
            prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
            pred = (prob >= 0.5).astype(int)
            results.append({
                "group": group,
                "stage": stage,
                "n_train": len(ytr),
                "n_test": len(yte),
                "auc_test": float(roc_auc_score(yte, prob)),
                "acc_test": float((pred == yte).mean()),
                "majority_test": float(max(yte.mean(), 1 - yte.mean())),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for r in results:
        print(f"  {r['group']:40s} {r['stage']:3s} AUC={r['auc_test']:.2f} acc={r['acc_test']:.2f} maj={r['majority_test']:.2f} n={r['n_train']}/{r['n_test']}")


if __name__ == "__main__":
    main()
