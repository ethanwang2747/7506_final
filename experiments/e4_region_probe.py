import argparse, json
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


def pool(x, tokens):
    if not tokens:
        return None
    return x[0, tokens, :].mean(dim=0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/e4/region_probe.json"))
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = cfg["tasks"]["spatial"]["core_groups"] + cfg["tasks"]["spatial"]["exploratory_groups"]
    rows = {}
    for split in ["dev", "test"]:
        rows[split] = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{split}"]) if r["group"] in groups]
    ids = set()
    for split_rows in rows.values():
        for r in split_rows:
            ids.add(r["source_image_id"])
            ids.add(r["target_image_id"])
    images = load_images(ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")

    feats = {}
    for split, split_rows in rows.items():
        for r in split_rows:
            for role, image_id in [("source", r["source_image_id"]), ("target", r["target_image_id"])]:
                key = (image_id, r["question"])
                if key in feats:
                    continue
                inp = prepare(proc, r["question"], images[image_id])
                cap = capture(model, inp, [])
                s1 = cap["S1"]
                tkey = "source_target" if role == "source" else "target_target"
                rkey = "source_reference" if role == "source" else "target_reference"
                ft = pool(s1, r[tkey]["tokens_9"])
                fr = pool(s1, r[rkey]["tokens_9"]) if rkey in r else None
                if ft is None or fr is None:
                    continue
                feats[key] = np.concatenate([ft, fr, ft - fr])
        print(f"extracted {len(feats)} features ({split})")

    results = []
    for group in groups:
        Xtr, ytr, Xte, yte = [], [], [], []
        for split, X, y in [("dev", Xtr, ytr), ("test", Xte, yte)]:
            for r in rows[split]:
                if r["group"] != group:
                    continue
                k_s = (r["source_image_id"], r["question"])
                k_t = (r["target_image_id"], r["question"])
                if k_s not in feats or k_t not in feats:
                    continue
                X.append(feats[k_s]); y.append(1)
                X.append(feats[k_t]); y.append(0)
        Xtr, Xte = np.array(Xtr), np.array(Xte)
        ytr, yte = np.array(ytr), np.array(yte)
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(scaler.transform(Xtr), ytr)
        prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append({
            "group": group, "stage": "S1_region_pool",
            "n_train": len(ytr), "n_test": len(yte),
            "auc_test": float(roc_auc_score(yte, prob)),
            "acc_test": float((pred == yte).mean()),
            "majority_test": float(max(yte.mean(), 1 - yte.mean())),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for r in results:
        print(f"  {r['group']:40s} AUC={r['auc_test']:.3f} acc={r['acc_test']:.3f} maj={r['majority_test']:.3f} n={r['n_train']}/{r['n_test']}")


if __name__ == "__main__":
    main()
