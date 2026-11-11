import argparse, collections, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.common import ROOT, load_images, load_model, load_processor
from experiments.e0_validate import capture, prepare
from experiments.e2_ablation import unique_objects

CONFIG = ROOT / "data/derived/experiment_config.json"
SG_PATH = ROOT / "data/raw/gqa/val_sceneGraphs.json"
STAGES = ["S0", "S1", "S6", "S12"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def expand27(tokens):
    out = []
    for t in tokens:
        r, c = divmod(t, 9)
        for dr in range(3):
            for dc in range(3):
                out.append((3 * r + dr) * 27 + 3 * c + dc)
    return out


def pool(x, tokens, stage):
    idx = expand27(tokens) if stage == "S0" else tokens
    return x[0, idx, :].mean(dim=0).float().cpu().numpy()


def position_feat(obj, ref, W, H):
    def box(o):
        x, y, w, h = o["bbox"]
        return np.array([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H], dtype=np.float32)
    a, b = box(obj), box(ref)
    return np.concatenate([a, b, a - b])


def bootstrap_auc(y, p, n=500, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/e4/stage_probe.json"))
    ap.add_argument("--pos-threshold", type=float, default=0.2)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    groups = cfg["tasks"]["spatial"]["core_groups"] + cfg["tasks"]["spatial"]["exploratory_groups"]
    rows = {}
    for split in ["dev", "test"]:
        rows[split] = [r for r in read_jsonl(ROOT / cfg["data"][f"spatial_pairs_{split}"]) if r["group"] in groups]
    ids = set()
    for split_rows in rows.values():
        for r in split_rows:
            ids.add(r["source_image_id"]); ids.add(r["target_image_id"])
    images = load_images(ids)
    proc = load_processor(splitting=False)
    model = load_model(attn="sdpa")
    with open(SG_PATH) as f:
        sg = json.load(f)

    feats = {}
    for split, split_rows in rows.items():
        for r in split_rows:
            for role, image_id in [("source", r["source_image_id"]), ("target", r["target_image_id"])]:
                key = (image_id, r["question"])
                if key in feats:
                    continue
                inp = prepare(proc, r["question"], images[image_id])
                pos = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
                v0 = int(pos[0])
                cap = capture(model, inp, [6, 12])
                tkey = "source_target" if role == "source" else "target_target"
                rkey = "source_reference" if role == "source" else "target_reference"
                obj, ref = r[tkey], r[rkey]
                g = sg[image_id]
                f = {}
                for stage in STAGES:
                    x = cap[stage]
                    if stage in ("S6", "S12"):
                        x = x[:, v0:v0 + 81, :]
                    f[stage] = np.concatenate([pool(x, obj["tokens_9"], stage), pool(x, ref["tokens_9"], stage)])
                    f[stage] = np.concatenate([f[stage], f[stage][:f[stage].shape[0] // 2] - f[stage][f[stage].shape[0] // 2:]])
                f["position"] = position_feat(obj, ref, g["width"], g["height"])
                feats[key] = f
        print(f"extracted {len(feats)} ({split})")

    results = []
    for group in groups:
        for stage in STAGES + ["position"]:
            Xtr, ytr, Xte, yte = [], [], [], []
            Ptr, Pte = [], []
            for split, X, y, P in [("dev", Xtr, ytr, Ptr), ("test", Xte, yte, Pte)]:
                for r in rows[split]:
                    if r["group"] != group:
                        continue
                    for role, image_id in [("source", r["source_image_id"]), ("target", r["target_image_id"])]:
                        key = (image_id, r["question"])
                        if key in feats:
                            X.append(feats[key][stage]); y.append(1 if role == "source" else 0)
                            P.append(feats[key]["position"])
            Xtr, Xte = np.array(Xtr), np.array(Xte)
            ytr, yte = np.array(ytr), np.array(yte)
            Ptr, Pte = np.array(Ptr), np.array(Pte)
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                continue
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
            clf.fit(scaler.transform(Xtr), ytr)
            prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
            auc = float(roc_auc_score(yte, prob))
            # random-label control
            rng = np.random.default_rng(0)
            rand = []
            for _ in range(5):
                ysh = rng.permutation(ytr)
                clf2 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
                clf2.fit(scaler.transform(Xtr), ysh)
                rand.append(float(roc_auc_score(yte, clf2.predict_proba(scaler.transform(Xte))[:, 1])))
            resid = LinearRegression().fit(Ptr, Xtr)
            Xr_tr = Xtr - resid.predict(Ptr)
            Xr_te = Xte - resid.predict(Pte)
            scaler_r = StandardScaler().fit(Xr_tr)
            clf_r = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
            clf_r.fit(scaler_r.transform(Xr_tr), ytr)
            prob_r = clf_r.predict_proba(scaler_r.transform(Xr_te))[:, 1]
            # position-matched subset: source/target target centers close
            Xmtr, ymtr, Xmte, ymte = [], [], [], []
            for split, X, y in [("dev", Xmtr, ymtr), ("test", Xmte, ymte)]:
                for r in rows[split]:
                    if r["group"] != group:
                        continue
                    ks = (r["source_image_id"], r["question"]); kt = (r["target_image_id"], r["question"])
                    if ks not in feats or kt not in feats:
                        continue
                    d = float(np.linalg.norm(feats[ks]["position"][:2] - feats[kt]["position"][:2]))
                    if d <= args.pos_threshold:
                        X.append(feats[ks][stage]); y.append(1)
                        X.append(feats[kt][stage]); y.append(0)
            matched_auc = None
            if len(ymtr) >= 10 and len(ymte) >= 10 and len(set(ymtr)) > 1 and len(set(ymte)) > 1:
                sc = StandardScaler().fit(Xmtr)
                clf_m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
                clf_m.fit(sc.transform(Xmtr), ymtr)
                matched_auc = float(roc_auc_score(ymte, clf_m.predict_proba(sc.transform(Xmte))[:, 1]))
            results.append({
                "group": group, "stage": stage, "n_train": len(ytr), "n_test": len(yte),
                "matched_auc": matched_auc, "matched_n_train": len(ymtr), "matched_n_test": len(ymte),
                "auc": auc, "auc_ci95": bootstrap_auc(yte, prob),
                "residual_auc": float(roc_auc_score(yte, prob_r)),
                "residual_auc_ci95": bootstrap_auc(yte, prob_r),
                "random_label_auc": float(np.mean(rand)),
                "acc": float(((prob >= 0.5) == yte).mean()),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for r in results:
        print(f"  {r['group']:40s} {r['stage']:8s} AUC={r['auc']:.2f} CI[{r['auc_ci95'][0]:.2f},{r['auc_ci95'][1]:.2f}] rand={r['random_label_auc']:.2f} n={r['n_train']}/{r['n_test']}")


if __name__ == "__main__":
    main()
