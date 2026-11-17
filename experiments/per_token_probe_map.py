"""Map where information is linearly readable across the 9x9 visual token grid.

For every visual token cell (9x9 after pixel shuffle) we train a separate
linear probe on that cell's cached features and record held-out performance.
The point is to turn "the information is in the representation" into a
spatial map and separate three things that whole-image probes conflate:

  1. answer      -- within one relation type (question fixed), can a single
                    token tell whether this image satisfies the relation?
                    If no cell does, the decision needs composition across
                    regions rather than local evidence.
  2. membership  -- does a token know it belongs to one of the two objects
                    named in the question, rather than to the rest of the
                    image? (local evidence is present)
  3. pointing    -- can a token be matched to the object *name* in the
                    question, and does the argmax land on that object's box?
                    (local evidence is addressable)

Model correctness is mapped the same way as a reference.

All probes are grouped 5-fold cross-validated by image id, so no image is
shared between train and test; every reported number is out-of-fold. Probes
are run per relation type, because training one probe across "left of" and
"above" with a diluted question vector mixes incompatible decision rules.
The name and membership probes use boxes at train time: diagnostics, not
deployable readers.

Outputs:
    results/diagnostics/per_token_probe.json
    results/figures/per_token_probe_dev.png
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT
from experiments.readout_experiment import REGION_PATHS, attach_regions, read_jsonl

GRID = 9
NCELL = GRID * GRID
STAGES = ["S1", "S6", "S12"]
RELS = ["to the left of", "to the right of", "above", "below"]


def load_payload(path):
    p = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    recs = {}
    for rp in REGION_PATHS:
        for r in read_jsonl(rp):
            recs[(r["pair_id"], "source")] = (r["source_target"], r["source_reference"])
            recs[(r["pair_id"], "target")] = (r["target_target"], r["target_reference"])
    attach_regions(p["examples"], recs)
    for e in p["examples"]:
        base = "source" if e["role"].startswith("source") else "target"
        rec = recs.get((e["pair_id"], base))
        if rec:
            e["name_target"] = rec[0]["name"].strip().lower()
            e["name_ref"] = rec[1]["name"].strip().lower()
    return p


def cell_mask(tokens):
    m = np.zeros(NCELL, dtype=np.float32)
    m[list(tokens)] = 1.0
    return m


def build(p):
    meta = {m["key"]: m for m in p["meta"]}
    rows = []
    for i, e in enumerate(p["examples"]):
        m = meta[e["key"]]
        rows.append({
            "image_id": str(m["image_id"]),
            "pair_id": e["pair_id"],
            "relation": e["group"].split("|")[1],
            "role": 1.0 if e["role"].startswith("source") else 0.0,
            "correct": 1.0 if float(p["margin"][i]) > 0 else 0.0,
            "target": cell_mask(e.get("region_target") or []),
            "ref": cell_mask(e.get("region_ref") or []),
            "name_target": e.get("name_target", ""),
            "name_ref": e.get("name_ref", ""),
            "position": np.array(e["position"], dtype=np.float32),
        })
    X = {s: p[s].float().numpy() for s in STAGES}
    return X, rows


def fit_pca(Xtr, n_comp, seed=0):
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    if flat.shape[0] > 20000:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.shape[0], 20000, replace=False)]
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    pca.fit(flat)
    return pca


def residualize(Xtr, Ptr, Xte, Pte):
    """Remove the linear component of each token's feature predicted by the
    12-dim object geometry, fitted on training examples only."""
    A = np.concatenate([Ptr, np.ones((len(Ptr), 1), dtype=np.float32)], axis=1)
    B = np.linalg.lstsq(A, Xtr.reshape(len(Xtr), -1), rcond=None)[0]
    out_tr = (Xtr - (A @ B).reshape(Xtr.shape)).astype(np.float32)
    Ate = np.concatenate([Pte, np.ones((len(Pte), 1), dtype=np.float32)], axis=1)
    out_te = (Xte - (Ate @ B).reshape(Xte.shape)).astype(np.float32)
    return out_tr, out_te


def reduce_fold(X, tr, te, n_comp, P=None, resid=False, seed=0):
    """One PCA per stage for this fold, reused by every probe."""
    out = {}
    for s in STAGES:
        Xtr, Xte = X[s][tr], X[s][te]
        if resid:
            Xtr, Xte = residualize(Xtr, P[tr], Xte, P[te])
        pca = fit_pca(Xtr, n_comp, seed=seed)
        Ztr = pca.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(len(tr), NCELL, -1)
        Zte = pca.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(len(te), NCELL, -1)
        out[s] = (Ztr.astype(np.float32), Zte.astype(np.float32))
    return out


def cell_probe(Ztr, ytr, Zte, cell_ok=None, y2=None, seed=0):
    """One binary probe per cell; returns (n_te, 81) scores. If cell_ok is
    given, only rows where it is true are used and y2 supplies per-row labels."""
    scores = np.full((len(Zte), NCELL), np.nan, dtype=np.float32)
    for c in range(NCELL):
        if cell_ok is None:
            if len(np.unique(ytr)) < 2:
                continue
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            clf.fit(Ztr[:, c], ytr)
        else:
            keep = cell_ok[:, c] > 0
            if keep.sum() < 20 or len(np.unique(y2[keep, c])) < 2:
                continue
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            clf.fit(Ztr[keep, c], y2[keep, c])
        scores[:, c] = clf.decision_function(Zte[:, c])
    return scores


def pooled_probe(Ztr, ytr, Zte, seed=0):
    Xtr = Ztr.mean(axis=1)
    Xte = Zte.mean(axis=1)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.decision_function(Xte)


def map_metrics(scores, y):
    acc = np.full(NCELL, np.nan)
    auc = np.full(NCELL, np.nan)
    for c in range(NCELL):
        s = scores[:, c]
        ok = ~np.isnan(s)
        if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
            continue
        acc[c] = ((s[ok] > 0).astype(np.float32) == y[ok]).mean()
        auc[c] = roc_auc_score(y[ok], s[ok])
    return acc.reshape(GRID, GRID), auc.reshape(GRID, GRID)


def membership_stats(scores, rows):
    """AUC and argmax hit rate for the 'token is one of the two objects' probe."""
    on = np.clip(np.stack([r["target"] + r["ref"] for r in rows]), 0, 1)
    aucs, hits, chance, dist, hit_groups = [], [], [], [], []
    cell_auc = np.full(NCELL, np.nan)
    for c in range(NCELL):
        sc, yy = scores[:, c], on[:, c]
        ok = ~np.isnan(sc)
        if ok.sum() < 30 or len(np.unique(yy[ok])) < 2:
            continue
        cell_auc[c] = roc_auc_score(yy[ok], sc[ok])
    for i in range(len(rows)):
        s, yy = scores[i], on[i]
        ok = ~np.isnan(s)
        if ok.sum() < NCELL - 5 or len(np.unique(yy[ok])) < 2:
            continue
        aucs.append(roc_auc_score(yy[ok], s[ok]))
        if yy.sum() > 0:
            order = np.argsort(-np.where(ok, s, -np.inf))
            cells = np.flatnonzero(yy > 0)
            hits.append(float(order[0] in cells))
            hit_groups.append(rows[i]["image_id"])
            chance.append(float(yy.sum() / NCELL))
            centers = np.divmod(cells, GRID)
            r0, c0 = divmod(order[0], GRID)
            dist.append(min(abs(r0 - a) + abs(c0 - b) for a, b in zip(*centers)))
    return {
        "n": len(aucs),
        "auc": float(np.mean(aucs)) if aucs else None,
        "top1_hit": float(np.mean(hits)) if hits else None,
        "chance": float(np.mean(chance)) if chance else None,
        "mean_l1_distance": float(np.mean(dist)) if dist else None,
        "auc_map": np.round(cell_auc.reshape(GRID, GRID), 3).tolist(),
        "auc_peak_cell": float(np.nanmax(cell_auc)),
    }, hits, hit_groups


def name_pointing_fold(Ztr, tr_rows, Zte, te_rows, name_to_id, seed=0):
    """Train one multi-class name probe on object cells, score every cell."""
    Xr, yr = [], []
    for j, r in enumerate(tr_rows):
        overlap = (r["target"] > 0) & (r["ref"] > 0)
        for cells, nm in ((np.flatnonzero(r["target"] > 0), r["name_target"]),
                          (np.flatnonzero(r["ref"] > 0), r["name_ref"])):
            if nm not in name_to_id:
                continue
            for c in cells:
                if overlap[c]:
                    continue
                Xr.append(Ztr[j, c])
                yr.append(name_to_id[nm])
    if not Xr:
        return None
    Xr, yr = np.stack(Xr), np.array(yr)
    if len(Xr) > 40000:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(Xr), 40000, replace=False)
        Xr, yr = Xr[sel], yr[sel]
    clf = LogisticRegression(max_iter=300, tol=1e-3, random_state=seed)
    clf.fit(Xr, yr)
    proba = clf.predict_proba(Zte.reshape(-1, Zte.shape[-1])).reshape(len(te_rows), NCELL, -1)
    out_t = np.full((len(te_rows), NCELL), np.nan, dtype=np.float32)
    out_r = np.full((len(te_rows), NCELL), np.nan, dtype=np.float32)
    for j, r in enumerate(te_rows):
        if r["name_target"] in name_to_id:
            out_t[j] = proba[j, :, name_to_id[r["name_target"]]]
        if r["name_ref"] in name_to_id:
            out_r[j] = proba[j, :, name_to_id[r["name_ref"]]]
    return out_t, out_r


def bootstrap_ci(values, groups, n=1000, seed=0):
    vals = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(idx):
            out.append(float(vals[idx].mean()))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None


def occupancy(rows):
    tgt = np.mean([r["target"] for r in rows], axis=0).reshape(GRID, GRID)
    ref = np.mean([r["ref"] for r in rows], axis=0).reshape(GRID, GRID)
    return tgt, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/readout/features_dev_strict.pt")
    ap.add_argument("--out", default="results/diagnostics/per_token_probe.json")
    ap.add_argument("--fig", default="results/figures/per_token_probe_dev.png")
    ap.add_argument("--n-comp", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    p = load_payload(args.features)
    X, rows = build(p)
    n = len(rows)
    groups = np.array([r["image_id"] for r in rows])
    folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(n), groups=groups))
    print(f"{n} examples, {len(set(groups))} images, {args.folds} folds")

    names = [f"answer_{r}" for r in RELS] + ["correct", "membership", "membership_resid"]
    pred = {nm: {s: np.full((n, NCELL), np.nan, dtype=np.float32) for s in STAGES} for nm in names}
    pooled = {nm: {s: np.full(n, np.nan, dtype=np.float32) for s in STAGES} for nm in names}
    pt = {s: {"target": np.full((n, NCELL), np.nan, dtype=np.float32),
              "ref": np.full((n, NCELL), np.nan, dtype=np.float32)} for s in STAGES}

    P = np.stack([r["position"] for r in rows])
    y_correct = np.array([r["correct"] for r in rows], dtype=np.float32)
    on = np.clip(np.stack([r["target"] + r["ref"] for r in rows]), 0, 1)

    counts = collections.Counter()
    for r in rows:
        counts[r["name_target"]] += 1
        counts[r["name_ref"]] += 1
    label_names = [nm for nm, _ in counts.most_common(args.topk) if nm]
    name_to_id = {nm: i for i, nm in enumerate(label_names)}
    coverage = sum(counts[nm] for nm in label_names) / sum(counts.values())

    for k, (tr, te) in enumerate(folds):
        Z = reduce_fold(X, tr, te, args.n_comp)
        Zr = reduce_fold(X, tr, te, args.n_comp, P=P, resid=True)
        # answer probes, one relation type at a time (question fixed within type)
        for rel in RELS:
            mtr = np.array([rows[i]["relation"] == rel for i in tr])
            mte = np.array([rows[i]["relation"] == rel for i in te])
            if mtr.sum() < 60 or mte.sum() < 20:
                continue
            ytr = np.array([rows[i]["role"] for i in tr])[mtr]
            for s in STAGES:
                pred[f"answer_{rel}"][s][te[mte]] = cell_probe(Z[s][0][mtr], ytr, Z[s][1][mte])
                pooled[f"answer_{rel}"][s][te[mte]] = pooled_probe(Z[s][0][mtr], ytr, Z[s][1][mte])
        # correctness
        for s in STAGES:
            pred["correct"][s][te] = cell_probe(Z[s][0], y_correct[tr], Z[s][1])
            pooled["correct"][s][te] = pooled_probe(Z[s][0], y_correct[tr], Z[s][1])
        # membership (token is one of the two queried objects)
        for tag, src in (("membership", Z), ("membership_resid", Zr)):
            for s in STAGES:
                pred[tag][s][te] = cell_probe(src[s][0], None, src[s][1],
                                              cell_ok=np.ones_like(on[tr]), y2=on[tr])
        # name pointing
        for s in STAGES:
            got = name_pointing_fold(Z[s][0], [rows[i] for i in tr], Z[s][1],
                                     [rows[i] for i in te], name_to_id)
            if got is not None:
                pt[s]["target"][te], pt[s]["ref"][te] = got
        print(f"  fold {k + 1}/{len(folds)} done")

    results = {
        "features": args.features,
        "n_examples": n,
        "n_images": len(set(groups)),
        "cv": f"GroupKFold({args.folds}) by image_id, all metrics out-of-fold",
        "n_components": args.n_comp,
        "note": "membership and name probes use boxes at train time; diagnostics only",
        "counts": {rel: int(sum(r["relation"] == rel for r in rows)) for rel in RELS},
    }
    tgt_occ, ref_occ = occupancy(rows)
    results["occupancy"] = {"target": tgt_occ.round(3).tolist(), "ref": ref_occ.round(3).tolist()}

    for nm in names:
        if nm.startswith("membership"):
            continue
        entry = {}
        for s in STAGES:
            if nm.startswith("answer_"):
                rel = nm[len("answer_"):]
                keep = np.array([r["relation"] == rel for r in rows])
            else:
                keep = np.ones(n, dtype=bool)
            sc = pred[nm][s][keep]
            y = (np.array([r["role"] for r in rows])[keep] if nm.startswith("answer_")
                 else y_correct[keep])
            acc_map, auc_map = map_metrics(sc, y)
            psc = pooled[nm][s][keep]
            pok = ~np.isnan(psc)
            entry[s] = {
                "n": int(keep.sum()),
                "acc_map": np.round(acc_map, 3).tolist(),
                "auc_map": np.round(auc_map, 3).tolist(),
                "acc_mean": float(np.nanmean(acc_map)),
                "acc_peak": float(np.nanmax(acc_map)),
                "auc_mean": float(np.nanmean(auc_map)),
                "pooled_acc": float(((psc[pok] > 0) == y[pok]).mean()) if pok.any() else None,
                "pooled_auc": float(roc_auc_score(y[pok], psc[pok])) if pok.sum() > 30 else None,
            }
            print(f"  {nm:22s} {s}: mean acc {entry[s]['acc_mean']:.3f} "
                  f"peak {entry[s]['acc_peak']:.3f} pooled {entry[s]['pooled_acc']:.3f}")
        results[nm] = entry

    for nm in ("membership", "membership_resid"):
        entry = {}
        for s in STAGES:
            st, hits, hg = membership_stats(pred[nm][s], rows)
            st["top1_ci"] = bootstrap_ci(hits, hg) if hits else None
            entry[s] = st
            print(f"  {nm:22s} {s}: auc {st['auc']:.3f} top1 {st['top1_hit']:.3f} "
                  f"chance {st['chance']:.3f} l1 {st['mean_l1_distance']:.2f}")
        results[nm] = entry

    results["name_pointing"] = {"names": label_names, "coverage": coverage}
    for s in STAGES:
        hits, both, chance, hit_groups = [], [], [], []
        for i, r in enumerate(rows):
            if r["name_target"] not in name_to_id or r["name_ref"] not in name_to_id:
                continue
            mt, mr = r["target"] > 0, r["ref"] > 0
            if mt.sum() == 0 or mr.sum() == 0:
                continue
            st, sr = pt[s]["target"][i], pt[s]["ref"][i]
            if np.isnan(st).any() or np.isnan(sr).any():
                continue
            ht = np.argmax(st) in np.flatnonzero(mt)
            hr = np.argmax(sr) in np.flatnonzero(mr)
            hits.append(0.5 * (float(ht) + float(hr)))
            both.append(float(ht and hr))
            chance.append(0.5 * (mt.sum() + mr.sum()) / NCELL)
            hit_groups.append(r["image_id"])
        results["name_pointing"][s] = {
            "n": len(hits),
            "top1_hit": float(np.mean(hits)) if hits else None,
            "both_hit": float(np.mean(both)) if both else None,
            "chance": float(np.mean(chance)) if chance else None,
            "top1_ci": bootstrap_ci(hits, hit_groups) if hits else None,
        }
        print(f"  name_pointing {s}: n {len(hits)} top1 {results['name_pointing'][s]['top1_hit']:.3f} "
              f"both {results['name_pointing'][s]['both_hit']:.3f} "
              f"chance {results['name_pointing'][s]['chance']:.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")
    make_figure(results, tgt_occ, ref_occ, ROOT / args.fig)


def make_figure(results, tgt_occ, ref_occ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("answer_to the left of", "acc_map", "answer left of"),
              ("answer_to the right of", "acc_map", "answer right of"),
              ("correct", "acc_map", "model correctness"),
              ("membership", "auc_map", "queried-object membership (AUC)")]
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 12.4))
    for r, (name, key, title) in enumerate(panels):
        for c, s in enumerate(STAGES):
            ax = axes[r, c]
            m = np.array(results[name][s][key])
            im = ax.imshow(m, cmap="viridis", vmin=np.nanmin(m), vmax=np.nanmax(m))
            ax.set_title(f"{title} {s}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Per-token probe maps on the 9x9 visual grid (grouped CV by image)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
