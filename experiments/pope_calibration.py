"""Held-out calibration / flip-consistency check for POPE."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT


def main():
    data = json.loads((ROOT / "results/pope/pope_300.json").read_text())
    rows = data["per_question"]
    for r in rows:
        r["y"] = 1 if r["label"] == "yes" else 0
        r["m"] = r["margin"]
        r["mf"] = r.get("margin_flip", 0.0)
        r["img"] = r["image"]
    images = sorted(set(r["img"] for r in rows))
    alphas = np.linspace(-1, 2, 13)
    out = []
    for seed in range(20):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
                      .split(images, groups=images))
        tr_imgs = set(images[i] for i in tr)
        ev_imgs = set(images[i] for i in te)
        train = [r for r in rows if r["img"] in tr_imgs]
        ev = [r for r in rows if r["img"] in ev_imgs]
        ytr = np.array([r["y"] for r in train])
        yev = np.array([r["y"] for r in ev])
        mtr = np.array([r["m"] for r in train])
        mftr = np.array([r["mf"] for r in train])
        mev = np.array([r["m"] for r in ev])
        mfev = np.array([r["mf"] for r in ev])
        best = (-1, None, None)
        for a in alphas:
            s_tr = mtr + a * mftr
            for t in np.unique(np.quantile(s_tr, np.linspace(0.05, 0.95, 37))):
                acc = accuracy_score(ytr, s_tr > t)
                if acc > best[0]:
                    best = (acc, a, t)
        _, a, t = best
        out.append({
            "baseline_acc": float(accuracy_score(yev, mev > 0)),
            "flip_combined_acc": float(accuracy_score(yev, (mev + mfev) > 0)),
            "calibrated_acc": float(accuracy_score(yev, (mev + a * mfev) > t)),
            "baseline_f1": float(f1_score(yev, mev > 0)),
            "flip_combined_f1": float(f1_score(yev, (mev + mfev) > 0)),
            "alpha": float(a), "threshold": float(t),
        })
    summary = {}
    for key in ("baseline_acc", "flip_combined_acc", "calibrated_acc",
                "baseline_f1", "flip_combined_f1"):
        vals = [x[key] for x in out]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
    summary["flip_wins"] = int(sum(x["flip_combined_acc"] > x["baseline_acc"] for x in out))
    summary["n_splits"] = len(out)
    path = ROOT / "results/pope/pope_calibration.json"
    path.write_text(json.dumps({"summary": summary, "splits": out}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
