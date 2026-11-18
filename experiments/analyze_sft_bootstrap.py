"""Paired bootstrap for the SFT relation deltas (by pair id)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT

import glob
TAGS = ["base"] + sorted(
    Path(f).stem.replace("eval_relation_", "")
    for f in glob.glob(str(ROOT / "results/sft/eval_relation_*.json"))
    if "base" not in Path(f).stem and "confirm" not in Path(f).stem)


def main():
    data = {t: json.loads((ROOT / f"results/sft/eval_relation_{t}.json").read_text()) for t in TAGS}
    y = np.array([m["label"] for m in data["base"]["meta"]])
    pairs = np.array([m["pair_id"] for m in data["base"]["meta"]])
    uniq = sorted(set(pairs))
    rng = np.random.default_rng(0)
    out = {}
    for t in TAGS[1:]:
        m = np.array(data[t]["margins"])
        b = np.array(data["base"]["margins"])
        draws = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(pairs == p) for p in pick])
            draws.append((roc_auc_score(y[idx], m[idx]) - roc_auc_score(y[idx], b[idx]),
                          ((m[idx] > 0).astype(int) == y[idx]).mean()
                          - ((b[idx] > 0).astype(int) == y[idx]).mean()))
        d = np.array(draws)
        out[t] = {
            "d_auc": float(roc_auc_score(y, m) - roc_auc_score(y, b)),
            "d_auc_ci": [float(np.percentile(d[:, 0], 2.5)), float(np.percentile(d[:, 0], 97.5))],
            "d_acc": float(((m > 0).astype(int) == y).mean() - ((b > 0).astype(int) == y).mean()),
            "d_acc_ci": [float(np.percentile(d[:, 1], 2.5)), float(np.percentile(d[:, 1], 97.5))],
        }
        print(t, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v])
                  for k, v in out[t].items()})
    dst = ROOT / "results/sft/bootstrap_deltas.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
