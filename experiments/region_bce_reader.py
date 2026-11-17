"""Token-level region scorer with bbox-mask supervision (no boxes at test)."""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT
from experiments.readout_experiment import (FeatureDataset, FeatureSet, collate,
                                            group_split, load_regions, attach_regions,
                                            metrics, STAGES)

NUM_TOKENS = 81


class RegionBCEReader(nn.Module):
    def __init__(self, dims, q_dim, hidden=128, dropout=0.4):
        super().__init__()
        self.v_norm = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.q_norm = nn.LayerNorm(q_dim)
        self.q_proj = nn.Linear(q_dim, hidden)
        self.v_proj = nn.ModuleList([nn.Linear(d, hidden) for d in dims])
        self.scorer = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in dims
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * (len(dims) + 1)),
            nn.Linear(hidden * (len(dims) + 1), hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, v, q, return_aux=False):
        qh = self.q_proj(self.q_norm(q))
        pooled, logits_list, attns = [], [], []
        for i in range(v.shape[1]):
            x = self.v_proj[i](self.v_norm[i](v[:, i]))
            xq = x * qh.unsqueeze(1)
            qe = qh.unsqueeze(1).expand(-1, x.shape[1], -1)
            s = self.scorer[i](torch.cat([x, xq, qe], dim=-1)).squeeze(-1)
            a = torch.softmax(s, dim=-1)
            pooled.append((a.unsqueeze(-1) * x).sum(dim=1))
            logits_list.append(s)
            attns.append(a)
        h = torch.cat(pooled + [qh], dim=-1)
        out = self.head(h).squeeze(-1)
        return (out, logits_list, attns) if return_aux else out


def region_loss(logits_list, attns, masks, pos_weight=3.0):
    loss = 0.0
    for s, a in zip(logits_list, attns):
        loss = loss + F.binary_cross_entropy_with_logits(
            s, masks.float(), pos_weight=torch.tensor(pos_weight, device=s.device))
        mass = (a * masks.float()).sum(dim=1).clamp_min(1e-6)
        loss = loss - mass.log().mean()
    return loss / len(logits_list)


def predict(model, loader, device):
    model.eval()
    scores, labels, pairs, groups = [], [], [], []
    with torch.no_grad():
        for b in loader:
            v, q = b["v"].to(device).float(), b["q"].to(device).float()
            s = model(v, q).float().cpu().numpy()
            scores.extend(s.tolist()); labels.extend(b["y"].tolist())
            pairs.extend(b["pair_id"]); groups.extend(b["group"])
    return np.array(scores), np.array(labels), pairs, groups


def fit(model, train_examples, val_examples, payloads, args, seed, device):
    torch.manual_seed(seed)
    tr = FeatureDataset(payloads, train_examples, stages=STAGES, q_key="E_mean")
    va = FeatureDataset(payloads, val_examples, stages=STAGES, q_key="E_mean")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = (-1, None, 0)
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            v, q, y = b["v"].to(device).float(), b["q"].to(device).float(), b["y"].to(device)
            mask = b["region"].to(device)
            out, logits_list, attns = model(v, q, return_aux=True)
            loss = F.binary_cross_entropy_with_logits(out, y)
            loss = loss + args.mask_weight * region_loss(logits_list, attns, mask)
            opt.zero_grad(); loss.backward(); opt.step()
        s, yv, pid, _ = predict(model, vl, device)
        auc = roc_auc_score(yv, s) if len(set(yv.tolist())) > 1 else 0.5
        if auc > best[0]:
            best = (auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= args.patience:
            break
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pairs", nargs="+", default=[
        "results/readout/features_dev_strict.pt", "results/readout/features_dev_core.pt"])
    ap.add_argument("--test-pairs", default="results/readout/features_test_core.pt")
    ap.add_argument("--test-flip", default="results/readout/features_test_core_flip.pt")
    ap.add_argument("--out", default="results/region_reader/region_bce.json")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mask-weight", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = FeatureSet([(p, "pairs") for p in args.train_pairs])
    regions = load_regions()
    train_examples = train.subset()
    attach_regions(train_examples, regions)
    test = FeatureSet([(args.test_pairs, "pairs")])
    attach_regions(test.examples, regions)
    flip = FeatureSet([(args.test_flip, "flip_test")])
    attach_regions(flip.examples, regions)

    print(f"train {len(train_examples)} pairs={len(set(e['pair_id'] for e in train_examples))}")
    models = []
    for seed in range(args.seeds):
        tr, va = group_split(train_examples, seed)
        model = RegionBCEReader([2048] * len(STAGES), 2048, hidden=args.hidden).to(device)
        model, auc, ep = fit(model, tr, va, train.payloads, args, seed, device)
        models.append(model)
        print(f"seed {seed} val_auc={auc:.3f} best_ep={ep}")

    results = {"config": vars(args), "test_pairs": {}, "test_flip": {}}
    for name, ds in [("test_pairs", test), ("test_flip", flip)]:
        all_scores = []
        for model in models:
            loader = DataLoader(FeatureDataset(ds.payloads, ds.examples), batch_size=64,
                                shuffle=False, collate_fn=collate)
            s, y, pid, g = predict(model, loader, device)
            all_scores.append(s)
        scores = np.mean(all_scores, axis=0)
        labels = np.array([e["label"] for e in ds.examples])
        pairs = [e["pair_id"] for e in ds.examples]
        groups = [e["group"] for e in ds.examples]
        results[name] = metrics(labels, scores, pairs, groups, bootstrap_n=args.bootstrap)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"saved {path}")
    for name in ("test_pairs", "test_flip"):
        r = results[name]
        print(f"{name:12s} AUC={r['auc']:.3f} CI{r['auc_ci95']} pair_acc={r['pair_acc']:.3f}")


if __name__ == "__main__":
    main()
