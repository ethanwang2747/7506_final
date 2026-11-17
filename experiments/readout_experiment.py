"""Train lightweight readers on cached SmolVLM2 features and evaluate them.

The experiment compares the frozen model's own yes/no margin with:
  - a position-only logistic baseline (the shortcut control),
  - a linear probe on the final prompt token,
  - mean-pooled visual features,
  - question-conditioned attention pooling over visual tokens.

Evaluation covers held-out source/target pairs and same-image horizontal flips.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import ROOT

STAGES = ["S1", "S6", "S12"]
REGION_PATHS = [
    "data/derived/spatial_pairs_dev_strict.jsonl",
    "data/derived/spatial_pairs_dev.jsonl",
    "data/derived/spatial_pairs_test_strict.jsonl",
    "data/derived/spatial_pairs_test.jsonl",
]


def read_jsonl(path):
    with open(ROOT / path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_regions():
    recs = {}
    for path in REGION_PATHS:
        for r in read_jsonl(path):
            recs[(r["pair_id"], "source")] = (r["source_target"], r["source_reference"])
            recs[(r["pair_id"], "target")] = (r["target_target"], r["target_reference"])
    return recs


def attach_regions(examples, recs):
    for e in examples:
        base = "source" if e["role"] in ("source", "source_flipped") else "target"
        pair = recs.get((e["pair_id"], base))
        target, reference = set(), set()
        if pair:
            target = set(pair[0]["tokens_9"])
            reference = set(pair[1]["tokens_9"])
        if e["is_flip"]:
            target = {(t // 9) * 9 + (8 - t % 9) for t in target}
            reference = {(t // 9) * 9 + (8 - t % 9) for t in reference}
        e["region_target"] = sorted(target)
        e["region_ref"] = sorted(reference)
        e["region"] = sorted(target | reference)


def load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["key_to_idx"] = {k: i for i, k in enumerate(payload["keys"])}
    return payload


class FeatureSet:
    def __init__(self, specs):
        # specs: list of (path, kind), kind in {"pairs", "flip"}
        self.payloads = []
        self.examples = []
        seen = set()
        for pi, (path, kind) in enumerate(specs):
            payload = load_payload(path)
            self.payloads.append(payload)
            for e in payload["examples"]:
                if kind == "flip_train" and e["is_flip"] == 0:
                    continue
                sig = (e["pair_id"], e["role"], e["is_flip"])
                if sig in seen:
                    continue
                seen.add(sig)
                e = dict(e)
                e["payload_idx"] = pi
                e["key_idx"] = payload["key_to_idx"][e["key"]]
                e["kind"] = kind
                self.examples.append(e)

    def subset(self, groups=None, exclude_groups=None):
        out = self.examples
        if groups:
            out = [e for e in out if e["group"] in set(groups)]
        if exclude_groups:
            out = [e for e in out if e["group"] not in set(exclude_groups)]
        return out


class FeatureDataset(Dataset):
    def __init__(self, payloads, examples, stages=None, q_key=None):
        self.payloads = payloads
        self.examples = examples
        self.stages = stages or STAGES
        self.q_key = q_key or "E_mean"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        p = self.payloads[e["payload_idx"]]
        k = e["key_idx"]
        v = torch.stack([p[s][k] for s in self.stages])
        q = p[self.q_key][k]
        region = set(e.get("region", []))
        return {
            "v": v,
            "q": q,
            "pos": torch.tensor(e["position"], dtype=torch.float32),
            "y": torch.tensor(float(e["label"])),
            "pair_id": e["pair_id"],
            "group": e["group"],
            "kind": e["kind"],
            "region": torch.tensor([i in region for i in range(81)], dtype=torch.bool),
        }


def collate(batch):
    return {
        "v": torch.stack([b["v"] for b in batch]),
        "q": torch.stack([b["q"] for b in batch]),
        "pos": torch.stack([b["pos"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "pair_id": [b["pair_id"] for b in batch],
        "group": [b["group"] for b in batch],
        "kind": [b["kind"] for b in batch],
        "region": torch.stack([b["region"] for b in batch]),
    }


class MeanPoolReader(nn.Module):
    def __init__(self, dims, q_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.v_norm = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.v_proj = nn.ModuleList([nn.Linear(d, hidden) for d in dims])
        self.q_norm = nn.LayerNorm(q_dim)
        self.q_proj = nn.Linear(q_dim, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, v, q):
        h = 0
        for i in range(v.shape[1]):
            x = self.v_norm[i](v[:, i])
            h = h + self.v_proj[i](x.mean(dim=1))
        h = h / v.shape[1] + self.q_proj(self.q_norm(q))
        return self.head(h).squeeze(-1)


class AttnPoolReader(nn.Module):
    def __init__(self, dims, q_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.v_norm = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.q_norm = nn.LayerNorm(q_dim)
        self.q_proj = nn.Linear(q_dim, hidden)
        self.k_proj = nn.ModuleList([nn.Linear(d, hidden) for d in dims])
        self.v_proj = nn.ModuleList([nn.Linear(d, hidden) for d in dims])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.scale = math.sqrt(hidden)

    def forward(self, v, q, return_attn=False):
        qh = self.q_proj(self.q_norm(q))
        pooled, attns = [], []
        for i in range(v.shape[1]):
            x = self.v_norm[i](v[:, i])
            k = self.k_proj[i](x)
            val = self.v_proj[i](x)
            scores = (k * qh.unsqueeze(1)).sum(-1) / self.scale
            attn = torch.softmax(scores, dim=-1)
            attns.append(attn)
            pooled.append((attn.unsqueeze(-1) * val).sum(dim=1))
        h = torch.stack(pooled, dim=1).mean(dim=1) + qh
        logits = self.head(h).squeeze(-1)
        return (logits, attns) if return_attn else logits


class MLPReader(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, opt, device, args):
    model.train()
    total = 0.0
    for b in loader:
        v, q, y = b["v"].to(device).float(), b["q"].to(device).float(), b["y"].to(device)
        if isinstance(model, MLPReader):
            logits = model(q)
        elif args.aux_region and isinstance(model, AttnPoolReader):
            logits, attns = model(v, q, return_attn=True)
            mask = b["region"].to(device).float()
            valid = mask.sum(dim=1) > 0
            aux = 0.0
            if valid.any():
                for attn in attns:
                    mass = (attn * mask).sum(dim=1)
                    aux = aux - mass[valid].clamp_min(1e-6).log().mean()
                aux = aux / max(len(attns), 1)
        else:
            logits = model(v, q)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if args.aux_region and isinstance(model, AttnPoolReader):
            loss = loss + args.aux_weight * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, labels, pair_ids, groups = [], [], [], []
    for b in loader:
        v, q = b["v"].to(device).float(), b["q"].to(device).float()
        if isinstance(model, MLPReader):
            logits = model(q)
        else:
            logits = model(v, q)
        scores.extend(logits.float().cpu().tolist())
        labels.extend(b["y"].tolist())
        pair_ids.extend(b["pair_id"])
        groups.extend(b["group"])
    return np.array(scores), np.array(labels), pair_ids, groups


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def pair_accuracy(labels, scores, pair_ids):
    d = collections.defaultdict(list)
    for y, s, p in zip(labels, scores, pair_ids):
        d[p].append((float(y), float(s)))
    ok = []
    for vals in d.values():
        src = [s for y, s in vals if y == 1]
        tgt = [s for y, s in vals if y == 0]
        if src and tgt:
            ok.append(src[0] > tgt[0])
    return float(np.mean(ok)) if ok else float("nan")


def bootstrap_auc(labels, scores, pair_ids, n=1000, seed=0):
    pairs = sorted(set(pair_ids))
    by_pair = collections.defaultdict(list)
    for i, p in enumerate(pair_ids):
        by_pair[p].append(i)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(pairs, size=len(pairs), replace=True)
        idx = [i for p in sample for i in by_pair[p]]
        y = labels[idx]
        if len(set(y.tolist())) < 2:
            continue
        vals.append(roc_auc_score(y, scores[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def metrics(labels, scores, pair_ids, groups, bootstrap_n=500, seed=0):
    out = {
        "n_examples": int(len(labels)),
        "n_pairs": int(len(set(pair_ids))),
        "auc": safe_auc(labels, scores),
        "auc_ci95": bootstrap_auc(labels, scores, pair_ids, n=bootstrap_n, seed=seed),
        "pair_acc": pair_accuracy(labels, scores, pair_ids),
    }
    per_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        y, s = labels[idx], scores[idx]
        pid = [pair_ids[i] for i in idx]
        per_group[g] = {
            "n": len(idx),
            "auc": safe_auc(y, s),
            "auc_ci95": bootstrap_auc(y, s, pid, n=bootstrap_n, seed=seed) if len(set(y.tolist())) > 1 else None,
            "pair_acc": pair_accuracy(y, s, pid),
        }
    out["per_group"] = per_group
    return out


def group_split(examples, seed, val_frac=0.2):
    groups = sorted(set(e["pair_id"] for e in examples))
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_g, va_g = next(splitter.split(groups, groups=groups))
    tr = set(groups[i] for i in tr_g)
    va = set(groups[i] for i in va_g)
    return [e for e in examples if e["pair_id"] in tr], [e for e in examples if e["pair_id"] in va]


def fit_neural(model, train_ex, val_ex, args, seed, device, stages, q_key):
    torch.manual_seed(seed)
    train_ds = FeatureDataset(train_ex[0], train_ex[1], stages=stages, q_key=q_key)
    val_ds = FeatureDataset(val_ex[0], val_ex[1], stages=stages, q_key=q_key)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, device, args)
        s, y, _, _ = predict(model, val_loader, device)
        auc = safe_auc(y, s)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_auc, loss


def fit_fixed(model, payloads, examples, args, epochs, seed, device, stages, q_key):
    torch.manual_seed(seed)
    ds = FeatureDataset(payloads, examples, stages=stages, q_key=q_key)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(max(1, epochs)):
        train_epoch(model, loader, opt, device, args)
    return model


def eval_original(payloads, examples, bootstrap_n):
    scores = np.array([payloads[e["payload_idx"]]["margin"][e["key_idx"]].item() for e in examples])
    labels = np.array([e["label"] for e in examples], dtype=int)
    pairs = [e["pair_id"] for e in examples]
    groups = [e["group"] for e in examples]
    return metrics(labels, scores, pairs, groups, bootstrap_n=bootstrap_n)


def eval_position(train_examples, test_examples, payloads, bootstrap_n):
    Xtr = np.array([e["position"] for e in train_examples])
    ytr = np.array([e["label"] for e in train_examples])
    Xte = np.array([e["position"] for e in test_examples])
    yte = np.array([e["label"] for e in test_examples])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    scores = clf.predict_proba(sc.transform(Xte))[:, 1]
    return metrics(yte, scores, [e["pair_id"] for e in test_examples],
                   [e["group"] for e in test_examples], bootstrap_n=bootstrap_n)


def eval_linear_token(train_examples, train_payloads, test_examples, test_payloads, q_key, bootstrap_n):
    Xtr = np.array([train_payloads[e["payload_idx"]][q_key][e["key_idx"]].float().numpy() for e in train_examples])
    ytr = np.array([e["label"] for e in train_examples])
    Xte = np.array([test_payloads[e["payload_idx"]][q_key][e["key_idx"]].float().numpy() for e in test_examples])
    yte = np.array([e["label"] for e in test_examples])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    scores = clf.predict_proba(sc.transform(Xte))[:, 1]
    return metrics(yte, scores, [e["pair_id"] for e in test_examples],
                   [e["group"] for e in test_examples], bootstrap_n=bootstrap_n)


def train_models(model_cls, train_examples, train_payloads, args, device, stages, q_key):
    models, val_aucs, epochs = [], [], []
    for seed in range(args.seeds):
        tr, va = group_split(train_examples, seed)
        model = model_cls().to(device)
        model, best_epoch, best_auc, _ = fit_neural(
            model, (train_payloads, tr), (train_payloads, va), args, seed, device, stages, q_key,
        )
        models.append(model)
        val_aucs.append(best_auc)
        epochs.append(best_epoch + 1)
    return models, val_aucs, epochs


def eval_models(models, test_examples, test_payloads, bootstrap_n, stages, q_key):
    all_scores = []
    for model in models:
        ds = FeatureDataset(test_payloads, test_examples, stages=stages, q_key=q_key)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
        s, y, pid, g = predict(model, loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        all_scores.append(s)
    scores = np.mean(all_scores, axis=0)
    labels = np.array([e["label"] for e in test_examples], dtype=int)
    pairs = [e["pair_id"] for e in test_examples]
    groups = [e["group"] for e in test_examples]
    return metrics(labels, scores, pairs, groups, bootstrap_n=bootstrap_n, seed=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pairs", nargs="+", required=True)
    ap.add_argument("--train-flip", nargs="*", default=[])
    ap.add_argument("--test-pairs", nargs="+", required=True)
    ap.add_argument("--test-flip", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--exclude-train-groups", default=None,
                    help="comma-separated groups to exclude from training (unseen-group test)")
    ap.add_argument("--random-label", action="store_true",
                    help="shuffle training labels as a negative control")
    ap.add_argument("--aux-region", action="store_true",
                    help="supervise attention with training-time region tokens")
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--stages", default="S1,S6,S12")
    ap.add_argument("--q-key", default="E_mean",
                    choices=["E_mean", "E_last", "T6", "T12", "H6", "H12"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_specs = [(p, "pairs") for p in args.train_pairs] + [(p, "flip_train") for p in args.train_flip]
    test_pair_specs = [(p, "pairs") for p in args.test_pairs]
    test_flip_specs = [(p, "flip_test") for p in args.test_flip]

    train = FeatureSet(train_specs)
    test_pairs = FeatureSet(test_pair_specs) if args.test_pairs else FeatureSet([])
    test_flip = FeatureSet(test_flip_specs) if args.test_flip else FeatureSet([])

    exclude = set(args.exclude_train_groups.split(",")) if args.exclude_train_groups else None
    train_examples = train.subset(exclude_groups=exclude)
    if args.random_label:
        rng = np.random.default_rng(123)
        labels = [e["label"] for e in train_examples]
        perm = rng.permutation(len(labels))
        train_examples = [dict(e, label=labels[i]) for e, i in zip(train_examples, perm)]
    regions = load_regions()
    attach_regions(train_examples, regions)
    attach_regions(test_pairs.examples, regions)
    attach_regions(test_flip.examples, regions)
    train_pairs_only = [e for e in train_examples if e["is_flip"] == 0]
    attach_regions(train_pairs_only, regions)
    print(f"train examples={len(train_examples)} pairs={len(set(e['pair_id'] for e in train_examples))}")
    print(f"test pairs examples={len(test_pairs.examples)} test flip examples={len(test_flip.examples)}")

    stages = args.stages.split(",")
    if len(stages) > 1 and "S0_mean" in stages:
        raise SystemExit("S0_mean can only be used as a single stage")
    dim_map = {"S0_mean": 1152, "S1": 2048, "S6": 2048, "S12": 2048}
    dims = [dim_map[s] for s in stages]
    model_specs = {
        "meanpool": lambda: MeanPoolReader(dims, 2048, hidden=args.hidden, dropout=args.dropout),
        "attention": lambda: AttnPoolReader(dims, 2048, hidden=args.hidden, dropout=args.dropout),
    }

    results = {"config": vars(args), "train": {"n_examples": len(train_examples),
                                               "n_pairs": len(set(e["pair_id"] for e in train_examples))},
               "test_pairs": {}, "test_flip": {}}

    if test_pairs.examples:
        results["test_pairs"]["original"] = eval_original(
            test_pairs.payloads, test_pairs.examples, args.bootstrap)
        results["test_pairs"]["position"] = eval_position(
            train_pairs_only, test_pairs.examples, test_pairs.payloads, args.bootstrap)
        results["test_pairs"]["final_token_h12"] = eval_linear_token(
            train_pairs_only, train.payloads, test_pairs.examples, test_pairs.payloads,
            "H12", args.bootstrap)
        results["test_pairs"]["question_only_linear"] = eval_linear_token(
            train_pairs_only, train.payloads, test_pairs.examples, test_pairs.payloads,
            "E_mean", args.bootstrap)

    if test_flip.examples:
        results["test_flip"]["original"] = eval_original(
            test_flip.payloads, test_flip.examples, args.bootstrap)
        results["test_flip"]["position"] = eval_position(
            train_pairs_only, test_flip.examples, test_flip.payloads, args.bootstrap)
        results["test_flip"]["final_token_h12"] = eval_linear_token(
            train_pairs_only, train.payloads, test_flip.examples, test_flip.payloads,
            "H12", args.bootstrap)
        results["test_flip"]["question_only_linear"] = eval_linear_token(
            train_pairs_only, train.payloads, test_flip.examples, test_flip.payloads,
            "E_mean", args.bootstrap)

    for name, fn in model_specs.items():
        print(f"training {name}...")
        models, val_aucs, epochs = train_models(
            fn, train_examples, train.payloads, args, device, stages, args.q_key)
        stats = {
            "name": name,
            "seed_val_auc_mean": float(np.mean(val_aucs)),
            "seed_val_auc_std": float(np.std(val_aucs)),
            "epochs": epochs,
        }
        if test_pairs.examples:
            r = eval_models(models, test_pairs.examples, test_pairs.payloads,
                            args.bootstrap, stages, args.q_key)
            r.update(stats)
            results["test_pairs"][name] = r
        if test_flip.examples:
            r = eval_models(models, test_flip.examples, test_flip.payloads,
                            args.bootstrap, stages, args.q_key)
            r.update(stats)
            results["test_flip"][name] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    for split in ["test_pairs", "test_flip"]:
        if not results[split]:
            continue
        print(f"== {split}")
        for name, r in results[split].items():
            ci = r.get("auc_ci95")
            print(f"  {name:18s} AUC={r['auc']:.3f} CI{ci} pair_acc={r['pair_acc']:.3f}")


if __name__ == "__main__":
    main()
