"""Blend FM/BPR logits with leakage-free historical preference target encodings.

CTR/ranking competition writeups for recommenders commonly add user-item and
session/item interaction features to a neural or GBDT ranker (e.g. Kaggle OTTO
20th place: https://www.kaggle.com/competitions/otto-recommender-system/writeups/kicchotto-20th-place-solution).
Here the target metric is within-user, so we add smoothed train-only historical
preferences for user-author, user-video and user-tab, then blend them at readable
weight with the two strongest FM losses.  No raw long_view CSV column is read.
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def train_bce(splits, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu'):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        model.train()
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device); yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc


def build_pair_sampler(y, users):
    pos_by_user = defaultdict(list); neg_by_user = defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0.5:
            pos_by_user[u].append(i)
        else:
            neg_by_user[u].append(i)
    pos_idx, neg_pools = [], []
    for u, ps in pos_by_user.items():
        ns = neg_by_user.get(u)
        if not ns:
            continue
        arr = np.asarray(ns, dtype=np.int64)
        for p in ps:
            pos_idx.append(p); neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def train_bpr(splits, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
              bce_weight=0.05, device='cpu'):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    pos_idx, neg_pools = build_pair_sampler(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('no within-user pairs')
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_pairs = max(len(ytr), len(pos_idx))
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        order = rng.integers(0, len(pos_idx), size=n_pairs, dtype=np.int64)
        model.train()
        for i in range(0, len(order), bs):
            which = order[i:i + bs]
            p_np = pos_idx[which]
            n_np = np.empty(len(which), dtype=np.int64)
            for j, w in enumerate(which):
                pool = neg_pools[int(w)]
                n_np[j] = pool[rng.integers(0, len(pool))]
            pair_np = np.concatenate([p_np, n_np])
            xb = Xtr_t[torch.from_numpy(pair_np)].to(device)
            logits = model(xb)
            sp, sn = logits[:len(p_np)], logits[len(p_np):]
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            if bce_weight > 0:
                yb = ytr_t[torch.from_numpy(pair_np)].to(device)
                loss = loss + bce_weight * bce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc


def add_count(d, key, y):
    v = d[key]; v[0] += float(y); v[1] += 1.0


def prob(pos, cnt, prior, alpha):
    return (pos + alpha * prior) / (cnt + alpha) if cnt > 0 else prior


def logit(p):
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def historical_scores(train_rows, target_rows):
    global_pos = sum(float(r[6]) for r in train_rows); global_cnt = float(len(train_rows))
    gp = global_pos / max(global_cnt, 1.0)
    user = defaultdict(lambda: [0.0, 0.0]); ua = defaultdict(lambda: [0.0, 0.0])
    uv = defaultdict(lambda: [0.0, 0.0]); ut = defaultdict(lambda: [0.0, 0.0])
    author = defaultdict(lambda: [0.0, 0.0]); video = defaultdict(lambda: [0.0, 0.0])
    tab = defaultdict(lambda: [0.0, 0.0]); at = defaultdict(lambda: [0.0, 0.0])
    for r in train_rows:
        y = float(r[6]); u = r[1]; v = r[2]; a = r[3]; t = r[4]
        add_count(user, u, y); add_count(ua, (u, a), y); add_count(uv, (u, v), y); add_count(ut, (u, t), y)
        add_count(author, a, y); add_count(video, v, y); add_count(tab, t, y); add_count(at, (a, t), y)
    out = np.zeros(len(target_rows), dtype=np.float32)
    g_log = logit(gp)
    for i, r in enumerate(target_rows):
        u = r[1]; v = r[2]; a = r[3]; t = r[4]
        upos, ucnt = user.get(u, (0.0, 0.0))
        up = prob(upos, ucnt, gp, 30.0)
        ulog = logit(up)
        s = 0.0
        p, c = ua.get((u, a), (0.0, 0.0)); s += 1.05 * (logit(prob(p, c, up, 8.0)) - ulog)
        p, c = uv.get((u, v), (0.0, 0.0)); s += 1.25 * (logit(prob(p, c, up, 5.0)) - ulog)
        p, c = ut.get((u, t), (0.0, 0.0)); s += 0.75 * (logit(prob(p, c, up, 15.0)) - ulog)
        p, c = author.get(a, (0.0, 0.0)); s += 0.35 * (logit(prob(p, c, gp, 50.0)) - g_log)
        p, c = video.get(v, (0.0, 0.0)); s += 0.20 * (logit(prob(p, c, gp, 20.0)) - g_log)
        p, c = tab.get(t, (0.0, 0.0)); s += 0.25 * (logit(prob(p, c, gp, 200.0)) - g_log)
        p, c = at.get((a, t), (0.0, 0.0)); s += 0.15 * (logit(prob(p, c, gp, 80.0)) - g_log)
        out[i] = s
    return out


def z_by_user(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(scores, dtype=np.float64)
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    for idxs in groups.values():
        vals = scores[idxs]
        sd = vals.std()
        if sd > 1e-8:
            out[idxs] = (vals - vals.mean()) / sd
        else:
            out[idxs] = 0.0
    return out


def cached_member(name, split_name, target, seed, train_fn, splits, device):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'008_{name}_{split_name}_seed{seed}.npy')
    if os.path.isfile(path):
        return np.load(path)
    model, enc = train_fn(splits, seed=seed, device=device)
    X, _, _ = enc[target]
    preds = model.predict(X, device=device).astype(np.float64)
    np.save(path, preds)
    return preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'; split_name = 'dev'
    else:
        splits = load(a.data_dir); target = a.split; split_name = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}; hist blend")

    t0 = time.time()
    bce_pred = cached_member('bce', split_name, target, a.seed, train_bce, splits, a.device)
    bpr_pred = cached_member('bpr05', split_name, target, a.seed, train_bpr, splits, a.device)
    hist_pred = historical_scores(splits['train'], splits[target]).astype(np.float64)
    _, _, users = encode(splits)[0][target]
    scores = (0.35 * z_by_user(bce_pred, users) +
              0.35 * z_by_user(bpr_pred, users) +
              0.30 * z_by_user(hist_pred, users))
    print(f"built blended predictions in {time.time() - t0:.1f}s")

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        X, y, users = encode(splits)[0][target]
        print(evaluate(users, y, scores))
