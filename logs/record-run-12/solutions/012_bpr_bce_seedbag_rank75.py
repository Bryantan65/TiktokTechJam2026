"""Seed-bagged BPR/BCE FM blend using per-user rank fusion at 75/25.

This refines node 11 by changing only the rank-fusion blend weight from 70/30
to 75/25 (BPR/BCE). Member training code and cache names are unchanged from
007/008/011, so existing predictions are reused; if caches are absent the
script trains the six members standalone.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
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
        self.eval(); out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def train_bce_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    X_t = torch.from_numpy(Xtr.astype(np.int64)); y_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); losses = []
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = torch.from_numpy(perm[i:i + bs])
            xb = X_t[idx].to(device); yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  bce seed {seed} epoch {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def make_positive_index_pairs(y, users):
    y = np.asarray(y); users = np.asarray(users)
    order = np.argsort(users, kind='mergesort'); users_s = users[order]
    pos_indices, pos_gids, neg_by_gid = [], [], []
    start = 0; gid = 0; n = len(order)
    while start < n:
        end = start + 1
        while end < n and users_s[end] == users_s[start]: end += 1
        idx = order[start:end]; yy = y[idx]
        pos = idx[yy > 0.5]; neg = idx[yy <= 0.5]
        if len(pos) > 0 and len(neg) > 0:
            pos_indices.append(pos.astype(np.int64)); pos_gids.append(np.full(len(pos), gid, dtype=np.int32)); neg_by_gid.append(neg.astype(np.int64)); gid += 1
        start = end
    if not pos_indices:
        raise RuntimeError('No users with both positive and negative impressions')
    return np.concatenate(pos_indices), np.concatenate(pos_gids), neg_by_gid


def sample_negatives_for_batch(gids, neg_by_gid, rng):
    neg = np.empty(len(gids), dtype=np.int64)
    for g in np.unique(gids):
        m = (gids == g); pool = neg_by_gid[int(g)]
        neg[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
    return neg


def train_bpr_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5, repeats=2, bce_weight=0.10, device='cpu', verbose=False):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_base, pos_gid_base, neg_by_gid = make_positive_index_pairs(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); losses = []
        for _ in range(repeats):
            perm = rng.permutation(len(pos_base))
            for i in range(0, len(perm), bs):
                psel = perm[i:i + bs]
                pos_idx = pos_base[psel]
                neg_idx = sample_negatives_for_batch(pos_gid_base[psel], neg_by_gid, rng)
                xb_pos = Xtr_t[torch.from_numpy(pos_idx)].to(device)
                xb_neg = Xtr_t[torch.from_numpy(neg_idx)].to(device)
                xb = torch.cat([xb_pos, xb_neg], dim=0)
                opt.zero_grad(set_to_none=True)
                logits = model(xb); m = len(pos_idx)
                loss = F.softplus(-(logits[:m] - logits[m:])).mean()
                if bce_weight > 0:
                    labels = torch.cat([torch.ones(m, device=device), torch.zeros(m, device=device)])
                    loss = loss + bce_weight * F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  bpr seed {seed} epoch {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def percentile_rank_by_user(scores, users):
    scores = np.asarray(scores, dtype=np.float64); users = np.asarray(users)
    out = np.empty_like(scores, dtype=np.float64)
    order = np.argsort(users, kind='mergesort'); us = users[order]
    start = 0; n = len(order)
    while start < n:
        end = start + 1
        while end < n and us[end] == us[start]: end += 1
        idx = order[start:end]
        ord2 = idx[np.argsort(scores[idx], kind='mergesort')]
        m = len(ord2)
        if m <= 1:
            out[ord2] = 0.0
        else:
            out[ord2] = np.arange(m, dtype=np.float64) / (m - 1.0)
        start = end
    return out


def get_member_preds(name, train_fn, enc, dim, Xtar, seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache', f'007_{name}_seed{seed}_n{len(Xtar)}.npy')
    if os.path.isfile(cache_path):
        print(f'loading cached {name} seed {seed} predictions {cache_path}')
        return np.load(cache_path).astype(np.float64)
    print(f'training {name} seed {seed} member')
    model = train_fn(enc, dim, seed=seed, device=device, verbose=verbose)
    preds = model.predict(Xtar, device=device).astype(np.float64)
    np.save(cache_path, preds)
    return preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(0)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    enc, dim = encode(splits)
    Xtar, _, utar = enc[target]
    verbose = (a.out is None)

    blended = []
    for member_seed in (0, 1, 2):
        bpr = get_member_preds('bpr_anchor_v1', train_bpr_member, enc, dim, Xtar, member_seed, a.device, verbose)
        bce = get_member_preds('bce_v1', train_bce_member, enc, dim, Xtar, member_seed, a.device, verbose)
        blended.append(0.75 * percentile_rank_by_user(bpr, utar) + 0.25 * percentile_rank_by_user(bce, utar))
    scores = np.mean(blended, axis=0)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print('done')
