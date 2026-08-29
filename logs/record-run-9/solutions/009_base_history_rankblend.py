"""Blend node-6 base BPR/BCE ensemble with node-8 history-feature ensemble.

Both components are strong but not identical.  This keeps the new history signal
readable by giving it half the vote, while loading node-6/node-8 member caches
when available.  If caches are absent, all members are trained from scratch.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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


def pos_bucket(c):
    if c <= 0: return 0
    if c == 1: return 1
    if c == 2: return 2
    if c <= 5: return 3
    return 4


def neg_bucket(c):
    if c <= 0: return 0
    if c <= 2: return 1
    if c <= 5: return 2
    if c <= 10: return 3
    return 4


def ratio_bucket(p, n):
    t = p + n
    if t <= 0: return 0
    r = p / t
    if r < 0.20: return 1
    if r < 0.40: return 2
    if r < 0.60: return 3
    if r < 0.80: return 4
    return 5


def build_history_raw_features(splits):
    up_v = defaultdict(int); un_v = defaultdict(int)
    up_a = defaultdict(int); un_a = defaultdict(int)
    raw, labels, users = {}, {}, {}

    def row_feats(row, update=False):
        date, u, v, a, tab, dur, y = row
        dur_bucket = int(dur) // 10000
        kv = (u, v); ka = (u, a)
        pv, nv = up_v[kv], un_v[kv]
        pa, na = up_a[ka], un_a[ka]
        feats = [u, v, a, tab, dur_bucket,
                 pos_bucket(pv), neg_bucket(nv),
                 pos_bucket(pa), neg_bucket(na),
                 ratio_bucket(pa, na), ratio_bucket(pv, nv)]
        if update:
            if y > 0.5:
                up_v[kv] += 1; up_a[ka] += 1
            else:
                un_v[kv] += 1; un_a[ka] += 1
        return feats

    feats, ys, us = [], [], []
    for row in splits['train']:
        feats.append(row_feats(row, update=True))
        ys.append(row[6]); us.append(row[1])
    raw['train'] = feats
    labels['train'] = np.asarray(ys, dtype=np.float32)
    users['train'] = np.asarray(us, dtype=np.int64)

    for sp in ('valid', 'test'):
        feats, ys, us = [], [], []
        for row in splits[sp]:
            feats.append(row_feats(row, update=False))
            ys.append(row[6]); us.append(row[1])
        raw[sp] = feats
        labels[sp] = np.asarray(ys, dtype=np.float32)
        users[sp] = np.asarray(us, dtype=np.int64)
    return raw, labels, users


def encode_history(splits):
    raw, labels, users = build_history_raw_features(splits)
    n_fields = len(raw['train'][0])
    maps, offset = [], 0
    for j in range(n_fields):
        vals = set()
        for sp in ('train', 'valid', 'test'):
            vals.update(r[j] for r in raw[sp])
        mp = {v: offset + i for i, v in enumerate(sorted(vals))}
        maps.append(mp)
        offset += len(mp)
    enc = {}
    for sp in ('train', 'valid', 'test'):
        X = np.empty((len(raw[sp]), n_fields), dtype=np.int64)
        for i, r in enumerate(raw[sp]):
            for j, v in enumerate(r):
                X[i, j] = maps[j][v]
        enc[sp] = (X, labels[sp], users[sp])
    return enc, offset


def make_bpr_training_arrays(ytr, utr):
    y = np.asarray(ytr); u = np.asarray(utr)
    order = np.argsort(u, kind='stable')
    us = u[order]
    cuts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(order)]
    neg_by_user, eligible_pos, eligible_user = {}, [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        rows = order[a:b]
        yy = y[rows]
        pos = rows[yy > 0.5]
        neg = rows[yy <= 0.5]
        if len(pos) and len(neg):
            uid = int(u[rows[0]])
            neg_by_user[uid] = neg.astype(np.int64)
            eligible_pos.append(pos.astype(np.int64))
            eligible_user.append(np.full(len(pos), uid, dtype=np.int64))
    if not eligible_pos:
        raise RuntimeError('no users with both positive and negative rows')
    return np.concatenate(eligible_pos), np.concatenate(eligible_user), neg_by_user


def train_bce(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=False):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BCE seed {seed} epoch {ep:2d} valid {va['primary']:.6f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def train_bpr(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=False):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_rows, pos_users, neg_by_user = make_bpr_training_arrays(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(pos_rows))
        t0 = time.time()
        model.train()
        for st in range(0, len(perm), bs):
            psel = perm[st:st + bs]
            pidx_np = pos_rows[psel]
            users_np = pos_users[psel]
            nidx_np = np.empty(len(users_np), dtype=np.int64)
            for j, u in enumerate(users_np):
                negs = neg_by_user[int(u)]
                nidx_np[j] = negs[rng.integers(len(negs))]
            xp = Xtr_t[torch.from_numpy(pidx_np)].to(device)
            xn = Xtr_t[torch.from_numpy(nidx_np)].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp) - model(xn))).mean()
            loss.backward(); opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BPR seed {seed} epoch {ep:2d} valid {va['primary']:.6f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def within_user_ranks(scores, users):
    scores = np.asarray(scores); users = np.asarray(users)
    out = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='stable')
    us = users[order]
    cuts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        n = len(idx)
        if n == 1:
            out[idx] = 0.5
        else:
            ord2 = np.argsort(scores[idx], kind='mergesort')
            r = np.empty(n, dtype=np.float64)
            r[ord2] = np.arange(n, dtype=np.float64) / (n - 1.0)
            out[idx] = r
    return out


def member_pred(cache_prefix, name, train_fn, enc, dim, Xout, split, outer_seed,
                member_seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache',
                              f'{cache_prefix}_{name}_split{split}_outer{outer_seed}_mseed{member_seed}.npy')
    if os.path.isfile(cache_path):
        return np.load(cache_path)
    torch.manual_seed(member_seed)
    model = train_fn(enc, dim, seed=member_seed, device=device, verbose=verbose)
    preds = model.predict(Xout, device=device).astype(np.float64)
    np.save(cache_path, preds)
    return preds


def ensemble_rank(enc, dim, Xout, uout, split, seed, device, verbose,
                  cache_prefix, bpr_name, bce_name):
    bpr_sum = np.zeros(len(Xout), dtype=np.float64)
    for ms in [seed, seed + 101, seed + 202]:
        p = member_pred(cache_prefix, bpr_name, train_bpr, enc, dim, Xout, split,
                        seed, ms, device, verbose)
        bpr_sum += within_user_ranks(p, uout)
    bpr_rank = bpr_sum / 3.0
    bce_pred = member_pred(cache_prefix, bce_name, train_bce, enc, dim, Xout,
                           split, seed, seed + 303, device, verbose)
    bce_rank = within_user_ranks(bce_pred, uout)
    return 0.70 * bpr_rank + 0.30 * bce_rank


def run_predict(splits, split='valid', seed=0, device='cpu', verbose=False):
    enc_base, dim_base = encode(splits)
    Xbase, _, uout = enc_base[split]
    base_rank = ensemble_rank(enc_base, dim_base, Xbase, uout, split, seed, device,
                              verbose, '006', 'bpr', 'bce')

    enc_hist, dim_hist = encode_history(splits)
    Xhist, _, uhist = enc_hist[split]
    hist_rank = ensemble_rank(enc_hist, dim_hist, Xhist, uhist, split, seed, device,
                              verbose, '008', 'bprhist', 'bcehist')
    # Equal weight: the history-feature mechanism has a readable 50% vote.
    return 0.50 * base_rank + 0.50 * hist_rank


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+history blend")
    scores = run_predict(splits, split=a.split, seed=a.seed, device=a.device,
                         verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(scores[:10])
