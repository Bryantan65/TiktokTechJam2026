"""Bagged BPR+BCE FM with linear crosses plus a smoothed user-history blend.

Node 8 learned exact user crosses as model weights. This adds the same kind of
signal as a post-hoc Bayesian history rank: user-author/user-video/user-tab and
item/author CTRs from the training split only, blended at 30% so the mechanism
has a readable effect.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFMLinearCross(torch.nn.Module):
    def __init__(self, dim_base, dim_cross, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim_base, k)).astype(np.float32)))
        self.Wb = torch.nn.Parameter(torch.zeros(dim_base, dtype=torch.float32))
        self.Wc = torch.nn.Parameter(torch.zeros(dim_cross, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, Xb, Xc):
        E = self.V[Xb]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.Wb[Xb].sum(1) + self.Wc[Xc].sum(1) + inter

    @torch.no_grad()
    def predict(self, Xb, Xc, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(Xb), bs):
            xb = torch.from_numpy(Xb[i:i + bs].astype(np.int64)).to(device)
            xc = torch.from_numpy(Xc[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb, xc).cpu().numpy())
        return np.concatenate(out)


def cross_keys(row):
    u = row[1]
    return ((0, u, row[3]), (1, u, row[2]), (2, u, row[4]))


def add_linear_crosses(splits):
    base_enc, dim_base = encode(splits)
    mapping = {}
    next_id = 1
    for row in splits['train']:
        for key in cross_keys(row):
            if key not in mapping:
                mapping[key] = next_id
                next_id += 1
    enc = {}
    for sp, rows in splits.items():
        Xb, y, users = base_enc[sp]
        Xc = np.zeros((len(rows), 3), dtype=np.int64)
        for i, row in enumerate(rows):
            ks = cross_keys(row)
            Xc[i, 0] = mapping.get(ks[0], 0)
            Xc[i, 1] = mapping.get(ks[1], 0)
            Xc[i, 2] = mapping.get(ks[2], 0)
        enc[sp] = (Xb, Xc, y, users)
    return enc, dim_base, next_id


def train_bce(enc, dim_base, dim_cross, k=16, lr=0.001, l2=1e-6, epochs=40,
              bs=8192, patience=4, seed=0, device='cpu', verbose=True):
    Xb_tr, Xc_tr, ytr, _ = enc['train']
    Xb_va, Xc_va, yva, uva = enc['valid']
    model = TorchFMLinearCross(dim_base, dim_cross, k=k, seed=seed + 123).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.Wb, model.Wc], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xb_t = torch.from_numpy(Xb_tr.astype(np.int64))
    Xc_t = torch.from_numpy(Xc_tr.astype(np.int64))
    y_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed + 12345)
    best, best_state, bad = -1.0, None, 0
    n = len(Xb_tr)
    for ep in range(1, epochs + 1):
        t0 = time.time(); perm = rng.permutation(n); model.train(); losses = []
        for i in range(0, n, bs):
            idx = torch.from_numpy(perm[i:i + bs])
            xb = Xb_t[idx].to(device); xc = Xc_t[idx].to(device); yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb, xc), yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xb_va, Xc_va, device=device))
        if verbose:
            print(f"  BCE epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  BCE early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def make_user_pairs(y, users):
    y = np.asarray(y); users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    groups = []; n_pos = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        pos = idx[y[idx] > 0.5]; neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64))); n_pos += len(pos)
    return groups, n_pos


def sample_epoch_pairs(groups, n_pos, negs_per_pos, rng):
    n_pairs = n_pos * negs_per_pos
    pos_all = np.empty(n_pairs, dtype=np.int64); neg_all = np.empty(n_pairs, dtype=np.int64)
    p = 0
    for pos, neg in groups:
        mm = len(pos) * negs_per_pos
        pos_all[p:p + mm] = np.repeat(pos, negs_per_pos)
        neg_all[p:p + mm] = neg[rng.integers(0, len(neg), size=mm)]
        p += mm
    perm = rng.permutation(n_pairs)
    return pos_all[perm], neg_all[perm]


def train_bpr(enc, dim_base, dim_cross, k=16, lr=0.001, l2=1e-6, epochs=40,
              bs=8192, patience=4, negs_per_pos=3, seed=0, device='cpu', verbose=True):
    Xb_tr, Xc_tr, ytr, utr = enc['train']
    Xb_va, Xc_va, yva, uva = enc['valid']
    groups, n_pos = make_user_pairs(ytr, utr)
    if verbose:
        print(f"BPR(seed={seed}) users={len(groups):,d} positives={n_pos:,d} epoch_pairs={n_pos*negs_per_pos:,d}")
    model = TorchFMLinearCross(dim_base, dim_cross, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.Wb, model.Wc], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xb_t = torch.from_numpy(Xb_tr.astype(np.int64)); Xc_t = torch.from_numpy(Xc_tr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        pos_idx, neg_idx = sample_epoch_pairs(groups, n_pos, negs_per_pos, rng)
        t0 = time.time(); model.train(); losses = []
        for i in range(0, len(pos_idx), bs):
            ip = torch.from_numpy(pos_idx[i:i + bs]); ine = torch.from_numpy(neg_idx[i:i + bs])
            xp_b = Xb_t[ip].to(device); xp_c = Xc_t[ip].to(device)
            xn_b = Xb_t[ine].to(device); xn_c = Xc_t[ine].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp_b, xp_c) - model(xn_b, xn_c))).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xb_va, Xc_va, device=device))
        if verbose:
            print(f"  BPR epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  BPR early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def user_percentile_ranks(scores, users):
    scores = np.asarray(scores); users = np.asarray(users)
    out = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]; m = len(idx)
        if m <= 1:
            out[idx] = 0.5
        else:
            r = np.empty(m, dtype=np.float64)
            r[np.argsort(scores[idx], kind='mergesort')] = np.arange(m, dtype=np.float64) / (m - 1.0)
            out[idx] = r
    return out


def add_stat(d, key, y):
    s = d.get(key)
    if s is None:
        d[key] = [float(y), 1.0]
    else:
        s[0] += float(y); s[1] += 1.0


def build_history_stats(train_rows):
    stats = {}
    total_pos = 0.0
    for row in train_rows:
        y = float(row[6]); total_pos += y
        u, v, a, tab = row[1], row[2], row[3], row[4]
        add_stat(stats, ('ua', u, a), y)
        add_stat(stats, ('uv', u, v), y)
        add_stat(stats, ('ut', u, tab), y)
        add_stat(stats, ('a', a), y)
        add_stat(stats, ('v', v), y)
        add_stat(stats, ('t', tab), y)
    g = total_pos / max(1.0, float(len(train_rows)))
    return stats, min(max(g, 1e-5), 1 - 1e-5)


def smoothed_logit(stats, key, global_p, alpha):
    s = stats.get(key)
    if s is None:
        p = global_p
    else:
        p = (s[0] + alpha * global_p) / (s[1] + alpha)
        p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1.0 - p))


def history_scores(train_rows, target_rows):
    stats, g = build_history_stats(train_rows)
    out = np.empty(len(target_rows), dtype=np.float64)
    for i, row in enumerate(target_rows):
        u, v, a, tab = row[1], row[2], row[3], row[4]
        out[i] = (2.0 * smoothed_logit(stats, ('ua', u, a), g, 5.0) +
                  1.2 * smoothed_logit(stats, ('uv', u, v), g, 2.0) +
                  0.8 * smoothed_logit(stats, ('ut', u, tab), g, 20.0) +
                  0.7 * smoothed_logit(stats, ('a', a), g, 30.0) +
                  0.5 * smoothed_logit(stats, ('v', v), g, 20.0) +
                  0.2 * smoothed_logit(stats, ('t', tab), g, 50.0))
    return out


def run(splits, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True):
    enc, dim_base, dim_cross = add_linear_crosses(splits)
    if verbose:
        print(f"base_dim={dim_base:,d} cross_dim={dim_cross:,d}; fields={FIELDS}+linear_crosses")
        print("training BPR-3neg member A")
    bpr1 = train_bpr(enc, dim_base, dim_cross, k=k, lr=lr, epochs=epochs, seed=seed, device=device, verbose=verbose)
    if verbose: print("training BPR-3neg member B")
    bpr2 = train_bpr(enc, dim_base, dim_cross, k=k, lr=lr, epochs=epochs, seed=seed + 1009, device=device, verbose=verbose)
    if verbose: print("training BCE member")
    bce = train_bce(enc, dim_base, dim_cross, k=k, lr=lr, epochs=epochs, seed=seed, device=device, verbose=verbose)
    return bpr1, bpr2, bce, enc


def fm_blended_scores(bpr1, bpr2, bce, Xb, Xc, users, device='cpu'):
    r1 = user_percentile_ranks(bpr1.predict(Xb, Xc, device=device), users)
    r2 = user_percentile_ranks(bpr2.predict(Xb, Xc, device=device), users)
    rb = user_percentile_ranks(bce.predict(Xb, Xc, device=device), users)
    return 0.35 * r1 + 0.35 * r2 + 0.30 * rb


def final_scores(bpr1, bpr2, bce, enc_tuple, train_rows, target_rows, device='cpu'):
    Xb, Xc, _y, users = enc_tuple
    fm = fm_blended_scores(bpr1, bpr2, bce, Xb, Xc, users, device=device)
    hist = user_percentile_ranks(history_scores(train_rows, target_rows), users)
    return 0.70 * fm + 0.30 * hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    bpr1, bpr2, bce, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                               seed=a.seed, device=a.device, verbose=a.out is None)

    scores = final_scores(bpr1, bpr2, bce, enc[target], splits['train'], splits[target], device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== history_blend (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                ss = final_scores(bpr1, bpr2, bce, enc[sp], splits['train'], splits[sp], device=a.device)
                r = evaluate(enc[sp][3], enc[sp][2], ss)
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
