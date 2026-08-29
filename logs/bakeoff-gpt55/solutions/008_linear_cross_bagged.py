"""Bagged BPR+BCE FM with explicit linear user cross features.

Keeps node 7's two-BPR plus BCE rank ensemble, but adds linear-only memorisation
features for (user, author), (user, video), and (user, tab). The base FM already
has low-rank user/item interactions; these crosses test whether sparse exact
within-user affinities add signal without introducing random FM interactions for
unseen pairs.
"""
import argparse
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
        V0 = rng.normal(0, 0.01, (dim_base, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
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
    # row = (date, user_id, video_id, author_id, tab, duration_ms, label)
    u = row[1]
    return ((0, u, row[3]),     # user-author affinity
            (1, u, row[2]),     # repeated user-video affinity
            (2, u, row[4]))     # personalised tab preference


def add_linear_crosses(splits):
    base_enc, dim_base = encode(splits)
    mapping = {}
    next_id = 1  # 0 is unknown and remains exactly zero (unseen in train)
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
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xb_t = torch.from_numpy(Xb_tr.astype(np.int64))
    Xc_t = torch.from_numpy(Xc_tr.astype(np.int64))
    y_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed + 12345)
    best, best_state, bad = -1.0, None, 0
    n = len(Xb_tr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        model.train()
        losses = []
        for i in range(0, n, bs):
            idx = torch.from_numpy(perm[i:i + bs])
            xb = Xb_t[idx].to(device)
            xc = Xc_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb, xc), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xb_va, Xc_va, device=device))
        if verbose:
            print(f"  BCE epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  BCE early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def make_user_pairs(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    groups = []
    n_pos = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
            n_pos += len(pos)
    return groups, n_pos


def sample_epoch_pairs(groups, n_pos, negs_per_pos, rng):
    n_pairs = n_pos * negs_per_pos
    pos_all = np.empty(n_pairs, dtype=np.int64)
    neg_all = np.empty(n_pairs, dtype=np.int64)
    p = 0
    for pos, neg in groups:
        m = len(pos)
        mm = m * negs_per_pos
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
        print(f"BPR(seed={seed}) users={len(groups):,d} positives={n_pos:,d} "
              f"epoch_pairs={n_pos * negs_per_pos:,d}")
    model = TorchFMLinearCross(dim_base, dim_cross, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.Wb, model.Wc], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xb_t = torch.from_numpy(Xb_tr.astype(np.int64))
    Xc_t = torch.from_numpy(Xc_tr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        pos_idx, neg_idx = sample_epoch_pairs(groups, n_pos, negs_per_pos, rng)
        n_pairs = len(pos_idx)
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, n_pairs, bs):
            ip = torch.from_numpy(pos_idx[i:i + bs])
            ineg = torch.from_numpy(neg_idx[i:i + bs])
            xp_b = Xb_t[ip].to(device)
            xp_c = Xc_t[ip].to(device)
            xn_b = Xb_t[ineg].to(device)
            xn_c = Xc_t[ineg].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp_b, xp_c) - model(xn_b, xn_c))).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xb_va, Xc_va, device=device))
        if verbose:
            print(f"  BPR epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  BPR early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def user_percentile_ranks(scores, users):
    scores = np.asarray(scores)
    users = np.asarray(users)
    out = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        m = len(idx)
        if m <= 1:
            out[idx] = 0.5
        else:
            r = np.empty(m, dtype=np.float64)
            r[np.argsort(scores[idx], kind='mergesort')] = np.arange(m, dtype=np.float64) / (m - 1.0)
            out[idx] = r
    return out


def run(splits, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True):
    enc, dim_base, dim_cross = add_linear_crosses(splits)
    if verbose:
        print(f"base_dim={dim_base:,d} cross_dim={dim_cross:,d}; fields={FIELDS}+linear_crosses")
        print("training BPR-3neg member A")
    bpr1 = train_bpr(enc, dim_base, dim_cross, k=k, lr=lr, epochs=epochs,
                     seed=seed, device=device, verbose=verbose)
    if verbose:
        print("training BPR-3neg member B")
    bpr2 = train_bpr(enc, dim_base, dim_cross, k=k, lr=lr, epochs=epochs,
                     seed=seed + 1009, device=device, verbose=verbose)
    if verbose:
        print("training BCE member")
    bce = train_bce(enc, dim_base, dim_cross, k=k, lr=lr, epochs=epochs,
                    seed=seed, device=device, verbose=verbose)
    return bpr1, bpr2, bce, enc


def blended_scores(bpr1, bpr2, bce, Xb, Xc, users, device='cpu'):
    r1 = user_percentile_ranks(bpr1.predict(Xb, Xc, device=device), users)
    r2 = user_percentile_ranks(bpr2.predict(Xb, Xc, device=device), users)
    rb = user_percentile_ranks(bce.predict(Xb, Xc, device=device), users)
    return 0.35 * r1 + 0.35 * r2 + 0.30 * rb


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
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    bpr1, bpr2, bce, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                               seed=a.seed, device=a.device, verbose=a.out is None)

    Xb, Xc, y, users = enc[target]
    scores = blended_scores(bpr1, bpr2, bce, Xb, Xc, users, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== linear_cross_bagged (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs_b, Xs_c, ys, us = enc[sp]
                ss = blended_scores(bpr1, bpr2, bce, Xs_b, Xs_c, us, device=a.device)
                r = evaluate(us, ys, ss)
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
