"""Rank-averaged ensemble of pointwise BCE FM and within-user BPR FM.

The best standalone model is BPR-3neg, but BCE optimises a different signal and
may preserve useful calibrated/static effects. To avoid arbitrary logit scale
mismatch, final predictions are converted to within-user percentile ranks and
then blended with BPR dominant weight.
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


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
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


def train_bce(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed + 123).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed + 12345)
    best, best_state, bad = -1.0, None, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        model.train()
        losses = []
        for i in range(0, n, bs):
            idx = torch.from_numpy(perm[i:i + bs])
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
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


def train_bpr(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, negs_per_pos=3, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    groups, n_pos = make_user_pairs(ytr, utr)
    if verbose:
        print(f"BPR users={len(groups):,d} positives={n_pos:,d} "
              f"epoch_pairs={n_pos * negs_per_pos:,d}")
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
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
            xp = Xtr_t[ip].to(device)
            xn = Xtr_t[ineg].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp) - model(xn))).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
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
    enc, dim = encode(splits)
    if verbose:
        print("training BPR-3neg member")
    bpr = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                    device=device, verbose=verbose)
    if verbose:
        print("training BCE member")
    bce = train_bce(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                    device=device, verbose=verbose)
    return bpr, bce, enc


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

    bpr, bce, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                        device=a.device, verbose=a.out is None)

    X, y, users = enc[target]
    sbpr = bpr.predict(X, device=a.device)
    sbce = bce.predict(X, device=a.device)
    rbpr = user_percentile_ranks(sbpr, users)
    rbce = user_percentile_ranks(sbce, users)
    scores = 0.70 * rbpr + 0.30 * rbce

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bce_bpr_rank_ensemble (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                s1 = bpr.predict(Xs, device=a.device)
                s2 = bce.predict(Xs, device=a.device)
                ss = 0.70 * user_percentile_ranks(s1, us) + 0.30 * user_percentile_ranks(s2, us)
                r = evaluate(us, ys, ss)
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
