"""FM with BCE warmup followed by correctly constructed within-user BPR fine-tuning.

Bugfix of 003: encoder user outputs may be Python lists, so convert users and
labels to NumPy arrays before sorting/indexing when constructing same-user
positive/negative BPR pairs.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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


def make_user_pos_neg(users, y):
    """Return per-user positive/negative row-index arrays for users with both."""
    users = np.asarray(users)
    y = np.asarray(y)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    yy = y[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    groups = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        lab = yy[a:b]
        pos = idx[lab > 0.5]
        neg = idx[lab <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
    return groups


def sample_bpr_pairs(groups, rng):
    """One negative sampled for every positive impression, same user only."""
    pos_parts = []
    neg_parts = []
    for pos, neg in groups:
        pos_parts.append(pos)
        neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
    p = np.concatenate(pos_parts)
    n = np.concatenate(neg_parts)
    perm = rng.permutation(len(p))
    return p[perm], n[perm]


def eval_and_maybe_save(model, Xva, yva, uva, best, best_state, bad, phase, ep,
                        losses, verbose, device, t0):
    va = evaluate(uva, yva, model.predict(Xva, device=device))
    if verbose:
        print(f"  {phase:4s} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
              f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
              f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
    if va['primary'] > best + 1e-5:
        best = va['primary']
        best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
    return best, best_state, bad


def run(splits, k=16, lr=0.001, l2=1e-6, bce_epochs=12, bpr_epochs=12,
        bs=8192, patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, bce_epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = bce_loss(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        best, best_state, bad = eval_and_maybe_save(
            model, Xva, yva, uva, best, best_state, bad, 'bce', ep,
            losses, verbose, device, t0)
        if bad >= patience:
            if verbose:
                print(f"  BCE early stop at epoch {ep}")
            break

    groups = make_user_pos_neg(utr, ytr)
    if verbose:
        npos = sum(len(p) for p, _ in groups)
        print(f"  BPR groups: {len(groups):,d} users with both labels, {npos:,d} positives/pairs per epoch")

    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr * 0.5, betas=(0.9, 0.999), eps=1e-8)
    bad = 0
    for ep in range(1, bpr_epochs + 1):
        pidx, nidx = sample_bpr_pairs(groups, rng)
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs])
            ns = torch.from_numpy(nidx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            loss = F.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        best, best_state, bad = eval_and_maybe_save(
            model, Xva, yva, uva, best, best_state, bad, 'bpr', ep,
            losses, verbose, device, t0)
        if bad >= patience:
            if verbose:
                print(f"  BPR early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40, help='accepted for compatibility; unused')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, seed=a.seed,
                     device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bce_then_bpr_fix_users (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
