"""FM trained with within-user BPR pairwise ranking loss.

This keeps the baseline feature set/model/optimizer, but replaces pointwise BCE with
same-user positive-vs-negative pairs:

    loss = -log sigmoid(score(pos impression) - score(neg impression))

Pairs are sampled only within a user because the target metrics rank impressions
within user; cross-user pairs would train an irrelevant ordering.
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
from data import load, encode, FIELDS          # noqa: E402  official, unmodified
from evaluate import evaluate                  # noqa: E402  official, unmodified


class TorchFM(torch.nn.Module):
    """Same arithmetic as baseline.py's FM class.

    logits = b + sum(W[x]) + 0.5 * ((sum E)^2 - sum(E^2))
    """

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


def make_user_pair_pools(y, users):
    """Return positive row ids and per-positive user's negative pool.

    One training epoch samples one negative impression from the same user for
    every positive impression whose user has at least one negative impression.
    """
    by_user = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if u not in by_user:
            by_user[u] = [[], []]  # neg, pos
        by_user[u][1 if yy > 0.5 else 0].append(i)

    pos_rows = []
    neg_pools = []
    for negs, poss in by_user.values():
        if len(poss) and len(negs):
            neg_arr = np.asarray(negs, dtype=np.int64)
            for p in poss:
                pos_rows.append(p)
                neg_pools.append(neg_arr)
    return np.asarray(pos_rows, dtype=np.int64), neg_pools


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_rows, neg_pools = make_user_pair_pools(ytr, utr)
    if len(pos_rows) == 0:
        raise RuntimeError('no same-user positive/negative pairs in training data')

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        # Sample one in-user negative per positive for this epoch.
        neg_rows = np.empty_like(pos_rows)
        for j, pool in enumerate(neg_pools):
            neg_rows[j] = pool[rng.integers(len(pool))]

        order = rng.permutation(len(pos_rows))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            pidx = torch.from_numpy(pos_rows[sel])
            nidx = torch.from_numpy(neg_rows[sel])
            xp = Xtr_t[pidx].to(device)
            xn = Xtr_t[nidx].to(device)

            opt.zero_grad(set_to_none=True)
            diff = model(xp) - model(xn)
            loss = F.softplus(-diff).mean()  # -log(sigmoid(diff))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'],
                    help='which split to write predictions for')
    ap.add_argument('--out', default=None,
                    help='write predictions here as .npy, one score per row')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
