"""Factorization Machine trained with within-user BPR pairwise ranking loss.

This keeps the official FM feature set/model, but replaces pointwise BCE with a
pairwise objective over (positive, negative) impressions from the same user:
    -log sigmoid(score_pos - score_neg)
Rendle's BPR is designed for personalized top-N ranking, which matches GAUC and
nDCG better than global logloss.
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


def make_positive_index_pairs(y, users):
    """Return positives and a group id for sampling same-user negatives."""
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    users_s = users[order]
    pos_indices = []
    pos_gids = []
    neg_by_gid = []

    start = 0
    gid = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and users_s[end] == users_s[start]:
            end += 1
        idx = order[start:end]
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        if len(pos) > 0 and len(neg) > 0:
            pos_indices.append(pos.astype(np.int64))
            pos_gids.append(np.full(len(pos), gid, dtype=np.int32))
            neg_by_gid.append(neg.astype(np.int64))
            gid += 1
        start = end

    if not pos_indices:
        raise RuntimeError('No users with both positive and negative impressions')
    return np.concatenate(pos_indices), np.concatenate(pos_gids), neg_by_gid


def sample_negatives_for_batch(gids, neg_by_gid, rng):
    neg = np.empty(len(gids), dtype=np.int64)
    # Grouping inside a batch avoids one Python rng call per pair.
    for g in np.unique(gids):
        m = (gids == g)
        pool = neg_by_gid[int(g)]
        neg[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
    return neg


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5,
        repeats=2, bce_weight=0.10, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_base, pos_gid_base, neg_by_gid = make_positive_index_pairs(ytr, utr)
    rng = np.random.default_rng(seed)

    best, best_state, bad = -1.0, None, 0
    pair_bs = bs

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        # One epoch presents each positive 'repeats' times, each with a fresh
        # same-user negative. This weights users by positive count, like GAUC.
        for _ in range(repeats):
            perm = rng.permutation(len(pos_base))
            for i in range(0, len(perm), pair_bs):
                psel = perm[i:i + pair_bs]
                pos_idx = pos_base[psel]
                neg_idx = sample_negatives_for_batch(pos_gid_base[psel], neg_by_gid, rng)

                xb_pos = Xtr_t[torch.from_numpy(pos_idx)].to(device)
                xb_neg = Xtr_t[torch.from_numpy(neg_idx)].to(device)
                xb = torch.cat([xb_pos, xb_neg], dim=0)

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                m = len(pos_idx)
                sp, sn = logits[:m], logits[m:]
                loss = F.softplus(-(sp - sn)).mean()
                if bce_weight > 0:
                    # A small pointwise anchor keeps the first-order terms and
                    # global score scale from drifting while BPR does the ranking.
                    labels = torch.cat([torch.ones(m, device=device),
                                        torch.zeros(m, device=device)])
                    loss = loss + bce_weight * F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr loss {np.mean(losses):.4f} | valid "
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

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
