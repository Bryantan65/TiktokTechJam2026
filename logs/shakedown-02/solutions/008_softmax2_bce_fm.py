"""FM with same-user sampled softmax ranking loss plus BCE auxiliary.

This is a conservative loss-only change from the best BPR2+BCE model.  For each
positive training impression, sample two negatives from the same user and apply
a 3-way softmax loss that asks the positive to beat both negatives jointly.  A
small BCE auxiliary (0.10) is kept for pointwise calibration.
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


def build_user_pair_pools(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    cuts = np.flatnonzero(su[1:] != su[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(order)]

    pos_chunks, neg_chunks = [], []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        yy = y[idx] > 0.5
        if yy.any() and (~yy).any():
            pos_chunks.append(idx[yy].astype(np.int64, copy=False))
            neg_chunks.append(idx[~yy].astype(np.int64, copy=False))
    return pos_chunks, neg_chunks


def sample_sets(pos_chunks, neg_chunks, rng, negs_per_pos=2):
    base = sum(len(p) for p in pos_chunks)
    pos = np.empty(base, dtype=np.int64)
    negs = np.empty((base, negs_per_pos), dtype=np.int64)
    off = 0
    for p, n in zip(pos_chunks, neg_chunks):
        m = len(p)
        pos[off:off + m] = p
        for j in range(negs_per_pos):
            negs[off:off + m, j] = n[rng.integers(0, len(n), size=m)]
        off += m
    perm = rng.permutation(base)
    return pos[perm], negs[perm]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, aux_weight=0.10, negs_per_pos=2):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_chunks, neg_chunks = build_user_pair_pools(ytr, utr)
    if verbose:
        print(f"eligible users={len(pos_chunks):,d}; sets/epoch={sum(len(p) for p in pos_chunks):,d}; "
              f"negs_per_pos={negs_per_pos}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        pidx, nidx = sample_sets(pos_chunks, neg_chunks, rng, negs_per_pos=negs_per_pos)
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(pidx), bs):
            ps_np = pidx[i:i + bs]
            ns_np = nidx[i:i + bs]
            ps = torch.from_numpy(ps_np)
            ns = torch.from_numpy(ns_np.reshape(-1))
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn).view(-1, negs_per_pos)
            logits = torch.cat([sp[:, None], sn], dim=1)
            rank_loss = -torch.nn.functional.log_softmax(logits, dim=1)[:, 0].mean()
            bce_pos = torch.nn.functional.softplus(-sp).mean()
            bce_neg = torch.nn.functional.softplus(sn).mean()
            loss = rank_loss + aux_weight * 0.5 * (bce_pos + bce_neg)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | softmax+bce {np.mean(losses):.4f} | valid "
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
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
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
        print(f"\n=== softmax2_bce_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
