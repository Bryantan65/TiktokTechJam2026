"""Factorization Machine trained with a same-user BPR pairwise ranking loss.

This keeps the baseline FM architecture and encoding, but replaces pointwise BCE
with BPR: for each positive impression, sample a negative impression from the
same user and maximise log sigmoid(score_pos - score_neg). The metric ranks
within user, so cross-user pairs are deliberately not used.
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


def build_pair_pools(y, users):
    """Return positive indices whose user has negatives, and per-user negatives."""
    pos_by_u = {}
    neg_by_u = {}
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u.setdefault(uu, []).append(i)
        else:
            neg_by_u.setdefault(uu, []).append(i)

    pos_idx = []
    pos_user = []
    neg_by_u_arr = {}
    for uu, ps in pos_by_u.items():
        ns = neg_by_u.get(uu)
        if ns:
            pos_idx.extend(ps)
            pos_user.extend([uu] * len(ps))
            neg_by_u_arr[uu] = np.asarray(ns, dtype=np.int64)
    return np.asarray(pos_idx, dtype=np.int64), np.asarray(pos_user, dtype=object), neg_by_u_arr


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    pos_idx, pos_user, neg_by_user = build_pair_pools(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('no same-user positive/negative pairs found')

    # Match the pointwise baseline's amount of data per epoch approximately.
    pairs_per_epoch = len(ytr)
    steps_per_epoch = int(np.ceil(pairs_per_epoch / bs))

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            pick = rng.integers(0, len(pos_idx), size=bs)
            pidx_np = pos_idx[pick]
            # One sampled negative from the same user for each positive.
            nidx_np = np.empty(bs, dtype=np.int64)
            for j, uu in enumerate(pos_user[pick]):
                ns = neg_by_user[uu]
                nidx_np[j] = ns[rng.integers(0, len(ns))]

            pidx = torch.from_numpy(pidx_np)
            nidx = torch.from_numpy(nidx_np)
            xp = Xtr_t[pidx].to(device)
            xn = Xtr_t[nidx].to(device)
            x = torch.cat([xp, xn], dim=0)

            opt.zero_grad(set_to_none=True)
            s = model(x)
            sp, sn = s[:len(xp)], s[len(xp):]
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
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
