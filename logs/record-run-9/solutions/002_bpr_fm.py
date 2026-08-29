"""FM trained with a within-user pairwise BPR loss.

This keeps the official baseline features/model, but replaces pointwise BCE with
same-user positive-vs-negative comparisons so the optimized objective matches the
within-user ranking metrics better.
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


def make_bpr_training_arrays(ytr, utr):
    """Return positive row ids whose user has a negative, and neg lists by user."""
    y = np.asarray(ytr)
    u = np.asarray(utr)
    order = np.argsort(u, kind='stable')
    us = u[order]
    split = np.flatnonzero(us[1:] != us[:-1]) + 1
    chunks = np.split(order, split)

    neg_by_user = {}
    eligible_pos = []
    eligible_user = []
    for rows in chunks:
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
    pos_rows, pos_users, neg_by_user = make_bpr_training_arrays(ytr, utr)

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(pos_rows))
        t0 = time.time()
        model.train()
        losses = []
        for st in range(0, len(perm), bs):
            psel = perm[st:st + bs]
            pidx_np = pos_rows[psel]
            users_np = pos_users[psel]
            # Sample one seen negative from the same user for every positive.
            nidx_np = np.fromiter((neg_by_user[int(u)][rng.integers(len(neg_by_user[int(u)]))]
                                   for u in users_np), dtype=np.int64,
                                  count=len(users_np))

            pidx = torch.from_numpy(pidx_np).to(device)
            nidx = torch.from_numpy(nidx_np).to(device)
            xp = Xtr_t[pidx].to(device)
            xn = Xtr_t[nidx].to(device)

            opt.zero_grad(set_to_none=True)
            diff = model(xp) - model(xn)
            loss = torch.nn.functional.softplus(-diff).mean()
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
        print(f"\n=== bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
