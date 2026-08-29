"""Listwise softmax ranking loss for FM.

Copies 001_torch_fm.py (model, data pipeline, early stopping, output contract)
and changes ONE thing: the training loss.

Instead of pointwise BCE, we score every impression of a user and use the
listwise softmax cross-entropy over that user's impressions:

    loss = logsumexp(all logits) - logsumexp(positive logits)

This treats the positive impressions as the target distribution and directly
optimises within-user ranking, which is what GAUC and nDCG@5 measure.
Rows are grouped by user during training so each loss is computed over the full
history of that user; pairs are never sampled across users.

Reference for ranking losses on implicit feedback (BPR, softmax):
Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback",
https://arxiv.org/abs/1205.2618
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402  official, unmodified
from evaluate import evaluate                  # noqa: E402  official, unmodified


class TorchFM(torch.nn.Module):
    """Same arithmetic as 001_torch_fm.py / baseline.py."""

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


def group_rows_by_user(users):
    """Return a list of row-index arrays, one per unique user.

    users may be ints or strings; np.unique gives a stable inverse and this
    avoids an O(n_users * n_rows) scan.
    """
    users = np.asarray(users)
    _, inv = np.unique(users, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    sorted_inv = inv[order]
    bounds = np.searchsorted(sorted_inv, np.arange(inv.max() + 2))
    return [order[b:e] for b, e in zip(bounds[:-1], bounds[1:])]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, users_per_batch=256,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))

    # Every loss is computed over one user's full impression list, so training
    # batches are built from users, not from randomly scattered rows.
    user_indices = group_rows_by_user(utr)
    n_users = len(user_indices)

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        user_order = rng.permutation(n_users)
        losses = []
        n_used = 0

        for i in range(0, n_users, users_per_batch):
            u_sel = user_order[i:i + users_per_batch]
            groups = [user_indices[u] for u in u_sel]
            sizes = [len(g) for g in groups]
            sel = np.concatenate(groups)

            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            scores = model(xb)

            total_loss = torch.zeros((), dtype=torch.float32, device=device)
            n_valid = 0
            start = 0
            for size in sizes:
                s = scores[start:start + size]
                y = yb[start:start + size]
                start += size
                pos = y > 0.5
                n_pos = int(pos.sum().item())
                if n_pos == 0 or n_pos == size:
                    # No contrast inside this user: no positive to push up or
                    # no negative to push down. Nothing to learn from it.
                    continue
                log_all = torch.logsumexp(s, dim=0)
                log_pos = torch.logsumexp(s[pos], dim=0)
                total_loss = total_loss + (log_all - log_pos)
                n_valid += 1

            if n_valid == 0:
                continue
            loss = total_loss / n_valid
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            n_used += n_valid

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses) if losses else float('nan'):.4f} "
                  f"| users {n_used:5d} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
                  f"| {time.time() - t0:.1f}s")

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
    ap.add_argument('--split', default='valid',
                    choices=['train', 'valid', 'test', 'dev'],
                    help='which split to write predictions for. "dev" is a '
                         'train-only holdout for screening; see the block below')
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
        print(f"\n=== listwise_softmax (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
