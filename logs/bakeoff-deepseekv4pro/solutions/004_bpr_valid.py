"""Pairwise BPR loss for FM - valid promote of the dev-screened BPR.

Same model and loss as 003_bpr_pairs.py. Adds a --neg_per_pos argument so the
screened configuration can be promoted to a real valid run (the harness treats
the identical source file as a duplicate). Default neg_per_pos=1 reproduces the
exact dev-screened BPR.

    loss = -log sigmoid(score_pos - score_neg)

Every pair is sampled inside one user. Reference: Rendle et al., BPR,
https://arxiv.org/abs/1205.2618
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
    users = np.asarray(users)
    _, inv = np.unique(users, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    sorted_inv = inv[order]
    bounds = np.searchsorted(sorted_inv, np.arange(inv.max() + 2))
    return [order[b:e] for b, e in zip(bounds[:-1], bounds[1:])]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, neg_per_pos=1,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))

    user_indices = group_rows_by_user(utr)
    pos_by_user, neg_by_user = [], []
    for idx in user_indices:
        pos = idx[ytr[idx] == 1]
        neg = idx[ytr[idx] == 0]
        pos_by_user.append(pos)
        neg_by_user.append(neg)

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        pos_list, neg_list = [], []
        for pos, neg in zip(pos_by_user, neg_by_user):
            if len(pos) == 0 or len(neg) == 0:
                continue
            if neg_per_pos == 1:
                neg_choice = rng.integers(0, len(neg), size=len(pos))
                pos_list.append(pos)
                neg_list.append(neg[neg_choice])
            else:
                neg_choice = rng.integers(0, len(neg),
                                          size=(len(pos), neg_per_pos))
                pos_list.append(np.repeat(pos, neg_per_pos))
                neg_list.append(neg[neg_choice].reshape(-1))

        pos_idx = np.concatenate(pos_list)
        neg_idx = np.concatenate(neg_list)
        perm = rng.permutation(len(pos_idx))
        pos_idx = pos_idx[perm]
        neg_idx = neg_idx[perm]

        losses = []
        for i in range(0, len(pos_idx), bs):
            xb_pos = Xtr_t[pos_idx[i:i + bs]].to(device)
            xb_neg = Xtr_t[neg_idx[i:i + bs]].to(device)
            opt.zero_grad(set_to_none=True)
            s_pos = model(xb_pos)
            s_neg = model(xb_neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} "
                  f"| pairs {len(pos_idx):,d} | valid GAUC {va['GAUC']:.4f} "
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
    ap.add_argument('--neg_per_pos', type=int, default=1)
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

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                     neg_per_pos=a.neg_per_pos, seed=a.seed,
                     device=a.device, verbose=a.out is None)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_valid (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
