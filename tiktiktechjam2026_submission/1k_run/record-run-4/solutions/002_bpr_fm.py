"""FM trained with a same-user pairwise BPR ranking loss.

BPR optimises -log sigmoid(s_pos - s_neg) for a positive and negative
impression from the same user, matching the within-user ranking metric better
than pointwise BCE.  Paper/formulation: Rendle et al. 2009, "BPR: Bayesian
Personalized Ranking from Implicit Feedback" https://www.cs.uwm.edu/~lazic/papers/bpr.pdf
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
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


def make_pair_sampler(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)

    eligible_pos = []
    neg_by_user = {}
    user_by_index = {}
    for u, plist in pos.items():
        if u in neg and len(neg[u]) > 0:
            eligible_pos.extend(plist)
            neg_by_user[u] = np.asarray(neg[u], dtype=np.int64)
            for ii in plist:
                user_by_index[ii] = u

    eligible_pos = np.asarray(eligible_pos, dtype=np.int64)
    if len(eligible_pos) == 0:
        raise RuntimeError('no same-user positive/negative pairs available')
    return eligible_pos, neg_by_user, user_by_index


def sample_negatives(pos_idx, rng, neg_by_user, user_by_index):
    neg_idx = np.empty(len(pos_idx), dtype=np.int64)
    # batch size is only 8192, so a Python loop is cheap relative to the model step
    for j, pi in enumerate(pos_idx):
        pool = neg_by_user[user_by_index[int(pi)]]
        neg_idx[j] = pool[rng.integers(len(pool))]
    return neg_idx


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    eligible_pos, neg_by_user, user_by_index = make_pair_sampler(ytr, utr)

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    pairs_per_epoch = len(ytr)  # roughly match the pointwise baseline's update count

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        # Sample positives with replacement; each gets a fresh negative from the same user.
        epoch_pos = eligible_pos[rng.integers(len(eligible_pos), size=pairs_per_epoch)]
        order = rng.permutation(pairs_per_epoch)
        for i in range(0, pairs_per_epoch, bs):
            pidx = epoch_pos[order[i:i + bs]]
            nidx = sample_negatives(pidx, rng, neg_by_user, user_by_index)

            xb_pos = Xtr_t[torch.from_numpy(pidx)].to(device)
            xb_neg = Xtr_t[torch.from_numpy(nidx)].to(device)
            opt.zero_grad(set_to_none=True)
            diff = model(xb_pos) - model(xb_neg)
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
