"""Factorization Machine with same-user multi-negative sampled softmax loss.

Refines 003_bpr_bce_fm.py by changing the pairwise one-negative BPR loss to a
listwise sampled-softmax objective: for each positive impression, sample several
negative impressions from the same user and train the positive to be the top item
in that small within-user slate.  A small balanced BCE term is kept as in 003.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

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


def build_user_groups(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)
    groups = []
    for uu, p in pos.items():
        n = neg.get(uu)
        if n:
            groups.append((np.asarray(p, dtype=np.int64), np.asarray(n, dtype=np.int64)))
    return groups


def sample_slates(groups, rng, neg_k):
    pos_parts = []
    neg_parts = []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=(len(p), neg_k), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts, axis=0)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, neg_k=4, bce_weight=0.10):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups = build_user_groups(ytr, utr)
    if verbose:
        n_slates = sum(len(p) for p, _ in groups)
        print(f"softmax eligible users={len(groups):,d}, slates/epoch={n_slates:,d}, "
              f"neg_k={neg_k}, bce_weight={bce_weight}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_slates(groups, rng, neg_k)
        model.train()
        losses = []
        ces = []
        bces = []
        for i in range(0, len(pos_idx), bs):
            psel_np = pos_idx[i:i + bs]
            nsel_np = neg_idx[i:i + bs]
            bsz = len(psel_np)
            psel = torch.from_numpy(psel_np)
            nsel = torch.from_numpy(nsel_np.reshape(-1))
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)

            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn).view(bsz, neg_k)
            logits = torch.cat([sp.view(-1, 1), sn], dim=1)
            target = torch.zeros(bsz, dtype=torch.long, device=device)
            ce = F.cross_entropy(logits, target)
            bce = 0.5 * (F.softplus(-sp).mean() + F.softplus(sn).mean())
            loss = ce + bce_weight * bce
            loss.backward()
            opt.step()
            losses.append(loss.item())
            ces.append(ce.item())
            bces.append(bce.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} ce {np.mean(ces):.4f} "
                  f"bce {np.mean(bces):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | "
                  f"{time.time() - t0:.1f}s")

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
    ap.add_argument('--neg_k', type=int, default=4)
    ap.add_argument('--bce_weight', type=float, default=0.10)
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
                     device=a.device, verbose=a.out is None,
                     neg_k=a.neg_k, bce_weight=a.bce_weight)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== multineg_softmax_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
