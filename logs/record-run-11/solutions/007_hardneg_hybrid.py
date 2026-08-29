"""FM with BPR+BCE using same-user hard negative sampling.

Refines node 5.  Instead of pairing each positive with a uniformly sampled
same-user negative, each BPR epoch scores the train rows with the current model
and, for every positive, samples a few same-user negative candidates then uses
the highest-scoring one.  This should focus updates on mistakes that affect the
within-user top of the ranking while retaining node 5's balanced BCE anchor.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

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
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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


def build_user_groups(users, y):
    pos_by_u, neg_by_u = defaultdict(list), defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0.5:
            pos_by_u[u].append(i)
        else:
            neg_by_u[u].append(i)
    groups = []
    for u in pos_by_u.keys():
        if u in neg_by_u:
            groups.append((np.asarray(pos_by_u[u], dtype=np.int64),
                           np.asarray(neg_by_u[u], dtype=np.int64)))
    return groups


def make_pairs(groups, rng, train_scores=None, hard_k=8, hard_frac=0.75):
    """Sample same-user pairs, usually choosing the hardest of hard_k negatives."""
    left, right = [], []
    order_groups = np.arange(len(groups))
    rng.shuffle(order_groups)
    for gi in order_groups:
        ps, ns = groups[gi]
        m = len(ps)
        if train_scores is None or hard_k <= 1:
            chosen = rng.choice(ns, size=m, replace=True)
        else:
            use_hard = rng.random(m) < hard_frac
            chosen = rng.choice(ns, size=m, replace=True)
            hh = int(use_hard.sum())
            if hh > 0:
                cand = rng.choice(ns, size=(hh, hard_k), replace=True)
                best = np.argmax(train_scores[cand], axis=1)
                chosen[use_hard] = cand[np.arange(hh), best]
        left.append(ps)
        right.append(chosen.astype(np.int64, copy=False))
    p = np.concatenate(left)
    n = np.concatenate(right)
    perm = rng.permutation(len(p))
    return p[perm], n[perm]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, bce_warmup=1, bce_weight=0.15,
        hard_k=8, hard_frac=0.75):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bce = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    groups = build_user_groups(utr, ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        if ep <= bce_warmup:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), bs):
                sel = torch.from_numpy(idx[i:i + bs])
                xb = Xtr_t[sel].to(device)
                yb = ytr_t[sel].to(device)
                opt.zero_grad(set_to_none=True)
                loss = bce(model(xb), yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        else:
            train_scores = model.predict(Xtr, device=device)
            pidx, nidx = make_pairs(groups, rng, train_scores=train_scores,
                                    hard_k=hard_k, hard_frac=hard_frac)
            for i in range(0, len(pidx), bs):
                ps = torch.from_numpy(pidx[i:i + bs])
                ns = torch.from_numpy(nidx[i:i + bs])
                xp = Xtr_t[ps].to(device)
                xn = Xtr_t[ns].to(device)
                opt.zero_grad(set_to_none=True)
                sp = model(xp)
                sn = model(xn)
                bpr_loss = torch.nn.functional.softplus(-(sp - sn)).mean()
                point_scores = torch.cat([sp, sn])
                point_labels = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
                loss = bpr_loss + bce_weight * bce(point_scores, point_labels)
                loss.backward()
                opt.step()
                losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            phase = 'bce' if ep <= bce_warmup else 'hard_bpr_hybrid'
            print(f"  epoch {ep:2d} {phase} | loss {np.mean(losses):.4f} | valid "
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
        print(f"\n=== hardneg_hybrid (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
