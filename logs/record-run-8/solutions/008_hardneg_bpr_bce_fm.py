"""FM trained with BPR+BCE using a mix of random and hard same-user negatives.

This is a targeted refinement of 003_bpr_bce_fm.py.  The loss is unchanged,
but after the first epoch part of the sampled negative rows are drawn from the
currently highest-scoring negatives for that same user.  This keeps the within-
user pairwise objective while focusing updates on mistakes that matter most for
GAUC/nDCG top ranks.
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


def build_hard_pools(groups, scores, topn=20):
    pools = []
    for _, n in groups:
        if len(n) <= topn:
            pools.append(n)
        else:
            sc = scores[n]
            # argpartition is O(n); sort only the selected top tail for stable sampling.
            idx = np.argpartition(sc, -topn)[-topn:]
            idx = idx[np.argsort(sc[idx])[::-1]]
            pools.append(n[idx])
    return pools


def sample_pairs(groups, rng, hard_pools=None, hard_frac=0.35):
    pos_parts = []
    neg_parts = []
    use_hard = hard_pools is not None and hard_frac > 0.0
    for gi, (p, n) in enumerate(groups):
        m = len(p)
        neg = rng.choice(n, size=m, replace=True)
        if use_hard:
            mask = rng.random(m) < hard_frac
            if mask.any():
                hp = hard_pools[gi]
                neg[mask] = rng.choice(hp, size=int(mask.sum()), replace=True)
        pos_parts.append(p)
        neg_parts.append(neg)
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, bce_weight=0.15,
        hard_frac=0.35, hard_topn=20):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups = build_user_groups(ytr, utr)
    if verbose:
        n_pairs = sum(len(p) for p, _ in groups)
        print(f"BPR eligible users={len(groups):,d}, sampled pairs/epoch={n_pairs:,d}, "
              f"bce_weight={bce_weight}, hard_frac={hard_frac}, hard_topn={hard_topn}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    hard_pools = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if ep > 1 and hard_frac > 0.0:
            tr_scores = model.predict(Xtr, device=device)
            hard_pools = build_hard_pools(groups, tr_scores, topn=hard_topn)
        pos_idx, neg_idx = sample_pairs(groups, rng, hard_pools=hard_pools,
                                        hard_frac=hard_frac)
        model.train()
        losses = []
        bprs = []
        bces = []
        for i in range(0, len(pos_idx), bs):
            psel = torch.from_numpy(pos_idx[i:i + bs])
            nsel = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            bpr = -torch.nn.functional.logsigmoid(sp - sn).mean()
            bce = 0.5 * (torch.nn.functional.softplus(-sp).mean() +
                         torch.nn.functional.softplus(sn).mean())
            loss = bpr + bce_weight * bce
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bprs.append(bpr.item())
            bces.append(bce.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} bpr {np.mean(bprs):.4f} bce {np.mean(bces):.4f} | valid "
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
    ap.add_argument('--bce_weight', type=float, default=0.15)
    ap.add_argument('--hard_frac', type=float, default=0.35)
    ap.add_argument('--hard_topn', type=int, default=20)
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
                     bce_weight=a.bce_weight, hard_frac=a.hard_frac,
                     hard_topn=a.hard_topn)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== hardneg_bpr_bce_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
