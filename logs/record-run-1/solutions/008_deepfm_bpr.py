"""DeepFM trained with same-user BPR for within-user video ranking.

This keeps the strongest known ingredient (same-user positive-vs-negative BPR)
but replaces the pure FM scorer with a DeepFM scorer:

    score = FM(x) + deep_scale * MLP(flatten(field_embeddings))

The FM part captures first/second-order sparse interactions, while the small MLP
can learn higher-order interactions among user/video/author/tab/duration fields.
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


class DeepFM(torch.nn.Module):
    def __init__(self, dim, n_fields, k=16, seed=0, hidden=(64, 32), dropout=0.05):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        layers = []
        in_dim = int(n_fields * k)
        for h in hidden:
            lin = torch.nn.Linear(in_dim, int(h))
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            in_dim = int(h)
        out = torch.nn.Linear(in_dim, 1)
        torch.nn.init.xavier_uniform_(out.weight, gain=0.1)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        self.mlp = torch.nn.Sequential(*layers)
        self.deep_scale = torch.nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def fm_score(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E

    def forward(self, X):
        fm, E = self.fm_score(X)
        deep = self.mlp(E.reshape(E.shape[0], -1)).squeeze(1)
        return fm + self.deep_scale * deep

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def make_pos_user_negpools(y, users):
    pos_by_u = defaultdict(list)
    neg_by_u = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u[int(uu)].append(i)
        else:
            neg_by_u[int(uu)].append(i)

    pos_idx = []
    pos_user = []
    neg_pools = {}
    for u, ps in pos_by_u.items():
        ns = neg_by_u.get(u)
        if ns:
            neg_pools[u] = np.asarray(ns, dtype=np.int64)
            pos_idx.extend(ps)
            pos_user.extend([u] * len(ps))
    return (np.asarray(pos_idx, dtype=np.int64),
            np.asarray(pos_user, dtype=np.int64),
            neg_pools)


def sample_pairs(pos_idx, pos_user, neg_pools, rng, multiplier=2):
    n_base = len(pos_idx)
    order = rng.permutation(n_base)
    if multiplier > 1:
        order = np.tile(order, multiplier)
        rng.shuffle(order)
    p = pos_idx[order]
    n = np.empty_like(p)
    for j, u in enumerate(pos_user[order]):
        pool = neg_pools[int(u)]
        n[j] = pool[rng.integers(len(pool))]
    return p, n


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = DeepFM(dim, n_fields=Xtr.shape[1], k=k, seed=seed).to(device)
    # Apply L2 to sparse FM parameters; keep MLP regularized mostly by dropout.
    opt = torch.optim.Adam([
        {'params': [model.V, model.W, model.deep_scale], 'weight_decay': l2},
        {'params': model.mlp.parameters(), 'weight_decay': 1e-7},
        {'params': [model.b], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, pos_user, neg_pools = make_pos_user_negpools(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('No same-user positive/negative pairs found for BPR')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        pidx, nidx = sample_pairs(pos_idx, pos_user, neg_pools, rng, multiplier=2)
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs])
            ns = torch.from_numpy(nidx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            diff = model(xp) - model(xn)
            loss = torch.nn.functional.softplus(-diff).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        va_scores = model.predict(Xva, device=device)
        va = evaluate(uva, yva, va_scores)
        if verbose:
            print(f"  epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | scale {float(model.deep_scale.detach().cpu()):.3f} "
                  f"| pairs {len(pidx):,d} | {time.time() - t0:.1f}s")

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

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== deepfm_bpr (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
