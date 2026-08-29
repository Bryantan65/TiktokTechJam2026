"""Five-member BPR+BCE FM ensemble with top-focused rank aggregation.

Node 13 averaged within-user percentile ranks from five independently seeded
BPR+BCE FMs.  This keeps the trained members/objective unchanged, but squares
each member's percentile rank before averaging.  The nonlinear Borda vote gives
more separation to items that several members put near the top, which may help
nDCG@5 while preserving the within-user rank-normalisation benefit.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch


def _add_starter_to_path():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(os.getcwd(), 'kuairand-starter-kit'),
        os.path.join(here, '..', 'kuairand-starter-kit'),
        os.path.join(here, '..', '..', 'kuairand-starter-kit'),
        os.path.join(here, '..', '..', '..', 'kuairand-starter-kit'),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            sys.path.insert(0, p)
            return
    sys.path.insert(0, 'kuairand-starter-kit')


_add_starter_to_path()
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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
        (pos if yy > 0.5 else neg)[uu].append(i)
    groups = []
    for uu, p in pos.items():
        n = neg.get(uu)
        if n:
            groups.append((np.asarray(p, dtype=np.int64), np.asarray(n, dtype=np.int64)))
    return groups


def sample_pairs(groups, rng):
    pos_parts, neg_parts = [], []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=len(p), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def train_one(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True,
              bce_weight=0.15, tag=''):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    groups = build_user_groups(ytr, utr)
    if verbose:
        print(f"{tag} BPR eligible users={len(groups):,d}, sampled pairs/epoch={sum(len(p) for p, _ in groups):,d}")

    torch.manual_seed(seed)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        model.train()
        losses = []
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

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  {tag} early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def run(splits, k=16, lr=0.001, epochs=40, seed=0, device='cpu',
        verbose=True, bce_weight=0.15, n_models=5):
    enc, dim = encode(splits)
    models = []
    for m in range(n_models):
        s = int(seed + 997 * m)
        models.append(train_one(enc, dim, k=k, lr=lr, epochs=epochs, seed=s,
                                device=device, verbose=verbose,
                                bce_weight=bce_weight, tag=f"m{m}/seed{s}"))
    return models, enc


def user_percentile_scores(scores, users):
    out = np.zeros(len(scores), dtype=np.float64)
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    for idx in groups.values():
        idx = np.asarray(idx, dtype=np.int64)
        n = len(idx)
        if n == 1:
            out[idx[0]] = 0.0
            continue
        order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
        out[idx] = ranks
    return out


@torch.no_grad()
def predict_ensemble(models, X, users, device='cpu', rank_power=2.0):
    pred = np.zeros(len(X), dtype=np.float64)
    for model in models:
        raw = model.predict(X, device=device)
        r = user_percentile_scores(raw, users)
        if rank_power != 1.0:
            r = np.power(r, rank_power)
        pred += r
    pred /= float(len(models))
    return pred


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
    ap.add_argument('--n_models', type=int, default=5)
    ap.add_argument('--rank_power', type=float, default=2.0)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}, rank ensemble={a.n_models}, power={a.rank_power}")

    models, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                      device=a.device, verbose=a.out is None,
                      bce_weight=a.bce_weight, n_models=a.n_models)
    X, y, users = enc[target]
    scores = predict_ensemble(models, X, users, device=a.device, rank_power=a.rank_power)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== rank5_power2_ensemble_bpr_bce_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, predict_ensemble(models, Xs, us, device=a.device, rank_power=a.rank_power))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
