"""FM trained with a per-user listwise softmax loss.

This keeps the baseline FM/features but replaces pointwise BCE with a ranking loss:
for each user list, maximise the model probability mass assigned to positive
items, logsumexp(scores_pos) - logsumexp(scores_all).  The loss only compares
items within the same user, matching GAUC/nDCG ranking.  Formula is the binary
multi-positive variant of ListNet-style softmax cross entropy
(see e.g. ListNet top-one probability: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/).
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


def make_user_groups(users, y):
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    su = users[order]
    bounds = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1, len(su)]
    groups = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yy = y[idx]
        npos = int(yy.sum())
        # Homogeneous users have no within-user ordering signal for AUC; all-neg
        # users also contribute 0 to nDCG regardless of ordering.
        if npos > 0 and npos < len(idx):
            groups.append(idx.astype(np.int64))
    return groups


def listwise_user_loss(model, Xtr_t, ytr_t, groups, device, max_rows=24000):
    losses = []
    rows_used = 0
    for g in groups:
        if rows_used >= max_rows and losses:
            break
        xb = Xtr_t[g].to(device)
        yb = ytr_t[g].to(device)
        s = model(xb)
        pos = yb > 0.5
        # -log P(any positive | user) = logsumexp(all) - logsumexp(positives)
        losses.append(torch.logsumexp(s, dim=0) - torch.logsumexp(s[pos], dim=0))
        rows_used += len(g)
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.003, l2=1e-6, epochs=80, users_per_batch=48,
        patience=6, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    groups = make_user_groups(utr, ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(groups))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(perm), users_per_batch):
            batch_groups = [groups[j] for j in perm[i:i + users_per_batch]]
            opt.zero_grad(set_to_none=True)
            loss = listwise_user_loss(model, Xtr_t, ytr_t, batch_groups, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
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
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--epochs', type=int, default=80)
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
        print(f"\n=== user_softmax_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
