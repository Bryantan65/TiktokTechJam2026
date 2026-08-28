"""FM trained with a per-user listwise softmax loss.

For each train user with at least one positive and one negative impression, optimise
-log P(any positive | that user's impressed items) = logsumexp(all scores) -
logsumexp(positive scores).  This directly pushes positives above negatives inside
the same user list, closer to GAUC/nDCG than pointwise BCE.

Reference idea: ListMLE/listwise learning-to-rank softmax likelihood.
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


def make_user_groups(y, users):
    """Return row-index arrays for users having both classes."""
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    cuts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1, len(su)]
    groups = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        s = float(y[idx].sum())
        if s > 0.0 and s < (b - a):
            groups.append(idx.astype(np.int64, copy=False))
    return groups


def segment_logsumexp(z, gid, n_groups):
    # z: (N,), gid: int64 in [0, n_groups).  PyTorch scatter_reduce gives a
    # vectorised stable logsumexp over variable-length user segments.
    maxv = torch.full((n_groups,), -torch.inf, device=z.device, dtype=z.dtype)
    maxv.scatter_reduce_(0, gid, z, reduce='amax', include_self=True)
    expv = torch.exp(z - maxv[gid])
    sumv = torch.zeros((n_groups,), device=z.device, dtype=z.dtype)
    sumv.scatter_add_(0, gid, expv)
    return maxv + torch.log(sumv.clamp_min(1e-30))


def listwise_loss(logits, y, gid, n_groups):
    lse_all = segment_logsumexp(logits, gid, n_groups)
    pos = y > 0.5
    lse_pos = segment_logsumexp(logits[pos], gid[pos], n_groups)
    return (lse_all - lse_pos).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, user_bs=512, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    groups = make_user_groups(ytr, utr)

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        gorder = rng.permutation(len(groups))
        t0 = time.time()
        model.train()
        losses = []
        for st in range(0, len(gorder), user_bs):
            chosen = [groups[j] for j in gorder[st:st + user_bs]]
            sizes = np.fromiter((len(g) for g in chosen), dtype=np.int64)
            rows_np = np.concatenate(chosen)
            gid_np = np.repeat(np.arange(len(chosen), dtype=np.int64), sizes)

            rows = torch.from_numpy(rows_np)
            xb = Xtr_t[rows].to(device)
            yb = ytr_t[rows].to(device)
            gid = torch.from_numpy(gid_np).to(device)

            opt.zero_grad(set_to_none=True)
            loss = listwise_loss(model(xb), yb, gid, len(chosen))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | listloss {np.mean(losses):.4f} | valid "
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
        print(f"\n=== listwise_softmax_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
