"""FM trained with a within-user listwise softmax loss.

For each train user with both positive and negative impressions, optimize

    logsumexp(scores over all impressions for the user)
      - logsumexp(scores over positive impressions for the user)

This is the negative log probability that a softmax over the user's impressions
places mass on a positive row.  Unlike BPR it uses all negatives in the user's
slate together instead of one sampled negative per positive.
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
    """Eligible same-user slates: at least one positive and one negative."""
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    yy = y[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1]
    ends = np.r_[starts[1:], len(us)]
    groups = []
    total_rows = 0
    for st, en in zip(starts, ends):
        rows = order[st:en].astype(np.int64, copy=False)
        lab = yy[st:en] > 0.5
        if lab.any() and (~lab).any():
            groups.append((rows, lab.astype(np.bool_, copy=False)))
            total_rows += len(rows)
    return groups, total_rows


def iter_batches(groups, perm, max_rows):
    batch = []
    nrows = 0
    for gi in perm:
        g = groups[int(gi)]
        if batch and nrows + len(g[0]) > max_rows:
            yield batch
            batch = []
            nrows = 0
        batch.append(g)
        nrows += len(g[0])
    if batch:
        yield batch


def listwise_batch_loss(model, Xtr_t, batch, device):
    rows_np = np.concatenate([g[0] for g in batch])
    xb = Xtr_t[torch.from_numpy(rows_np)].to(device)
    scores = model(xb)
    losses = []
    off = 0
    for rows, posmask_np in batch:
        n = len(rows)
        s = scores[off:off + n]
        posmask = torch.from_numpy(posmask_np).to(device)
        losses.append(torch.logsumexp(s, dim=0) - torch.logsumexp(s[posmask], dim=0))
        off += n
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=16384, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    ytr = ytr.astype(np.float32, copy=False)
    utr = np.asarray(utr)
    groups, total_rows = make_user_groups(ytr, utr)
    if not groups:
        raise RuntimeError('no same-user mixed-label groups found')

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(groups))
        t0 = time.time()
        model.train()
        losses = []
        for batch in iter_batches(groups, perm, bs):
            opt.zero_grad(set_to_none=True)
            loss = listwise_batch_loss(model, Xtr_t, batch, device)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | list {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | groups {len(groups):,d} "
                  f"rows {total_rows:,d} | {time.time() - t0:.1f}s")

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
        print(f"\n=== listwise_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
