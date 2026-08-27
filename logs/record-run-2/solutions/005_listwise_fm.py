"""FM trained with a per-user listwise softmax ranking loss.

For each training user with both positive and negative impressions, optimize the
negative log probability that a softmax over that user's impressions assigns to
any positive item: logsumexp(all scores) - logsumexp(positive scores). This is a
within-user ranking objective like BPR, but uses all items in a user's list at
once instead of sampled pairs.
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


def build_user_groups(y, users):
    """Return row-index arrays for users having at least one pos and one neg."""
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    groups = []
    start = 0
    n = len(users)
    while start < n:
        end = start + 1
        while end < n and us[end] == us[start]:
            end += 1
        idx = order[start:end].astype(np.int64)
        yy = y[idx]
        if np.any(yy > 0.5) and np.any(yy <= 0.5):
            groups.append(idx)
        start = end
    return groups


def iter_group_batches(groups, y, rng, max_rows=8192):
    """Yield shuffled batches as (concatenated row idx, segment ends, pos mask)."""
    order = rng.permutation(len(groups))
    cur = []
    rows = 0
    for gi in order:
        g = groups[gi]
        if cur and rows + len(g) > max_rows:
            idx = np.concatenate(cur).astype(np.int64)
            ends = np.cumsum([len(x) for x in cur]).astype(np.int64)
            yield idx, ends, (y[idx] > 0.5)
            cur = []
            rows = 0
        cur.append(g)
        rows += len(g)
    if cur:
        idx = np.concatenate(cur).astype(np.int64)
        ends = np.cumsum([len(x) for x in cur]).astype(np.int64)
        yield idx, ends, (y[idx] > 0.5)


def listwise_loss(scores, ends, pos_mask):
    losses = []
    start = 0
    for end, pm in zip(ends.tolist(), torch.split(pos_mask, torch.diff(torch.cat([
            torch.zeros(1, dtype=ends.dtype, device=ends.device), ends])).tolist())):
        s = scores[start:end]
        losses.append(torch.logsumexp(s, dim=0) - torch.logsumexp(s[pm], dim=0))
        start = end
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    ytr = np.asarray(ytr, dtype=np.float32)

    groups = build_user_groups(ytr, utr)
    if verbose:
        nrows = sum(len(g) for g in groups)
        print(f"listwise groups={len(groups):,d} rows/epoch={nrows:,d}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for idx, ends_np, pm_np in iter_group_batches(groups, ytr, rng, max_rows=bs):
            xb = Xtr_t[torch.from_numpy(idx)].to(device)
            ends = torch.from_numpy(ends_np).to(device)
            pm = torch.from_numpy(pm_np.astype(np.bool_)).to(device)
            opt.zero_grad(set_to_none=True)
            scores = model(xb)
            loss = listwise_loss(scores, ends, pm)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | list {np.mean(losses):.4f} | valid "
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
        print(f"\n=== listwise_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
