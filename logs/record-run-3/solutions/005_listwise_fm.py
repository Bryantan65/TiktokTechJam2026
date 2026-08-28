"""FM trained with a within-user listwise softmax ranking loss.

For each training user with at least one positive and one negative impression,
optimize cross entropy between a softmax over that user's impressed items and a
uniform target distribution over the user's positives:
    loss_u = logsumexp(scores_u) - mean(scores_u[positives])
This keeps the baseline FM architecture and fields, but aligns the objective
with per-user ranking rather than pointwise calibration.
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
    """Return per-user row arrays and positive masks for mixed-label users."""
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]

    groups, masks, n_rows, n_pos = [], [], 0, 0
    for s, e in zip(starts, ends):
        idx = order[s:e].astype(np.int64)
        yy = y[idx] > 0.5
        p = int(yy.sum())
        if p > 0 and p < len(idx):
            groups.append(idx)
            masks.append(yy.astype(np.bool_))
            n_rows += len(idx)
            n_pos += p
    return groups, masks, n_rows, n_pos


def iter_user_batches(groups, masks, rng, max_rows=16384):
    """Yield shuffled batches of whole user groups capped by total rows."""
    order = rng.permutation(len(groups))
    bg, bm, rows = [], [], 0
    for j in order:
        g, m = groups[j], masks[j]
        if bg and rows + len(g) > max_rows:
            yield bg, bm
            bg, bm, rows = [], [], 0
        bg.append(g)
        bm.append(m)
        rows += len(g)
    if bg:
        yield bg, bm


def listwise_batch_loss(model, Xtr_t, batch_groups, batch_masks, device):
    idx = np.concatenate(batch_groups)
    lens = [len(g) for g in batch_groups]
    xb = Xtr_t[torch.from_numpy(idx)].to(device)
    scores = model(xb)
    losses = []
    off = 0
    for ln, pm_np in zip(lens, batch_masks):
        s = scores[off:off + ln]
        pm = torch.from_numpy(pm_np).to(device)
        # CE with target y/sum(y): -mean_pos log_softmax = logsumexp - mean_pos_score
        losses.append(torch.logsumexp(s, dim=0) - s[pm].mean())
        off += ln
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, max_rows=16384,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    groups, masks, n_rows, n_pos = make_user_groups(ytr, enc['train'][2])
    if verbose:
        print(f"Listwise users={len(groups):,d} rows/epoch={n_rows:,d} positives={n_pos:,d} max_rows={max_rows}")

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for bg, bm in iter_user_batches(groups, masks, rng, max_rows=max_rows):
            opt.zero_grad(set_to_none=True)
            loss = listwise_batch_loss(model, Xtr_t, bg, bm, device)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | listCE {np.mean(losses):.4f} | valid "
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
