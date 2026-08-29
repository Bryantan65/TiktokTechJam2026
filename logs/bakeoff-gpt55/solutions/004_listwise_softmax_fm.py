"""FM trained with a within-user listwise softmax loss.

For each train user having both positive and negative impressions, optimise
    logsumexp(scores over all impressions) - logsumexp(scores over positives)
which is the negative log probability that a positive item is selected by a
softmax over that user's impression set. This directly uses same-user ranking
sets rather than independently sampled pairs.
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
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    groups = []
    n_rows = n_pos = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b].astype(np.int64)
        yy = y[idx] > 0.5
        p = int(yy.sum())
        if p and p < len(idx):
            groups.append((idx, yy.astype(np.bool_)))
            n_rows += len(idx)
            n_pos += p
    return groups, n_rows, n_pos


def batch_listwise_loss(model, Xtr_t, batch_groups, device):
    lengths = [len(g[0]) for g in batch_groups]
    idx_np = np.concatenate([g[0] for g in batch_groups])
    pos_np = np.concatenate([g[1] for g in batch_groups])
    xb = Xtr_t[torch.from_numpy(idx_np)].to(device)
    scores = model(xb)
    pos_mask = torch.from_numpy(pos_np).to(device)
    losses = []
    start = 0
    for ln in lengths:
        s = scores[start:start + ln]
        m = pos_mask[start:start + ln]
        losses.append(torch.logsumexp(s, dim=0) - torch.logsumexp(s[m], dim=0))
        start += ln
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, user_bs=256,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups, n_rows, n_pos = make_user_groups(ytr, utr)
    if verbose:
        print(f"listwise users={len(groups):,d} rows={n_rows:,d} positives={n_pos:,d}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(groups))
        model.train()
        losses = []
        for st in range(0, len(order), user_bs):
            batch_groups = [groups[i] for i in order[st:st + user_bs]]
            opt.zero_grad(set_to_none=True)
            loss = batch_listwise_loss(model, Xtr_t, batch_groups, device)
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
        print(f"\n=== listwise_softmax_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
