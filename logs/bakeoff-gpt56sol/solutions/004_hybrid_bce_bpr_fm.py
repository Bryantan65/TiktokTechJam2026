"""FM trained jointly with pointwise BCE and within-user BPR."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402
from evaluate import evaluate  # noqa: E402


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


def make_pairs(y, users, rng):
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    us = users[order]
    cuts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(order)]
    pos_parts, neg_parts = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        rows = order[a:b]
        pos = rows[y[rows] > 0.5]
        neg = rows[y[rows] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, n, replace=len(pos) < n))
        neg_parts.append(rng.choice(neg, n, replace=len(neg) < n))
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k, seed).to(device)
    opt = torch.optim.Adam([
        {'params': [model.V, model.W], 'weight_decay': l2},
        {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        pos, neg = make_pairs(ytr, utr, rng)
        perm = rng.permutation(len(pos))
        t0 = time.time()
        losses = []
        model.train()
        for i in range(0, len(perm), bs):
            q = perm[i:i + bs]
            pi = pos[q]
            ni = neg[q]
            xp = Xtr_t[torch.from_numpy(pi)].to(device)
            xn = Xtr_t[torch.from_numpy(ni)].to(device)
            yp = ytr_t[torch.from_numpy(pi)].to(device)
            yn = ytr_t[torch.from_numpy(ni)].to(device)
            opt.zero_grad(set_to_none=True)
            sp, sn = model(xp), model(xn)
            bpr = -torch.nn.functional.logsigmoid(sp - sn).mean()
            bce = 0.5 * (torch.nn.functional.binary_cross_entropy_with_logits(sp, yp) +
                         torch.nn.functional.binary_cross_entropy_with_logits(sn, yn))
            loss = bce + 0.5 * bpr
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"epoch {ep:2d} loss {np.mean(losses):.4f} valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)
    scores = model.predict(enc[target][0], device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print('done', len(scores), FIELDS)
