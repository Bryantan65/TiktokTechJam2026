"""FM trained with a sampled within-user softmax ranking loss, using more negatives.

Node 3 showed the sampled listwise objective can beat pointwise BCE but was noisy.
This variant keeps the same model/training loop and samples 8 same-user negatives
per positive instead of 4, giving the softmax a harder/more complete candidate
set for top-rank separation.
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


def make_pair_sampler(y, users):
    """Return positive indices and per-positive negative pools, same user only."""
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    pos_chunks = []
    neg_pools = []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and us[j] == us[i]:
            j += 1
        idx = order[i:j]
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_chunks.append(pos.astype(np.int64, copy=False))
            neg = neg.astype(np.int64, copy=False)
            neg_pools.extend([neg] * len(pos))
        i = j
    if not pos_chunks:
        raise RuntimeError('no users with both positive and negative rows')
    return np.concatenate(pos_chunks), np.asarray(neg_pools, dtype=object)


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=4096, patience=4,
        neg_k=8, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, neg_pools = make_pair_sampler(ytr, utr)
    rng = np.random.default_rng(seed)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(pos_idx))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(perm), bs):
            psel = perm[i:i + bs]
            bsz = len(psel)
            pidx = pos_idx[psel]
            nidx = np.empty((bsz, neg_k), dtype=np.int64)
            for t, pool in enumerate(neg_pools[psel]):
                nidx[t] = pool[rng.integers(len(pool), size=neg_k)]

            xp = Xtr_t[torch.from_numpy(pidx)].to(device)
            xn_flat = Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).view(bsz, 1)
            sn = model(xn_flat).view(bsz, neg_k)
            logits = torch.cat([sp, sn], dim=1)
            target = torch.zeros(bsz, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | sampled_softmax_k8 {np.mean(losses):.4f} | valid "
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
        print(f"\n=== sampled_softmax_neg8_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
