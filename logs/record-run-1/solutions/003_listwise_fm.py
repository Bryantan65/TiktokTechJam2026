"""FM trained with same-user listwise softmax loss.

Architecture/features are unchanged from the baseline FM.  The training objective
is listwise within each user group:

    loss_u = logsumexp(scores for all impressions of u) - mean(score positives)

This is cross entropy between a softmax over a user's impressions and a uniform
target distribution over that user's positive impressions.  Users without both
positive and negative labels are skipped because they provide no within-user
ranking signal for GAUC.
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
    by_u = defaultdict(list)
    for i, u in enumerate(users):
        by_u[int(u)].append(i)

    groups = []
    pos_masks = []
    for _, idxs in by_u.items():
        idx = np.asarray(idxs, dtype=np.int64)
        yy = y[idx] > 0.5
        # All-positive/all-negative users have no within-user ordering signal.
        if yy.any() and (~yy).any():
            groups.append(idx)
            pos_masks.append(yy.astype(np.bool_))
    return groups, pos_masks


def iter_group_batches(groups, pos_masks, rng, max_rows):
    order = rng.permutation(len(groups))
    batch_g = []
    batch_m = []
    rows = 0
    for gi in order:
        g = groups[int(gi)]
        m = pos_masks[int(gi)]
        if batch_g and rows + len(g) > max_rows:
            yield batch_g, batch_m
            batch_g, batch_m, rows = [], [], 0
        batch_g.append(g)
        batch_m.append(m)
        rows += len(g)
    if batch_g:
        yield batch_g, batch_m


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    groups, pos_masks = make_user_groups(ytr, utr)
    if not groups:
        raise RuntimeError('No mixed-label users found for listwise training')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        nbatches = 0
        for batch_g, batch_m in iter_group_batches(groups, pos_masks, rng, bs):
            idx = np.concatenate(batch_g)
            xb = Xtr_t[torch.from_numpy(idx)].to(device)

            opt.zero_grad(set_to_none=True)
            scores = model(xb)

            loss_terms = []
            off = 0
            for g, pm_np in zip(batch_g, batch_m):
                n = len(g)
                sg = scores[off:off + n]
                pm = torch.from_numpy(pm_np).to(device)
                # Softmax CE with target mass spread uniformly over positives.
                loss_terms.append(torch.logsumexp(sg, dim=0) - sg[pm].mean())
                off += n
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            nbatches += 1

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | list {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | groups {len(groups):,d} "
                  f"batches {nbatches} | {time.time() - t0:.1f}s")

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
