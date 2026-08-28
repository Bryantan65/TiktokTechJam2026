"""Seven-member within-user rank ensemble of BPR FMs.

Refines 009 by keeping exactly the same BPR training and within-user Borda/rank
combiner, but averaging seven independent members instead of five.  This tests
whether the node-8 -> node-9 gain was still mostly variance reduction from more
BPR rankers rather than a saturated ensemble.
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


def make_user_pairs(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]

    pos_lists, neg_lists = [], []
    n_pos = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_lists.append(pos.astype(np.int64))
            neg_lists.append(neg.astype(np.int64))
            n_pos += len(pos)
    return pos_lists, neg_lists, n_pos


def sample_epoch_pairs(pos_lists, neg_lists, n_pos, rng):
    pos_all = np.empty(n_pos, dtype=np.int64)
    neg_all = np.empty(n_pos, dtype=np.int64)
    off = 0
    for pos, neg in zip(pos_lists, neg_lists):
        m = len(pos)
        pos_all[off:off + m] = pos
        neg_all[off:off + m] = rng.choice(neg, size=m, replace=True)
        off += m
    perm = rng.permutation(n_pos)
    return pos_all[perm], neg_all[perm]


def train_one(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=False, tag=''):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_lists, neg_lists, n_pos = make_user_pairs(ytr, enc['train'][2])
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    if verbose:
        print(f"{tag} BPR users={len(pos_lists):,d} positives paired/epoch={n_pos:,d}")

    for ep in range(1, epochs + 1):
        pos_idx, neg_idx = sample_epoch_pairs(pos_lists, neg_lists, n_pos, rng)
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, n_pos, bs):
            psel = torch.from_numpy(pos_idx[i:i + bs])
            nsel = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

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


def train_ensemble(splits, k=16, lr=0.001, epochs=40, seed=0,
                   device='cpu', verbose=True):
    enc, dim = encode(splits)
    seeds = [seed, seed + 1009, seed + 2027, seed + 3037, seed + 4051,
             seed + 5059, seed + 6073]
    models = []
    for j, s in enumerate(seeds):
        torch.manual_seed(s)
        m = train_one(enc, dim, k=k, lr=lr, epochs=epochs, seed=s,
                      device=device, verbose=verbose, tag=f"m{j+1}/7")
        models.append(m)
    return models, enc


def within_user_rank_scores(scores, users):
    """Map scores to [0,1] ranks within each user; higher score -> higher rank."""
    scores = np.asarray(scores)
    users = np.asarray(users)
    out = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        m = e - s
        if m <= 1:
            out[idx] = 0.0
            continue
        vals = scores[idx]
        ord_local = np.argsort(vals, kind='mergesort')
        ranks = np.empty(m, dtype=np.float64)
        ranks[ord_local] = np.arange(m, dtype=np.float64) / (m - 1.0)
        out[idx] = ranks
    return out


@torch.no_grad()
def ensemble_predict(models, X, users, device='cpu'):
    pred = np.zeros(len(X), dtype=np.float64)
    for m in models:
        raw = m.predict(X, device=device)
        pred += within_user_rank_scores(raw, users)
    pred /= len(models)
    return pred


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

    models, enc = train_ensemble(splits, k=a.k, lr=a.lr,
                                 epochs=a.epochs, seed=a.seed,
                                 device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = ensemble_predict(models, X, users, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_7_rank_ensemble (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, ensemble_predict(models, Xs, us, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
