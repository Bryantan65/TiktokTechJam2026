"""FM with candidate-conditioned user behaviour sequence count features.

Draft direction 2: approximate the DIN idea that a user's historical behaviours should
be matched to the candidate item. Instead of a full attention net, add leakage-safe
per-user prior counts/rates for the candidate video, author, tab and duration bucket,
then train the current best multi-negative BPR objective.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFMSeq(torch.nn.Module):
    def __init__(self, dim, n_num, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.num_w = torch.nn.Parameter(torch.zeros(n_num, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X, Z):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        num = (Z * self.num_w).sum(1)
        return self.b + self.W[X].sum(1) + inter + num

    @torch.no_grad()
    def predict(self, X, Z, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            zb = torch.from_numpy(Z[i:i + bs].astype(np.float32)).to(device)
            out.append(self(xb, zb).cpu().numpy())
        return np.concatenate(out)


def make_seq_features(enc):
    """Build leakage-safe behaviour summaries aligned to encoded rows.

    For train rows, features are cumulative history before that row only. For
    valid/test rows, features use the full train history and never their labels.
    """
    cols = [1, 2, 3, 4]  # video_id, author_id, tab, dur_bucket in encoded X
    n_per = 3            # log total, log positive, smoothed positive rate
    n_num = len(cols) * n_per
    hist = {}

    def user_hist(u):
        if u not in hist:
            hist[u] = [defaultdict(lambda: [0.0, 0.0]) for _ in cols]  # total,pos
        return hist[u]

    def row_feats(u, row):
        hs = user_hist(int(u))
        vals = []
        for j, c in enumerate(cols):
            tot, pos = hs[j].get(int(row[c]), (0.0, 0.0))
            vals.append(np.log1p(tot))
            vals.append(np.log1p(pos))
            vals.append((pos + 0.5) / (tot + 1.0))
        return vals

    out = {}
    Xtr, ytr, utr = enc['train']
    Ztr = np.empty((len(Xtr), n_num), dtype=np.float32)
    for i in range(len(Xtr)):
        u = int(utr[i])
        row = Xtr[i]
        Ztr[i] = row_feats(u, row)
        hs = user_hist(u)
        yy = 1.0 if ytr[i] > 0.5 else 0.0
        for j, c in enumerate(cols):
            rec = hs[j][int(row[c])]
            rec[0] += 1.0
            rec[1] += yy
    out['train'] = Ztr

    for sp in ('valid', 'test'):
        X, y, uarr = enc[sp]
        Z = np.empty((len(X), n_num), dtype=np.float32)
        for i in range(len(X)):
            Z[i] = row_feats(int(uarr[i]), X[i])
        out[sp] = Z

    mu = out['train'].mean(axis=0, keepdims=True)
    sd = out['train'].std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    for sp in out:
        out[sp] = ((out[sp] - mu) / sd).astype(np.float32)
    return out


def make_user_pair_pools(y, users):
    by_user = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if u not in by_user:
            by_user[u] = [[], []]  # neg, pos
        by_user[u][1 if yy > 0.5 else 0].append(i)

    pos_rows = []
    neg_pools = []
    for negs, poss in by_user.values():
        if len(poss) and len(negs):
            neg_arr = np.asarray(negs, dtype=np.int64)
            for p in poss:
                pos_rows.append(p)
                neg_pools.append(neg_arr)
    return np.asarray(pos_rows, dtype=np.int64), neg_pools


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        neg_k=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    zfeat = make_seq_features(enc)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_rows, neg_pools = make_user_pair_pools(ytr, utr)
    if len(pos_rows) == 0:
        raise RuntimeError('no same-user positive/negative pairs in training data')

    model = TorchFMSeq(dim, zfeat['train'].shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W, model.num_w], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Ztr_t = torch.from_numpy(zfeat['train'].astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        neg_rows = np.empty((len(pos_rows), neg_k), dtype=np.int64)
        for j, pool in enumerate(neg_pools):
            neg_rows[j] = pool[rng.integers(len(pool), size=neg_k)]

        order = rng.permutation(len(pos_rows))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            pidx = torch.from_numpy(pos_rows[sel])
            nidx_np = neg_rows[sel].reshape(-1)
            nidx = torch.from_numpy(nidx_np)
            xp = Xtr_t[pidx].to(device)
            zp = Ztr_t[pidx].to(device)
            xn = Xtr_t[nidx].to(device)
            zn = Ztr_t[nidx].to(device)

            opt.zero_grad(set_to_none=True)
            sp = model(xp, zp).view(-1, 1)
            sn = model(xn, zn).view(len(sel), neg_k)
            loss = F.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, zfeat['valid'], device=device))
        if verbose:
            print(f"  epoch {ep:2d} | seq_bpr {np.mean(losses):.4f} | valid "
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
    return model, enc, zfeat


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--neg_k', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc, zfeat = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                            neg_k=a.neg_k, seed=a.seed, device=a.device,
                            verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, zfeat[a.split], device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== seqcount_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, zfeat[sp], device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
