"""FM+BPR with explicit user behaviour history features (valid run copy).

Adds lightweight sequence-derived categorical fields to the multi-negative BPR FM:
for each row, bucketed counts of the user's prior positive/negative interactions
with the same author and same video. Train rows use only earlier train rows;
valid/test rows use the final train history only. This is a cheap DIN-inspired
way to expose target-specific user history without a heavy attention network.
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


def bucket_count(c):
    if c <= 0:
        return 0
    if c == 1:
        return 1
    if c == 2:
        return 2
    if c == 3:
        return 3
    if c <= 7:
        return 4
    if c <= 15:
        return 5
    if c <= 31:
        return 6
    return 7


def history_features(splits):
    """Return per-split categorical arrays: ua_pos, ua_neg, uv_pos, uv_neg."""
    ua_pos = defaultdict(int); ua_neg = defaultdict(int)
    uv_pos = defaultdict(int); uv_neg = defaultdict(int)
    out = {}

    def make_row(row, update):
        u = row[1]; v = row[2]; a = row[3]; y = row[6]
        ka = (u, a); kv = (u, v)
        feats = (bucket_count(ua_pos[ka]), bucket_count(ua_neg[ka]),
                 bucket_count(uv_pos[kv]), bucket_count(uv_neg[kv]))
        if update:
            if y > 0.5:
                ua_pos[ka] += 1; uv_pos[kv] += 1
            else:
                ua_neg[ka] += 1; uv_neg[kv] += 1
        return feats

    for sp in splits:
        arr = np.empty((len(splits[sp]), 4), dtype=np.int64)
        if sp == 'train':
            for i, row in enumerate(splits[sp]):
                arr[i] = make_row(row, update=True)
        else:
            for i, row in enumerate(splits[sp]):
                arr[i] = make_row(row, update=False)
        out[sp] = arr
    return out


def add_history_to_encoded(enc, hfeat):
    max_id = max(int(enc[sp][0].max()) for sp in enc) + 1
    offsets = max_id + np.arange(4, dtype=np.int64) * 8
    out = {}
    for sp, (X, y, u) in enc.items():
        H = hfeat[sp].astype(np.int64) + offsets[None, :]
        out[sp] = (np.concatenate([X.astype(np.int64), H], axis=1), y, u)
    return out, max_id + 32


def make_user_pos_negs(y, users):
    pos_by_u = defaultdict(list)
    neg_by_u = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u[uu].append(i)
        else:
            neg_by_u[uu].append(i)
    pos_idx, neg_lists = [], []
    for uu, ps in pos_by_u.items():
        ns = neg_by_u.get(uu)
        if not ns:
            continue
        ns = np.asarray(ns, dtype=np.int64)
        for p in ps:
            pos_idx.append(p)
            neg_lists.append(ns)
    return np.asarray(pos_idx, dtype=np.int64), neg_lists


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=4096, patience=4,
        seed=0, device='cpu', verbose=True, n_negs=4):
    base_enc, _ = encode(splits)
    hfeat = history_features(splits)
    enc, dim = add_history_to_encoded(base_enc, hfeat)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, neg_lists = make_user_pos_negs(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('no same-user positive/negative pairs in training data')
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        neg_idx = np.empty((len(pos_idx), n_negs), dtype=np.int64)
        for i, ns in enumerate(neg_lists):
            neg_idx[i] = ns[rng.integers(len(ns), size=n_negs)]
        order = rng.permutation(len(pos_idx))

        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            xp = Xtr_t[torch.from_numpy(pos_idx[sel])].to(device)
            xn = Xtr_t[torch.from_numpy(neg_idx[sel].reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).view(-1, 1)
            sn = model(xn).view(-1, n_negs)
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr_hist{n_negs} {np.mean(losses):.4f} | valid "
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+history_counts")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_history_valid_copy (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
