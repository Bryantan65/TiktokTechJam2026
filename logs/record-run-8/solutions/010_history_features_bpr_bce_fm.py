"""BPR+BCE FM with causal user-history categorical features.

A light-weight user behaviour sequence draft: instead of a full DIN network, append
candidate-specific summaries of the user's prior interactions (same author/video,
tab and duration-bin history).  Training features are causal within the train log;
validation/test features use train history only, so target labels are not leaked.
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


def cnt_bin(c):
    # 0, 1, 2, 3, 4-7, 8+
    if c <= 0:
        return 0
    if c <= 3:
        return int(c)
    if c <= 7:
        return 4
    return 5


def small_cnt_bin(c):
    # 0, 1, 2, 3+
    if c <= 0:
        return 0
    if c == 1:
        return 1
    if c == 2:
        return 2
    return 3


def rate_bin(pos, neg):
    n = pos + neg
    if n <= 0:
        return 0
    r = pos / n
    # 1..5 for quintiles, with no-history reserved as 0
    return 1 + min(4, int(r * 5.0))


def dur_bin_ms(ms):
    # Coarse duration buckets from raw duration_ms, independent of data.encode's bucket map.
    try:
        x = int(ms)
    except Exception:
        x = 0
    if x < 10_000:
        return 0
    if x < 30_000:
        return 1
    if x < 60_000:
        return 2
    if x < 120_000:
        return 3
    if x < 300_000:
        return 4
    return 5


class HistState:
    def __init__(self):
        self.ap = defaultdict(int); self.an = defaultdict(int)
        self.vp = defaultdict(int); self.vn = defaultdict(int)
        self.tp = defaultdict(int); self.tn = defaultdict(int)
        self.dp = defaultdict(int); self.dn = defaultdict(int)

    def features(self, r):
        u, v, a, tab = r[1], r[2], r[3], r[4]
        db = dur_bin_ms(r[5])
        ka = (u, a); kv = (u, v); kt = (u, tab); kd = (u, db)
        ap, an = self.ap.get(ka, 0), self.an.get(ka, 0)
        vp, vn = self.vp.get(kv, 0), self.vn.get(kv, 0)
        tp, tn = self.tp.get(kt, 0), self.tn.get(kt, 0)
        dp, dn = self.dp.get(kd, 0), self.dn.get(kd, 0)
        return [
            cnt_bin(ap), cnt_bin(an), rate_bin(ap, an), cnt_bin(ap + an),
            small_cnt_bin(vp), small_cnt_bin(vn), cnt_bin(vp + vn),
            rate_bin(tp, tn), cnt_bin(tp + tn), rate_bin(dp, dn)
        ]

    def update(self, r):
        u, v, a, tab = r[1], r[2], r[3], r[4]
        db = dur_bin_ms(r[5])
        y = r[6]
        if y > 0.5:
            self.ap[(u, a)] += 1; self.vp[(u, v)] += 1
            self.tp[(u, tab)] += 1; self.dp[(u, db)] += 1
        else:
            self.an[(u, a)] += 1; self.vn[(u, v)] += 1
            self.tn[(u, tab)] += 1; self.dn[(u, db)] += 1


def append_history_features(splits, enc, base_dim):
    # Cardinalities for the 10 appended feature columns above.
    cards = [6, 6, 6, 6, 4, 4, 6, 6, 6, 6]
    offsets = np.asarray([base_dim + sum(cards[:i]) for i in range(len(cards))], dtype=np.int64)
    dim = base_dim + sum(cards)

    state = HistState()
    hist_cols = {}

    # Train is causal: feature first, then update with this row's label.
    cols = np.empty((len(splits['train']), len(cards)), dtype=np.int64)
    for i, r in enumerate(splits['train']):
        cols[i] = np.asarray(state.features(r), dtype=np.int64) + offsets
        state.update(r)
    hist_cols['train'] = cols

    # Evaluation rows use only completed train history; do not update with valid/test labels.
    for sp, rows in splits.items():
        if sp == 'train':
            continue
        cols = np.empty((len(rows), len(cards)), dtype=np.int64)
        for i, r in enumerate(rows):
            cols[i] = np.asarray(state.features(r), dtype=np.int64) + offsets
        hist_cols[sp] = cols

    out = {}
    for sp in splits:
        X, y, users = enc[sp]
        out[sp] = (np.concatenate([X.astype(np.int64), hist_cols[sp]], axis=1), y, users)
    return out, dim


def build_user_groups(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)
    groups = []
    for uu, p in pos.items():
        n = neg.get(uu)
        if n:
            groups.append((np.asarray(p, dtype=np.int64), np.asarray(n, dtype=np.int64)))
    return groups


def sample_pairs(groups, rng):
    pos_parts = []
    neg_parts = []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=len(p), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, bce_weight=0.15):
    base_enc, base_dim = encode(splits)
    enc, dim = append_history_features(splits, base_enc, base_dim)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups = build_user_groups(ytr, utr)
    if verbose:
        n_pairs = sum(len(p) for p, _ in groups)
        print(f"BPR eligible users={len(groups):,d}, sampled pairs/epoch={n_pairs:,d}, "
              f"base_dim={base_dim:,d}, hist_dim={dim:,d}, bce_weight={bce_weight}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        model.train()
        losses = []
        bprs = []
        bces = []
        for i in range(0, len(pos_idx), bs):
            psel = torch.from_numpy(pos_idx[i:i + bs])
            nsel = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            bpr = -torch.nn.functional.logsigmoid(sp - sn).mean()
            bce = 0.5 * (torch.nn.functional.softplus(-sp).mean() +
                         torch.nn.functional.softplus(sn).mean())
            loss = bpr + bce_weight * bce
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bprs.append(bpr.item())
            bces.append(bce.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} bpr {np.mean(bprs):.4f} bce {np.mean(bces):.4f} | valid "
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
    ap.add_argument('--bce_weight', type=float, default=0.15)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS} + causal history bins")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None,
                     bce_weight=a.bce_weight)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== history_features_bpr_bce_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
