"""FM trained with same-user pairwise loss from continuous watch-time preference.

Instead of only comparing binary positives to negatives, this script reads raw
play_time_ms from the KuaiRand-Pure log CSVs and builds same-user pairs where one
impression has clearly higher normalized watch engagement than another.  The
preference value is a simple censored-watch-time proxy:

    value = log1p(min(play_time_ms, duration_ms) / duration_ms)

with a small bonus for the official binary label.  This keeps the official FM
features but gives the pairwise objective more graded supervision about video
engagement while avoiding validation labels for training.
"""
import argparse
import csv
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


def find_log_dir(data_dir):
    cands = [data_dir, os.path.join(data_dir, 'data'), os.path.dirname(data_dir)]
    for d in cands:
        if os.path.exists(os.path.join(d, 'log_standard_4_08_to_4_21_pure.csv')):
            return d
    return data_dir


def read_play_times(data_dir, n_total):
    """Read play_time_ms in the same row order as data.load: train+valid/test logs."""
    log_dir = find_log_dir(data_dir)
    files = ['log_standard_4_08_to_4_21_pure.csv',
             'log_standard_4_22_to_5_08_pure.csv']
    vals = []
    for fn in files:
        path = os.path.join(log_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            if 'play_time_ms' not in rdr.fieldnames:
                raise RuntimeError(f'play_time_ms not found in {path}; columns={rdr.fieldnames[:20]}')
            for row in rdr:
                try:
                    vals.append(float(row.get('play_time_ms') or 0.0))
                except ValueError:
                    vals.append(0.0)
                if len(vals) >= n_total:
                    break
        if len(vals) >= n_total:
            break
    if len(vals) < n_total:
        raise RuntimeError(f'Only read {len(vals)} raw rows but need {n_total}')
    return np.asarray(vals[:n_total], dtype=np.float32)


def engagement_values(splits, play_all):
    """Return per-split continuous engagement aligned with load() rows."""
    out = {}
    off = 0
    for sp in ('train', 'valid', 'test'):
        rows = splits[sp]
        n = len(rows)
        play = play_all[off:off+n]
        dur = np.asarray([max(float(r[5]), 1.0) for r in rows], dtype=np.float32)
        y = np.asarray([float(r[6]) for r in rows], dtype=np.float32)
        # Censored watch proxy: cap observed watch by video duration, compress tail,
        # and retain some direct alignment with the official binary target.
        ratio = np.clip(play / dur, 0.0, 1.0)
        val = np.log1p(ratio) + 0.25 * y
        out[sp] = val.astype(np.float32)
        off += n
    return out


def make_user_sorted_indices(values, users, labels, min_gap=0.05):
    """For each eligible user, keep train row ids sorted by continuous value."""
    by_u = defaultdict(list)
    for i, u in enumerate(users):
        by_u[int(u)].append(i)
    eligible_users = []
    sorted_by_u = {}
    for u, idxs in by_u.items():
        arr = np.asarray(idxs, dtype=np.int64)
        vals = values[arr]
        # Need meaningful variation and at least one official positive for metric alignment.
        if len(arr) >= 2 and labels[arr].max() > 0.5 and labels[arr].min() < 0.5 and (vals.max() - vals.min()) >= min_gap:
            order = np.argsort(vals)
            sorted_by_u[u] = arr[order]
            eligible_users.append(u)
    return np.asarray(eligible_users, dtype=np.int64), sorted_by_u


def sample_watch_pairs(eligible_users, sorted_by_u, values, rng, n_pairs):
    """Sample same-user high-engagement vs low-engagement pairs."""
    users = eligible_users[rng.integers(len(eligible_users), size=n_pairs)]
    hi = np.empty(n_pairs, dtype=np.int64)
    lo = np.empty(n_pairs, dtype=np.int64)
    weights = np.empty(n_pairs, dtype=np.float32)
    for j, u in enumerate(users):
        arr = sorted_by_u[int(u)]
        m = len(arr)
        # Draw from top and bottom halves to avoid nearly-tied noisy pairs.
        mid = max(1, m // 2)
        low_pos = rng.integers(0, mid)
        high_pos = rng.integers(mid, m) if mid < m else m - 1
        a = arr[high_pos]
        b = arr[low_pos]
        if values[a] < values[b]:
            a, b = b, a
        hi[j] = a
        lo[j] = b
        weights[j] = np.clip(values[a] - values[b], 0.05, 1.0)
    return hi, lo, weights


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    n_total = sum(len(splits[sp]) for sp in ('train', 'valid', 'test'))
    play_all = read_play_times(data_dir, n_total)
    vals = engagement_values(splits, play_all)
    train_val = vals['train']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    eligible_users, sorted_by_u = make_user_sorted_indices(train_val, utr, ytr)
    if len(eligible_users) == 0:
        raise RuntimeError('No eligible users for watch-time pairwise training')
    # Similar update count to BPR baseline: about two pairs per binary positive.
    n_pairs = int(max(1, 2 * np.sum(ytr > 0.5)))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        pidx, nidx, w = sample_watch_pairs(eligible_users, sorted_by_u, train_val, rng, n_pairs)
        order = rng.permutation(len(pidx))
        pidx, nidx, w = pidx[order], nidx[order], w[order]
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs])
            ns = torch.from_numpy(nidx[i:i + bs])
            wb = torch.from_numpy(w[i:i + bs]).to(device)
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            diff = model(xp) - model(xn)
            loss = (torch.nn.functional.softplus(-diff) * wb).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | wt-bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | pairs {len(pidx):,d} | "
                  f"users {len(eligible_users):,d} | {time.time() - t0:.1f}s")

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

    model, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== watchtime_pairwise_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
