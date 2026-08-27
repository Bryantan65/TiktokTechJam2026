"""BPR-FM with temporal context features from raw CSV logs.

This extends the best same-user BPR FM by adding categorical time fields:
  [user_id, video_id, author_id, tab, dur_bucket, date, hour, weekday, tab_hour]

The starter-kit load() tuples contain date but not hourmin, so hour is read from
raw CSVs.  The raw log files are read in the documented order and then sliced to
match train/valid/test row order returned by load().
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
from data import load, FIELDS                  # noqa: E402
from evaluate import evaluate                  # noqa: E402


def dur_bucket(duration_ms):
    # Match the starter-kit field name semantically with a conservative log2 ms bucket.
    # Exact cut points are less important than consistent train/valid/test encoding.
    try:
        d = float(duration_ms)
    except Exception:
        d = 0.0
    if d <= 0:
        return 0
    return int(min(31, np.log2(d / 1000.0 + 1.0) * 4.0))


def ymd_to_weekday(yyyymmdd):
    # Zeller-like via Python stdlib would be fine, but avoid importing datetime in hot loops.
    import datetime as _dt
    y = int(yyyymmdd) // 10000
    m = (int(yyyymmdd) // 100) % 100
    d = int(yyyymmdd) % 100
    return _dt.date(y, m, d).weekday()


def read_hourmins(data_dir, total_rows):
    paths = [
        os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv'),
    ]
    hours = []
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                hm = row.get('hourmin', '')
                try:
                    hm_int = int(float(hm))
                    hour = hm_int // 100
                    if hour < 0 or hour > 23:
                        hour = 0
                except Exception:
                    hour = 0
                hours.append(hour)
    if len(hours) < total_rows:
        raise RuntimeError(f'raw logs have {len(hours)} rows, need at least {total_rows}')
    return np.asarray(hours[:total_rows], dtype=np.int16)


def encode_with_time(splits, hours_by_split):
    """Encode categorical FM fields, including date/hour context."""
    maps = []
    for _ in range(9):
        maps.append({})

    def add(mp, val):
        if val not in mp:
            mp[val] = len(mp)
        return mp[val]

    # Build maps on all splits so valid/test unseen values get IDs, as starter encode() does.
    raw_feats = {}
    for sp, rows in splits.items():
        hs = hours_by_split[sp]
        feats_sp = []
        for r, hour in zip(rows, hs):
            date, user_id, video_id, author_id, tab, duration_ms, label = r
            db = dur_bucket(duration_ms)
            wd = ymd_to_weekday(date)
            h = int(hour)
            feats = [
                ('u', int(user_id)),
                ('v', int(video_id)),
                ('a', int(author_id)),
                ('t', int(tab)),
                ('d', int(db)),
                ('date', int(date)),
                ('hour', h),
                ('wd', wd),
                ('tab_hour', (int(tab), h)),
            ]
            feats_sp.append(feats)
            for j, (_, val) in enumerate(feats):
                add(maps[j], val)
        raw_feats[sp] = feats_sp

    offsets = []
    off = 0
    for mp in maps:
        offsets.append(off)
        off += len(mp)
    dim = off

    enc = {}
    for sp, rows in splits.items():
        X = np.empty((len(rows), 9), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=np.int64)
        for i, (r, feats) in enumerate(zip(rows, raw_feats[sp])):
            y[i] = float(r[6])
            users[i] = int(r[1])
            for j, (_, val) in enumerate(feats):
                X[i, j] = offsets[j] + maps[j][val]
        enc[sp] = (X, y, users)
    return enc, dim


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


def make_pos_user_negpools(y, users):
    pos_by_u = defaultdict(list)
    neg_by_u = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u[int(uu)].append(i)
        else:
            neg_by_u[int(uu)].append(i)

    pos_idx = []
    pos_user = []
    neg_pools = {}
    for u, ps in pos_by_u.items():
        ns = neg_by_u.get(u)
        if ns:
            neg_pools[u] = np.asarray(ns, dtype=np.int64)
            pos_idx.extend(ps)
            pos_user.extend([u] * len(ps))

    return (np.asarray(pos_idx, dtype=np.int64),
            np.asarray(pos_user, dtype=np.int64),
            neg_pools)


def sample_pairs(pos_idx, pos_user, neg_pools, rng, multiplier=2):
    n_base = len(pos_idx)
    order = rng.permutation(n_base)
    if multiplier > 1:
        order = np.tile(order, multiplier)
        rng.shuffle(order)

    p = pos_idx[order]
    n = np.empty_like(p)
    for j, u in enumerate(pos_user[order]):
        pool = neg_pools[int(u)]
        n[j] = pool[rng.integers(len(pool))]
    return p, n


def run(splits, hours_by_split, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode_with_time(splits, hours_by_split)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, pos_user, neg_pools = make_pos_user_negpools(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('No same-user positive/negative pairs found for BPR')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        pidx, nidx = sample_pairs(pos_idx, pos_user, neg_pools, rng, multiplier=2)
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs])
            ns = torch.from_numpy(nidx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            diff = model(xp) - model(xn)
            loss = torch.nn.functional.softplus(-diff).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | pairs {len(pidx):,d} | "
                  f"dim {dim:,d} | {time.time() - t0:.1f}s")

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
    print({k_: len(v) for k_, v in splits.items()}, f"base_fields={FIELDS}")

    total = len(splits['train']) + len(splits['valid']) + len(splits['test'])
    all_hours = read_hourmins(a.data_dir, total)
    ntr = len(splits['train'])
    nva = len(splits['valid'])
    hours_by_split = {
        'train': all_hours[:ntr],
        'valid': all_hours[ntr:ntr + nva],
        'test': all_hours[ntr + nva:ntr + nva + len(splits['test'])],
    }

    model, enc = run(splits, hours_by_split, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== time_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
