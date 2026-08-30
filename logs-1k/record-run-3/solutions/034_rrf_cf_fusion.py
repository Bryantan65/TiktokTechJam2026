"""RRF fusion of the unchanged node-29 members.

This solution reuses the member training/caching code from 033_histrepeat_boost.py
but drops the repeat-history member from the final blend.  It tests whether a
more top-heavy reciprocal-rank fusion of the strong seed-2 MTL/stat/CF members
can recover nDCG@5 beyond the linear percentile blend used by node 29.
"""
import argparse, os, sys, importlib.util
import numpy as np
import torch

# Load helper/member code from the previous complete solution.  If member caches
# are absent, its training functions still rebuild them from scratch.
SOL_DIR = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(SOL_DIR, '033_histrepeat_boost.py')
spec = importlib.util.spec_from_file_location('histboost033', HELPER)
histboost033 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(histboost033)

sys.path.insert(0, os.path.join(SOL_DIR, '..', 'kuairand-starter-kit'))
from data import load, FIELDS  # noqa


def per_user_percentile(p, users):
    p = np.asarray(p, dtype=np.float64); users = np.asarray(users)
    order = np.argsort(users, kind='stable'); su = users[order]
    bounds = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1, len(su)]
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]; sidx = idx[np.argsort(p[idx], kind='stable')]
        n = len(sidx); out[sidx] = 0.0 if n <= 1 else np.arange(n, dtype=np.float64) / (n - 1.0)
    return out


def weighted_rrf(comps, weights, users, k=25.0):
    users = np.asarray(users); nall = len(users)
    out = np.zeros(nall, dtype=np.float64)
    order = np.argsort(users, kind='stable'); su = users[order]
    bounds = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1, nall]
    comps = [np.asarray(c, dtype=np.float64) for c in comps]
    weights = np.asarray(weights, dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]; n = len(idx)
        acc = np.zeros(n, dtype=np.float64)
        for c, w in zip(comps, weights):
            # descending rank: rank 0 is best within this user
            loc = np.argsort(-c[idx], kind='stable')
            ranks = np.empty(n, dtype=np.float64); ranks[loc] = np.arange(n, dtype=np.float64)
            acc += w / (k + ranks + 1.0)
        out[idx] = acc
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    torch.manual_seed(2)
    print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'; cache_split = 'dev'
    else:
        splits = load(a.data_dir); target = a.split; cache_split = a.split
    print({k: len(v) for k, v in splits.items()}, f'fields={FIELDS}')

    # Same seed-2 members as node 29.  rr is computed by 033 but intentionally
    # ignored here because node 32 showed repeat history hurts this blend.
    n = len(splits[target])
    rm, ra, rh, rc, rr, users = histboost033.components(
        splits, target, cache_split, 2, a.data_dir, a.device, n)

    base = 0.50 * rm + 0.50 * rh
    top_aux = base + 0.35 * np.power(np.clip(base, 0.0, 1.0), 16.0) * (ra - base)
    wcf = 0.20 + 0.25 * np.power(np.clip(base, 0.0, 1.0), 8.0)
    node29 = (1.0 - wcf) * top_aux + wcf * rc

    # Top-heavy rank fusion: require agreement between stat/listwise base and CF,
    # while letting CF dominate only in the first few ranks.  Blend back with the
    # proven linear score to protect GAUC on the long tail.
    rrf25 = weighted_rrf([rm, rh, ra, rc, node29], [0.55, 0.70, 0.18, 1.10, 0.95], users, k=25.0)
    rrf8 = weighted_rrf([base, rc, node29], [0.75, 1.15, 0.90], users, k=8.0)
    rrf = per_user_percentile(0.65 * per_user_percentile(rrf25, users) + 0.35 * per_user_percentile(rrf8, users), users)

    preds = per_user_percentile(0.58 * node29 + 0.42 * rrf, users)
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f'wrote {len(preds):,d} RRF-CF predictions for split={a.split}')
    else:
        print(preds[:10])
