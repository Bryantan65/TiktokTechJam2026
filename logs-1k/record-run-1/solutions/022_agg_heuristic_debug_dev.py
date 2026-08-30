"""Aggregate-rate heuristic to debug the failed LightGBM tree direction.

If this simple train-only smoothed target-encoding score is sane, then the row
join/features are usable and the problem is LightGBM training/objective.  If it
is also disastrous, the aggregate feature construction or label convention is
wrong.  No metric is computed here; the harness scores predictions.
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, FIELDS          # noqa: E402


def dur_bucket(duration_ms):
    d = int(duration_ms)
    if d < 7000:
        return 0
    if d < 15000:
        return 1
    if d < 30000:
        return 2
    if d < 60000:
        return 3
    return 4


def logit(p):
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def build_stats(rows, specs):
    cnts = [defaultdict(int) for _ in specs]
    sums = [defaultdict(float) for _ in specs]
    for r in rows:
        y = float(r[6])
        vals = row_vals(r)
        for j, cols in enumerate(specs):
            key = tuple(vals[c] for c in cols)
            if len(key) == 1:
                key = key[0]
            cnts[j][key] += 1
            sums[j][key] += y
    return cnts, sums


def row_vals(r):
    # Names: user, video, author, tab, duration bucket, date.
    return (str(r[1]), str(r[2]), str(r[3]), int(r[4]), dur_bucket(r[5]), int(r[0]))


def predict_rows(rows, train_rows, specs, cnts, sums, weights, alphas, prior, loo=False):
    out = np.empty(len(rows), dtype=np.float64)
    prior_log = logit(prior)
    for i, r in enumerate(rows):
        y = float(r[6]) if loo else 0.0
        vals = row_vals(r)
        s = 0.15 * prior_log
        wsum = 0.15
        for j, cols in enumerate(specs):
            key = tuple(vals[c] for c in cols)
            if len(key) == 1:
                key = key[0]
            c = cnts[j].get(key, 0)
            sm = sums[j].get(key, 0.0)
            if loo:
                c = max(c - 1, 0)
                sm -= y
            rate = (sm + alphas[j] * prior) / (c + alphas[j])
            s += weights[j] * logit(rate)
            wsum += abs(weights[j])
        out[i] = s / max(wsum, 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}+agg_heuristic")

    train_rows = splits['train']
    prior = float(np.mean([float(r[6]) for r in train_rows]))
    # Tuple positions after row_vals: user, video, author, tab, dur_bucket, date.
    specs = [
        (0, 1),       # user-video repeat affinity
        (0, 2),       # user-author affinity
        (0, 2, 3),    # user-author within tab
        (0, 3),       # user-tab preference
        (1,),         # global video quality
        (2,),         # global author quality
        (2, 3),       # author by tab
        (3,),         # tab prior
        (4,),         # duration prior
    ]
    # Heavily smooth sparse user-video; let user-author and global item stats dominate.
    weights = [1.20, 1.00, 0.45, 0.35, 0.75, 0.55, 0.25, 0.15, 0.10]
    alphas = [8.0, 20.0, 30.0, 50.0, 25.0, 35.0, 50.0, 100.0, 100.0]
    cnts, sums = build_stats(train_rows, specs)
    print(f"prior={prior:.5f}; built {len(specs)} aggregate tables")

    preds = predict_rows(splits[target], train_rows, specs, cnts, sums, weights, alphas,
                         prior, loo=(target == 'train'))
    # Tiny seed-controlled jitter only to give deterministic tie breaks without changing scale.
    rng = np.random.default_rng(int(a.seed))
    preds = preds + rng.normal(0.0, 1e-8, size=len(preds))
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])


if __name__ == '__main__':
    main()
