"""LightGBM LambdaRank with train-only historical statistics.

Debug of 005: LightGBM caps each ranking query at 10k rows, so large users are
split into same-user chunks.  To keep runtime below the harness limit this draft
trains LambdaRank on a seeded sample of train rows while computing historical
statistics from the full train split.
"""
import argparse
import os
import sys

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, FIELDS  # noqa: E402

DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)
CAT_NAMES = ['user', 'video', 'author', 'tab', 'dur_bucket', 'date']
STAT_KEYS = [
    'video', 'author', 'tab', 'dur', 'date', 'author_tab', 'video_tab',
    'user_author', 'user_tab', 'user_dur', 'user_video'
]
N_CAT = 6
N_NUM = 2 + 2 * len(STAT_KEYS)
N_FEAT = N_CAT + N_NUM


def dur_bucket(x):
    return int(np.log1p(float(x)) // 1)


def make_maps(train_rows):
    maps = {name: {} for name in CAT_NAMES}
    cols = {'user': USER, 'video': VIDEO, 'author': AUTHOR, 'tab': TAB, 'date': DATE}
    for r in train_rows:
        for name, col in cols.items():
            d = maps[name]
            v = r[col]
            if v not in d:
                d[v] = len(d)
        db = dur_bucket(r[DUR])
        d = maps['dur_bucket']
        if db not in d:
            d[db] = len(d)
    return maps


def add_stat(tab, key, y):
    c, s = tab.get(key, (0, 0.0))
    tab[key] = (c + 1, s + float(y))


def build_stats(train_rows):
    stats = {k: {} for k in STAT_KEYS}
    n = 0
    sy = 0.0
    for r in train_rows:
        y = float(r[LABEL])
        u, v, a, t = r[USER], r[VIDEO], r[AUTHOR], r[TAB]
        db = dur_bucket(r[DUR])
        dt = r[DATE]
        add_stat(stats['video'], v, y)
        add_stat(stats['author'], a, y)
        add_stat(stats['tab'], t, y)
        add_stat(stats['dur'], db, y)
        add_stat(stats['date'], dt, y)
        add_stat(stats['author_tab'], (a, t), y)
        add_stat(stats['video_tab'], (v, t), y)
        add_stat(stats['user_author'], (u, a), y)
        add_stat(stats['user_tab'], (u, t), y)
        add_stat(stats['user_dur'], (u, db), y)
        add_stat(stats['user_video'], (u, v), y)
        n += 1
        sy += y
    return stats, sy / max(n, 1)


def lookup(tab, key, prior, smooth=20.0):
    c, s = tab.get(key, (0, 0.0))
    return (s + smooth * prior) / (c + smooth), np.log1p(c)


def build_features(rows, maps, stats, prior, need_y=True):
    n = len(rows)
    X = np.empty((n, N_FEAT), dtype=np.float32)
    users = np.empty(n, dtype=np.int32)
    y = np.empty(n, dtype=np.float32) if need_y else None
    local_users = {}
    for i, r in enumerate(rows):
        u, v, a, t = r[USER], r[VIDEO], r[AUTHOR], r[TAB]
        dur_ms = float(r[DUR])
        db = dur_bucket(dur_ms)
        dt = r[DATE]
        if u not in local_users:
            local_users[u] = len(local_users)
        users[i] = local_users[u]
        if need_y:
            y[i] = float(r[LABEL])
        X[i, :N_CAT] = [
            maps['user'].get(u, -1),
            maps['video'].get(v, -1),
            maps['author'].get(a, -1),
            maps['tab'].get(t, -1),
            maps['dur_bucket'].get(db, -1),
            maps['date'].get(dt, -1),
        ]
        nums = [np.log1p(dur_ms), dur_ms / 1000.0]
        for name, key in [
            ('video', v), ('author', a), ('tab', t), ('dur', db), ('date', dt),
            ('author_tab', (a, t)), ('video_tab', (v, t)),
            ('user_author', (u, a)), ('user_tab', (u, t)),
            ('user_dur', (u, db)), ('user_video', (u, v))
        ]:
            m, lc = lookup(stats[name], key, prior)
            nums.extend([m, lc])
        X[i, N_CAT:] = nums
    return X, y, users


def sort_for_rank(X, y, users, max_group=10000):
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    _, counts = np.unique(us, return_counts=True)
    groups = []
    for c in counts.tolist():
        while c > max_group:
            groups.append(max_group)
            c -= max_group
        if c > 0:
            groups.append(c)
    return X[order], y[order], groups


def sample_train_rows(rows, seed, max_rows=800000):
    n = len(rows)
    if n <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    idx.sort()
    return [rows[int(i)] for i in idx]


def train_and_predict(splits, target, seed=0):
    train_rows = splits['train']
    maps = make_maps(train_rows)
    stats, prior = build_stats(train_rows)
    sampled = sample_train_rows(train_rows, seed, max_rows=800000)
    Xtr, ytr, utr = build_features(sampled, maps, stats, prior, need_y=True)
    Xtr_s, ytr_s, gtr = sort_for_rank(Xtr, ytr, utr, max_group=10000)

    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=gtr,
                         categorical_feature=list(range(N_CAT)), free_raw_data=True)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 150,
        'min_data_per_group': 5,
        'lambda_l2': 1.0,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'max_bin': 255,
        'verbosity': -1,
        'seed': seed,
        'feature_fraction_seed': seed + 17,
        'bagging_seed': seed + 31,
        'data_random_seed': seed + 47,
        'num_threads': 0,
    }
    model = lgb.train(params, dtrain, num_boost_round=100,
                      callbacks=[lgb.log_evaluation(25)])

    Xt, _, _ = build_features(splits[target], maps, stats, prior, need_y=False)
    return model.predict(Xt, num_iteration=model.current_iteration())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")
    preds = train_and_predict(splits, target, seed=a.seed)
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
