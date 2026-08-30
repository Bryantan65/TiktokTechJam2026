"""LightGBM LambdaRank with train-only historical statistics.

This is still direction 1 (ranking loss), but uses LambdaRank/NDCG@5 directly
instead of sampling BPR pairs.  Groups are users, so the learner optimises the
same within-user ranking structure as the metric.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, FIELDS  # noqa: E402


# Tuple columns: (date, user_id, video_id, author_id, tab, duration_ms, label)
DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)


def make_maps(splits):
    maps = {name: {} for name in ['user', 'video', 'author', 'tab', 'dur_bucket', 'date']}
    cols = {'user': USER, 'video': VIDEO, 'author': AUTHOR, 'tab': TAB, 'date': DATE}
    for rows in splits.values():
        for r in rows:
            for name, col in cols.items():
                d = maps[name]
                v = r[col]
                if v not in d:
                    d[v] = len(d)
            # same bucket rule as starter-kit encode()
            db = int(np.log1p(float(r[DUR])) // 1)
            d = maps['dur_bucket']
            if db not in d:
                d[db] = len(d)
    return maps


def add_stat(tab, key, y):
    c, s = tab.get(key, (0, 0.0))
    tab[key] = (c + 1, s + float(y))


def build_stats(train_rows):
    stats = {k: {} for k in [
        'video', 'author', 'tab', 'dur', 'date', 'author_tab', 'video_tab',
        'user_author', 'user_tab', 'user_dur', 'user_video'
    ]}
    n = 0
    sy = 0.0
    for r in train_rows:
        y = r[LABEL]
        u, v, a, t = r[USER], r[VIDEO], r[AUTHOR], r[TAB]
        d = int(np.log1p(float(r[DUR])) // 1)
        dt = r[DATE]
        add_stat(stats['video'], v, y)
        add_stat(stats['author'], a, y)
        add_stat(stats['tab'], t, y)
        add_stat(stats['dur'], d, y)
        add_stat(stats['date'], dt, y)
        add_stat(stats['author_tab'], (a, t), y)
        add_stat(stats['video_tab'], (v, t), y)
        add_stat(stats['user_author'], (u, a), y)
        add_stat(stats['user_tab'], (u, t), y)
        add_stat(stats['user_dur'], (u, d), y)
        add_stat(stats['user_video'], (u, v), y)
        n += 1
        sy += float(y)
    prior = sy / max(n, 1)
    return stats, prior


def lookup(tab, key, prior, smooth=20.0):
    c, s = tab.get(key, (0, 0.0))
    return (s + smooth * prior) / (c + smooth), np.log1p(c)


def build_features(rows, maps, stats, prior):
    n = len(rows)
    # 6 categorical + 22 numerical features
    X = np.empty((n, 28), dtype=np.float32)
    users = np.empty(n, dtype=np.int32)
    y = np.empty(n, dtype=np.float32)
    min_date_code = 0
    for i, r in enumerate(rows):
        u, v, a, t = r[USER], r[VIDEO], r[AUTHOR], r[TAB]
        dur_ms = float(r[DUR])
        db_raw = int(np.log1p(dur_ms) // 1)
        dt = r[DATE]
        uc = maps['user'].get(u, -1)
        vc = maps['video'].get(v, -1)
        ac = maps['author'].get(a, -1)
        tc = maps['tab'].get(t, -1)
        dc = maps['dur_bucket'].get(db_raw, -1)
        datec = maps['date'].get(dt, -1)
        users[i] = uc
        y[i] = float(r[LABEL]) if len(r) > LABEL else 0.0
        vals = [uc, vc, ac, tc, dc, datec]
        nums = [np.log1p(dur_ms), dur_ms / 1000.0]
        for name, key in [
            ('video', v), ('author', a), ('tab', t), ('dur', db_raw), ('date', dt),
            ('author_tab', (a, t)), ('video_tab', (v, t)),
            ('user_author', (u, a)), ('user_tab', (u, t)),
            ('user_dur', (u, db_raw)), ('user_video', (u, v))
        ]:
            m, lc = lookup(stats[name], key, prior)
            nums.extend([m, lc])
        X[i, :6] = vals
        X[i, 6:] = nums
    return X, y, users


def sort_for_rank(X, y, users):
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    _, counts = np.unique(us, return_counts=True)
    return X[order], y[order], counts.tolist(), order


def train_and_predict(splits, target, seed=0):
    maps = make_maps(splits)
    stats, prior = build_stats(splits['train'])
    Xtr, ytr, utr = build_features(splits['train'], maps, stats, prior)
    Xva, yva, uva = build_features(splits['valid'], maps, stats, prior)
    Xtr_s, ytr_s, gtr, _ = sort_for_rank(Xtr, ytr, utr)
    Xva_s, yva_s, gva, ova = sort_for_rank(Xva, yva, uva)

    cat_cols = list(range(6))
    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=gtr, categorical_feature=cat_cols,
                         free_raw_data=True)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=gva, categorical_feature=cat_cols,
                         reference=dtrain, free_raw_data=True)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5],
        'learning_rate': 0.045,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 200,
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
    model = lgb.train(
        params, dtrain, num_boost_round=260,
        valid_sets=[dvalid], valid_names=['valid'],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(25)]
    )

    if target == 'valid':
        pred_s = model.predict(Xva_s, num_iteration=model.best_iteration)
        pred = np.empty_like(pred_s)
        pred[ova] = pred_s
        return pred
    else:
        Xt, yt, ut = build_features(splits[target], maps, stats, prior)
        return model.predict(Xt, num_iteration=model.best_iteration)


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
