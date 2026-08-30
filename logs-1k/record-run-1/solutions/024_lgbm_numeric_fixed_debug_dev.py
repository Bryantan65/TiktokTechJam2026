"""Numeric aggregate LightGBM with fixed boosting rounds.

Debugs node 23: early stopping on LightGBM's global AUC/logloss selected iteration
1, which is not aligned with within-user ranking.  Train the same numeric
aggregate features for a fixed number of rounds and output that model, so the
boosting signal is not discarded immediately.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402

ALPHAS = [8.0, 20.0, 30.0, 50.0, 25.0, 35.0, 50.0, 100.0, 100.0, 20.0, 35.0]


def build_aggregates(Xtr, ytr, specs):
    aggs = []
    for cols in specs:
        cnt = Counter()
        sm = defaultdict(float)
        for i in range(Xtr.shape[0]):
            if len(cols) == 1:
                key = int(Xtr[i, cols[0]])
            else:
                key = tuple(int(Xtr[i, c]) for c in cols)
            cnt[key] += 1
            sm[key] += float(ytr[i])
        aggs.append((cols, cnt, sm))
    return aggs


def make_features(X, y, aggs, prior, is_train):
    n = X.shape[0]
    cols_out = []
    for j, (cols, cnt, sm) in enumerate(aggs):
        alpha = ALPHAS[j] if j < len(ALPHAS) else 30.0
        cfeat = np.empty(n, dtype=np.float32)
        rfeat = np.empty(n, dtype=np.float32)
        liftfeat = np.empty(n, dtype=np.float32)
        for i in range(n):
            if len(cols) == 1:
                key = int(X[i, cols[0]])
            else:
                key = tuple(int(X[i, c]) for c in cols)
            c = cnt.get(key, 0)
            s = sm.get(key, 0.0)
            if is_train:
                c = max(c - 1, 0)
                s -= float(y[i])
            rate = (s + alpha * prior) / (c + alpha)
            cfeat[i] = np.log1p(c)
            rfeat[i] = rate
            liftfeat[i] = np.log((rate + 1e-5) / (prior + 1e-5))
        cols_out.extend([cfeat, rfeat, liftfeat])
    return np.vstack(cols_out).T.astype(np.float32)


def build_features(splits):
    enc, _ = encode(splits)
    Xtr, ytr, _ = enc['train']
    prior = float(np.mean(ytr))
    specs = [
        (0, 1), (0, 2), (0, 2, 3), (0, 3),
        (1,), (2,), (2, 3), (3,), (4,), (1, 3), (2, 4),
    ]
    aggs = build_aggregates(Xtr, ytr, specs)
    Xs, ys = {}, {}
    for sp, (X, y, u) in enc.items():
        Xs[sp] = make_features(X, y, aggs, prior, is_train=(sp == 'train'))
        ys[sp] = y.astype(np.float32)
    print(f"built numeric aggregate features: train={Xs['train'].shape}, prior={prior:.5f}")
    return Xs, ys


def train_predict(splits, target, seed):
    Xs, ys = build_features(splits)
    dtrain = lgb.Dataset(Xs['train'], label=ys['train'], free_raw_data=False)
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 5,
        'min_data_in_leaf': 1000,
        'feature_fraction': 0.95,
        'bagging_fraction': 0.90,
        'bagging_freq': 1,
        'lambda_l2': 30.0,
        'verbosity': -1,
        'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11,
        'bagging_seed': int(seed) + 17,
        'data_random_seed': int(seed) + 23,
        'num_threads': 4,
        'force_col_wise': True,
    }
    rounds = 180
    print(f"training fixed-round numeric LightGBM rows={Xs['train'].shape[0]} cols={Xs['train'].shape[1]} rounds={rounds}")
    booster = lgb.train(params, dtrain, num_boost_round=rounds,
                        valid_sets=[dtrain], valid_names=['train'],
                        callbacks=[lgb.log_evaluation(period=60)])
    return booster.predict(Xs[target], num_iteration=rounds)


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
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}+numeric_agg_lgb_fixed")
    preds = train_predict(splits, target, a.seed)
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
