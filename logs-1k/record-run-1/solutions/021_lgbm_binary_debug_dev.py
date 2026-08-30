"""LightGBM binary classifier with the same aggregate features as node 20.

Debugs whether node 20's disastrous dev result is caused by the LambdaRank query
setup or by the feature construction / row alignment itself.  Same count and
leave-one-out target encodings, but no query groups and a standard binary loss.
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


ALPHA = 20.0


def compact_codes(values):
    mp = {}
    out = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        if v not in mp:
            mp[v] = len(mp)
        out[i] = mp[v]
    return out, mp


def apply_codes(values, mp):
    unk = len(mp)
    out = np.empty(len(values), dtype=np.int32)
    miss = 0
    for i, v in enumerate(values):
        c = mp.get(v, unk)
        if c == unk:
            miss += 1
        out[i] = c
    if miss:
        print(f"apply_codes: {miss}/{len(values)} unknown values mapped to {unk}")
    return out


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


def add_stat_features(X, y, aggs, prior, is_train):
    n = X.shape[0]
    feats = []
    for cols, cnt, sm in aggs:
        cfeat = np.empty(n, dtype=np.float32)
        rfeat = np.empty(n, dtype=np.float32)
        for i in range(n):
            if len(cols) == 1:
                key = int(X[i, cols[0]])
            else:
                key = tuple(int(X[i, c]) for c in cols)
            c = cnt.get(key, 0)
            s = sm.get(key, 0.0)
            if is_train:
                c = max(c - 1, 0)
                s = s - float(y[i])
            cfeat[i] = np.log1p(c)
            rfeat[i] = (s + ALPHA * prior) / (c + ALPHA)
        feats.extend([cfeat, rfeat])
    return feats


def build_features(splits):
    enc, _ = encode(splits)
    Xtr, ytr, _ = enc['train']
    prior = float(np.mean(ytr))

    user_tr, user_map = compact_codes(Xtr[:, 0].astype(np.int64).tolist())
    video_tr, video_map = compact_codes(Xtr[:, 1].astype(np.int64).tolist())
    author_tr, author_map = compact_codes(Xtr[:, 2].astype(np.int64).tolist())
    tab_tr, tab_map = compact_codes(Xtr[:, 3].astype(np.int64).tolist())
    dur_tr, dur_map = compact_codes(Xtr[:, 4].astype(np.int64).tolist())

    specs = [
        (1,), (2,), (3,), (4,),
        (0, 1), (0, 2), (0, 3), (0, 4),
        (2, 3), (1, 3), (2, 4),
    ]
    aggs = build_aggregates(Xtr, ytr, specs)

    out = {}
    yout = {}
    for sp, (X, y, u) in enc.items():
        if sp == 'train':
            uc, vc, ac, tc, dc = user_tr, video_tr, author_tr, tab_tr, dur_tr
        else:
            uc = apply_codes(X[:, 0].astype(np.int64).tolist(), user_map)
            vc = apply_codes(X[:, 1].astype(np.int64).tolist(), video_map)
            ac = apply_codes(X[:, 2].astype(np.int64).tolist(), author_map)
            tc = apply_codes(X[:, 3].astype(np.int64).tolist(), tab_map)
            dc = apply_codes(X[:, 4].astype(np.int64).tolist(), dur_map)
        # Give the tree the same raw categorical ids as the FM, plus aggregate stats.
        base = [uc.astype(np.float32), vc.astype(np.float32), ac.astype(np.float32),
                tc.astype(np.float32), dc.astype(np.float32)]
        stats = add_stat_features(X, y, aggs, prior, is_train=(sp == 'train'))
        mat = np.vstack(base + stats).T.astype(np.float32)
        out[sp] = mat
        yout[sp] = y.astype(np.float32)
    print(f"built LightGBM binary features: {out['train'].shape[1]} columns from {FIELDS}; prior={prior:.4f}")
    return out, yout


def train_and_predict(splits, target, seed=0):
    Xs, ys = build_features(splits)
    dtrain = lgb.Dataset(Xs['train'], label=ys['train'],
                         categorical_feature=[0, 1, 2, 3, 4],
                         free_raw_data=False)
    dvalid = lgb.Dataset(Xs['valid'], label=ys['valid'],
                         categorical_feature=[0, 1, 2, 3, 4],
                         reference=dtrain, free_raw_data=False)
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'lambda_l2': 5.0,
        'verbosity': -1,
        'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11,
        'bagging_seed': int(seed) + 17,
        'data_random_seed': int(seed) + 23,
        'num_threads': 4,
        'force_col_wise': True,
    }
    print(f"training binary LightGBM rows={Xs['train'].shape[0]} cols={Xs['train'].shape[1]}")
    booster = lgb.train(
        params, dtrain, num_boost_round=220,
        valid_sets=[dvalid], valid_names=['valid'],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(period=50)]
    )
    print(f"best_iteration={booster.best_iteration}")
    return booster.predict(Xs[target], num_iteration=booster.best_iteration)


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
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}+agg_lgb_binary")
    preds = train_and_predict(splits, target=target, seed=a.seed)
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
