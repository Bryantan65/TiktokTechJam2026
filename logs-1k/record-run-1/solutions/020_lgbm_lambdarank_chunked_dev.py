"""LightGBM LambdaRank with chunked same-user groups (debug of node 19).

Node 19 failed because LightGBM refuses query groups above 10k rows.  This keeps
the same aggregate/count encoded tree-ranker idea, but splits very active users
into consecutive same-user chunks below the query-size limit.
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
MAX_QUERY = 9000


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
    for i, v in enumerate(values):
        out[i] = mp.get(v, unk)
    return out


def make_groups(user_codes, max_query=MAX_QUERY):
    order = np.argsort(user_codes, kind='mergesort')
    sorted_u = user_codes[order]
    if len(sorted_u) == 0:
        return order, []
    cuts = np.flatnonzero(sorted_u[1:] != sorted_u[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(sorted_u)]
    groups = []
    n_split = 0
    for s, e in zip(starts, ends):
        rem = int(e - s)
        if rem > max_query:
            n_split += 1
        while rem > max_query:
            groups.append(int(max_query))
            rem -= max_query
        if rem > 0:
            groups.append(int(rem))
    if n_split:
        print(f"split {n_split} oversized user query groups at max_query={max_query}")
    return order, groups


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
    tab_tr, tab_map = compact_codes(Xtr[:, 3].astype(np.int64).tolist())
    dur_tr, dur_map = compact_codes(Xtr[:, 4].astype(np.int64).tolist())

    specs = [
        (1,), (2,), (3,), (4,),
        (0, 1), (0, 2), (0, 3), (0, 4),
        (2, 3), (1, 3), (2, 4),
    ]
    aggs = build_aggregates(Xtr, ytr, specs)

    out = {}
    users = {}
    yout = {}
    for sp, (X, y, u) in enc.items():
        if sp == 'train':
            uc, tc, dc = user_tr, tab_tr, dur_tr
        else:
            uc = apply_codes(X[:, 0].astype(np.int64).tolist(), user_map)
            tc = apply_codes(X[:, 3].astype(np.int64).tolist(), tab_map)
            dc = apply_codes(X[:, 4].astype(np.int64).tolist(), dur_map)
        base = [uc.astype(np.float32), tc.astype(np.float32), dc.astype(np.float32)]
        stats = add_stat_features(X, y, aggs, prior, is_train=(sp == 'train'))
        raw = [dc.astype(np.float32)]
        mat = np.vstack(base + raw + stats).T.astype(np.float32)
        out[sp] = mat
        users[sp] = uc.astype(np.int32)
        yout[sp] = y.astype(np.float32)
    print(f"built LightGBM features: {out['train'].shape[1]} columns from {FIELDS}; prior={prior:.4f}")
    return out, yout, users


def train_and_predict(splits, target, seed=0):
    Xs, ys, users = build_features(splits)
    tr_order, tr_group = make_groups(users['train'])
    va_order, va_group = make_groups(users['valid'])

    dtrain = lgb.Dataset(Xs['train'][tr_order], label=ys['train'][tr_order],
                         group=tr_group, categorical_feature=[0, 1, 2],
                         free_raw_data=False)
    dvalid = lgb.Dataset(Xs['valid'][va_order], label=ys['valid'][va_order],
                         group=va_group, categorical_feature=[0, 1, 2],
                         reference=dtrain, free_raw_data=False)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5],
        'label_gain': [0, 1],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'lambda_l2': 2.0,
        'min_gain_to_split': 0.0,
        'verbosity': -1,
        'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11,
        'bagging_seed': int(seed) + 17,
        'data_random_seed': int(seed) + 23,
        'num_threads': 4,
        'force_col_wise': True,
    }
    print(f"training LambdaRank groups: train={len(tr_group)} valid={len(va_group)} rows={len(tr_order)} max_group={max(max(tr_group), max(va_group))}")
    booster = lgb.train(
        params, dtrain, num_boost_round=180,
        valid_sets=[dvalid], valid_names=['valid'],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=50)]
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
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}+agg_lambdarank_chunked")
    preds = train_and_predict(splits, target=target, seed=a.seed)
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
