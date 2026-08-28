"""Draft LambdaMART/LambdaRank with history features.

Standalone LightGBM ranker grouped by user.  It uses the tuple fields as native
categoricals plus the leak-free history statistics from the best FM branch, then
trains a fixed-round LambdaRank model and writes split predictions in original
row order.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402


class HistState:
    def __init__(self):
        self.ui = defaultdict(int)
        self.up = defaultdict(int)
        self.uai = defaultdict(int)
        self.uap = defaultdict(int)
        self.uvi = defaultdict(int)
        self.uvp = defaultdict(int)
        self.uti = defaultdict(int)
        self.utp = defaultdict(int)
        self.udi = defaultdict(int)
        self.udp = defaultdict(int)
        self.ai = defaultdict(int)
        self.ap = defaultdict(int)
        self.vi = defaultdict(int)
        self.vp = defaultdict(int)

    def copy(self):
        other = HistState()
        for name in ('ui', 'up', 'uai', 'uap', 'uvi', 'uvp', 'uti', 'utp',
                     'udi', 'udp', 'ai', 'ap', 'vi', 'vp'):
            setattr(other, name, defaultdict(int, getattr(self, name).copy()))
        return other

    def features_one(self, row):
        u, v, a, tab = row[1], row[2], row[3], row[4]
        dur = int(row[5]) // 10000
        ui, up = self.ui[u], self.up[u]
        uai, uap = self.uai[(u, a)], self.uap[(u, a)]
        uvi, uvp = self.uvi[(u, v)], self.uvp[(u, v)]
        uti, utp = self.uti[(u, tab)], self.utp[(u, tab)]
        udi, udp = self.udi[(u, dur)], self.udp[(u, dur)]
        ai, ap = self.ai[a], self.ap[a]
        vi, vp = self.vi[v], self.vp[v]
        return [
            np.log1p(ui), (up + 1.0) / (ui + 2.0),
            np.log1p(uai), (uap + 0.5) / (uai + 2.0), uap / (up + 1.0),
            np.log1p(uvi), (uvp + 0.5) / (uvi + 2.0),
            np.log1p(uti), (utp + 0.5) / (uti + 2.0),
            np.log1p(udi), (udp + 0.5) / (udi + 2.0),
            np.log1p(ai), (ap + 1.0) / (ai + 2.0),
            np.log1p(vi), (vp + 0.5) / (vi + 2.0),
        ]

    def update(self, row):
        u, v, a, tab = row[1], row[2], row[3], row[4]
        dur = int(row[5]) // 10000
        y = 1 if row[6] > 0 else 0
        self.ui[u] += 1; self.up[u] += y
        self.uai[(u, a)] += 1; self.uap[(u, a)] += y
        self.uvi[(u, v)] += 1; self.uvp[(u, v)] += y
        self.uti[(u, tab)] += 1; self.utp[(u, tab)] += y
        self.udi[(u, dur)] += 1; self.udp[(u, dur)] += y
        self.ai[a] += 1; self.ap[a] += y
        self.vi[v] += 1; self.vp[v] += y


def make_history_features(splits):
    state = HistState()
    feats = {}
    tr = np.empty((len(splits['train']), 15), dtype=np.float32)
    for i, row in enumerate(splits['train']):
        tr[i] = state.features_one(row)
        state.update(row)
    feats['train'] = tr
    train_state = state.copy()
    for sp in ('valid', 'test'):
        st = train_state.copy()
        arr = np.empty((len(splits[sp]), 15), dtype=np.float32)
        for i, row in enumerate(splits[sp]):
            arr[i] = st.features_one(row)
        feats[sp] = arr
    mu = feats['train'].mean(axis=0)
    sd = feats['train'].std(axis=0) + 1e-6
    for sp in feats:
        feats[sp] = (feats[sp] - mu) / sd
    return feats


def group_by_user(X, y, users):
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    su = users[order]
    _, counts = np.unique(su, return_counts=True)
    return X[order], y[order], counts.tolist(), order


def make_features(splits):
    enc, _ = encode(splits)
    hist = make_history_features(splits)
    out = {}
    for sp in ('train', 'valid', 'test'):
        Xcat, y, users = enc[sp]
        X = np.concatenate([Xcat.astype(np.float32), hist[sp].astype(np.float32)], axis=1)
        out[sp] = (X, np.asarray(y, dtype=np.int32), np.asarray(users))
    return out


def train_ranker(feats, seed):
    Xtr, ytr, utr = feats['train']
    Xs, ys, group, order = group_by_user(Xtr, ytr, utr)
    dtrain = lgb.Dataset(
        Xs, label=ys, group=group,
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=False,
    )
    params = {
        'objective': 'lambdarank',
        'metric': 'None',
        'label_gain': [0, 1],
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 63,
        'min_data_in_leaf': 80,
        'min_data_per_group': 1,
        'max_depth': -1,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'lambda_l1': 0.0,
        'lambda_l2': 2.0,
        'max_cat_threshold': 64,
        'cat_l2': 20.0,
        'cat_smooth': 20.0,
        'verbosity': -1,
        'num_threads': 0,
        'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11,
        'bagging_seed': int(seed) + 17,
        'data_random_seed': int(seed) + 23,
        'force_col_wise': True,
    }
    return lgb.train(params, dtrain, num_boost_round=260)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}+hist; lgbm_lambdarank")
    feats = make_features(splits)
    model = train_ranker(feats, a.seed)
    X, y, users = feats[a.split]
    scores = model.predict(X, num_iteration=model.best_iteration).astype(np.float64)
    if a.out:
        np.save(a.out, scores)
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(scores[:10])
