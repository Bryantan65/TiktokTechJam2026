"""LightGBM LambdaMART ranker with train-only historical statistics.

Debug of node 4: encode() can emit -1 for missing/unseen categorical values, but
np.bincount and LightGBM categorical features require non-negative integer codes.
Shift all encoded fields by +1 before computing counts/crosses so missing becomes
category 0.
"""
import argparse
import os
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402


def duration_array(rows):
    return np.asarray([float(r[5]) for r in rows], dtype=np.float32)


def shift_nonneg(X):
    X = X.astype(np.int64, copy=False)
    return X + 1  # encode() may use -1 for missing/unseen; make it category 0


def add_crosses(X, n_tab):
    X = shift_nonneg(X)
    vt = X[:, 1] * n_tab + X[:, 3]
    at = X[:, 2] * n_tab + X[:, 3]
    ut = X[:, 0] * n_tab + X[:, 3]
    return np.column_stack([X, vt, at, ut]).astype(np.int32, copy=False)


def stat_features(train_codes, train_y, codes, is_train, prior, m=50.0):
    train_codes = train_codes.astype(np.int64, copy=False)
    codes = codes.astype(np.int64, copy=False)
    max_code = int(max(train_codes.max(initial=0), codes.max(initial=0)))
    cnt = np.bincount(train_codes, minlength=max_code + 1).astype(np.float32)
    sm = np.bincount(train_codes, weights=train_y.astype(np.float32),
                     minlength=max_code + 1).astype(np.float32)
    c = cnt[codes]
    s = sm[codes]
    if is_train:
        yy = train_y.astype(np.float32)
        c_eff = np.maximum(c - 1.0, 0.0)
        mean = (s - yy + prior * m) / (c_eff + m)
        count = np.log1p(c_eff)
    else:
        mean = (s + prior * m) / (c + m)
        count = np.log1p(c)
    return mean.astype(np.float32), count.astype(np.float32)


def build_feature_matrices(splits, target):
    enc, _ = encode(splits)
    Xtr_base, ytr, _ = enc['train']
    Xva_base, yva, _ = enc['valid']
    Xt_base, _, _ = enc[target]

    tabs = np.concatenate([shift_nonneg(Xtr_base)[:, 3], shift_nonneg(Xva_base)[:, 3], shift_nonneg(Xt_base)[:, 3]])
    n_tab = int(tabs.max(initial=0) + 1)
    Xtr_cat = add_crosses(Xtr_base, n_tab)
    Xva_cat = add_crosses(Xva_base, n_tab)
    Xt_cat = Xva_cat if target == 'valid' else add_crosses(Xt_base, n_tab)

    prior = float(np.mean(ytr))
    stat_cols_tr = []
    stat_cols_va = []
    stat_cols_t = []
    for j in range(Xtr_cat.shape[1]):
        mn, ct = stat_features(Xtr_cat[:, j], ytr, Xtr_cat[:, j], True, prior)
        stat_cols_tr.extend([mn, ct])
        mn, ct = stat_features(Xtr_cat[:, j], ytr, Xva_cat[:, j], False, prior)
        stat_cols_va.extend([mn, ct])
        if target != 'valid':
            mn, ct = stat_features(Xtr_cat[:, j], ytr, Xt_cat[:, j], False, prior)
            stat_cols_t.extend([mn, ct])
    if target == 'valid':
        stat_cols_t = stat_cols_va

    dtr = duration_array(splits['train'])
    dva = duration_array(splits['valid'])
    dt = dva if target == 'valid' else duration_array(splits[target])

    def finish(Xcat, stat_cols, dur):
        dur = dur.astype(np.float32)
        extra = [np.log1p(np.maximum(dur, 0.0)).astype(np.float32),
                 (dur / 100000.0).astype(np.float32)]
        return np.column_stack([Xcat] + stat_cols + extra).astype(np.float32, copy=False)

    Ftr = finish(Xtr_cat, stat_cols_tr, dtr)
    Fva = finish(Xva_cat, stat_cols_va, dva)
    Ft = Fva if target == 'valid' else finish(Xt_cat, stat_cols_t, dt)
    return enc, Ftr, ytr.astype(np.int32), Fva, yva.astype(np.int32), Ft


def sort_by_user(F, y, users):
    order = np.argsort(users, kind='stable')
    us = users[order]
    _, counts = np.unique(us, return_counts=True)
    return F[order], y[order], counts.tolist()


def train_ranker(splits, target, seed=0, verbose=True):
    t0 = time.time()
    enc, Ftr, ytr, Fva, yva, Ft = build_feature_matrices(splits, target)
    _, _, utr = enc['train']
    _, _, uva = enc['valid']
    Ftr_s, ytr_s, gtr = sort_by_user(Ftr, ytr, utr)
    Fva_s, yva_s, gva = sort_by_user(Fva, yva, uva)

    if verbose:
        print(f"features train={Ftr_s.shape} valid={Fva_s.shape} target={Ft.shape} build {time.time()-t0:.1f}s")

    cat_idx = list(range(8))
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        eval_at=[5],
        label_gain=[0, 1],
        boosting_type='gbdt',
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
    )
    callbacks = [lgb.early_stopping(30, first_metric_only=True, verbose=verbose)]
    callbacks.append(lgb.log_evaluation(20 if verbose else 0))
    ranker.fit(Ftr_s, ytr_s, group=gtr,
               eval_set=[(Fva_s, yva_s)], eval_group=[gva], eval_at=[5],
               categorical_feature=cat_idx, callbacks=callbacks)
    if verbose:
        print(f"best_iteration={ranker.best_iteration_} total {time.time()-t0:.1f}s")
    return ranker.predict(Ft, num_iteration=ranker.best_iteration_)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    np.random.seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")

    scores = train_ranker(splits, target=target, seed=a.seed, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"wrote predictions in memory: {scores.shape} mean={float(np.mean(scores)):.6f}")
