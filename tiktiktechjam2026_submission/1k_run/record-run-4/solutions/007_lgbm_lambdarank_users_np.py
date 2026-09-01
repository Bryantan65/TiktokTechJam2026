"""LightGBM LambdaMART ranker with train-only historical statistics.

Debug of node 6: encode() returns users as a Python sequence, so indexing it by a
NumPy permutation fails.  Convert users to np.asarray before sorting into
LightGBM query groups.  The rest keeps node 6's dense non-negative categorical
factorization.
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


def factorize_three(train_col, valid_col, target_col=None):
    if target_col is None:
        allv = np.concatenate([train_col, valid_col]).astype(np.int64, copy=False)
        ntr = len(train_col)
        nva = len(valid_col)
        _, inv = np.unique(allv, return_inverse=True)
        return (inv[:ntr].astype(np.int32, copy=False),
                inv[ntr:ntr + nva].astype(np.int32, copy=False),
                None)
    allv = np.concatenate([train_col, valid_col, target_col]).astype(np.int64, copy=False)
    ntr = len(train_col)
    nva = len(valid_col)
    _, inv = np.unique(allv, return_inverse=True)
    return (inv[:ntr].astype(np.int32, copy=False),
            inv[ntr:ntr + nva].astype(np.int32, copy=False),
            inv[ntr + nva:].astype(np.int32, copy=False))


def make_cats(Xtr, Xva, Xt, target_is_valid):
    Xtr = Xtr.astype(np.int64, copy=False)
    Xva = Xva.astype(np.int64, copy=False)
    Xt = Xt.astype(np.int64, copy=False)
    tr_cols, va_cols, t_cols = [], [], []
    for j in range(Xtr.shape[1]):
        if target_is_valid:
            a, b, _ = factorize_three(Xtr[:, j], Xva[:, j], None)
            c = b
        else:
            a, b, c = factorize_three(Xtr[:, j], Xva[:, j], Xt[:, j])
        tr_cols.append(a); va_cols.append(b); t_cols.append(c)

    # Cross features aligned by factorizing original pair values across splits.
    for left, right in [(1, 3), (2, 3), (0, 3)]:  # video_tab, author_tab, user_tab
        if target_is_valid:
            all_pairs = np.vstack([
                np.column_stack([Xtr[:, left], Xtr[:, right]]),
                np.column_stack([Xva[:, left], Xva[:, right]])
            ])
            ntr = len(Xtr); nva = len(Xva)
            _, inv = np.unique(all_pairs, axis=0, return_inverse=True)
            a = inv[:ntr].astype(np.int32, copy=False)
            b = inv[ntr:ntr + nva].astype(np.int32, copy=False)
            c = b
        else:
            all_pairs = np.vstack([
                np.column_stack([Xtr[:, left], Xtr[:, right]]),
                np.column_stack([Xva[:, left], Xva[:, right]]),
                np.column_stack([Xt[:, left], Xt[:, right]])
            ])
            ntr = len(Xtr); nva = len(Xva)
            _, inv = np.unique(all_pairs, axis=0, return_inverse=True)
            a = inv[:ntr].astype(np.int32, copy=False)
            b = inv[ntr:ntr + nva].astype(np.int32, copy=False)
            c = inv[ntr + nva:].astype(np.int32, copy=False)
        tr_cols.append(a); va_cols.append(b); t_cols.append(c)

    return (np.column_stack(tr_cols).astype(np.int32, copy=False),
            np.column_stack(va_cols).astype(np.int32, copy=False),
            np.column_stack(t_cols).astype(np.int32, copy=False))


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

    target_is_valid = (target == 'valid')
    Xtr_cat, Xva_cat, Xt_cat = make_cats(Xtr_base, Xva_base, Xt_base, target_is_valid)

    prior = float(np.mean(ytr))
    stat_cols_tr, stat_cols_va, stat_cols_t = [], [], []
    for j in range(Xtr_cat.shape[1]):
        mn, ct = stat_features(Xtr_cat[:, j], ytr, Xtr_cat[:, j], True, prior)
        stat_cols_tr.extend([mn, ct])
        mn, ct = stat_features(Xtr_cat[:, j], ytr, Xva_cat[:, j], False, prior)
        stat_cols_va.extend([mn, ct])
        if target_is_valid:
            stat_cols_t = stat_cols_va
        else:
            mn, ct = stat_features(Xtr_cat[:, j], ytr, Xt_cat[:, j], False, prior)
            stat_cols_t.extend([mn, ct])

    dtr = duration_array(splits['train'])
    dva = duration_array(splits['valid'])
    dt = dva if target_is_valid else duration_array(splits[target])

    def finish(Xcat, stat_cols, dur):
        dur = dur.astype(np.float32)
        extra = [np.log1p(np.maximum(dur, 0.0)).astype(np.float32),
                 (dur / 100000.0).astype(np.float32)]
        return np.column_stack([Xcat] + stat_cols + extra).astype(np.float32, copy=False)

    Ftr = finish(Xtr_cat, stat_cols_tr, dtr)
    Fva = finish(Xva_cat, stat_cols_va, dva)
    Ft = Fva if target_is_valid else finish(Xt_cat, stat_cols_t, dt)
    return enc, Ftr, ytr.astype(np.int32), Fva, yva.astype(np.int32), Ft


def sort_by_user(F, y, users):
    users = np.asarray(users)
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

    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        eval_at=[5],
        label_gain=[0, 1],
        boosting_type='gbdt',
        n_estimators=220,
        learning_rate=0.06,
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
    callbacks = [lgb.early_stopping(25, first_metric_only=True, verbose=verbose),
                 lgb.log_evaluation(20 if verbose else 0)]
    ranker.fit(Ftr_s, ytr_s, group=gtr,
               eval_set=[(Fva_s, yva_s)], eval_group=[gva], eval_at=[5],
               categorical_feature=list(range(8)), callbacks=callbacks)
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
