"""Add a leakage-safe historical target-stat LightGBM member to node 24.

The incumbent is the node-24 per-user z-score fusion of two exact LightGBM
members plus one fast member.  This script keeps those cached members unchanged
and trains one different member with K-fold out-of-fold target/count encodings on
training rows, then blends it at 30% after per-user z-normalization.
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


def factorize_cols(Xtr, Xva, Xt, target_is_valid):
    Xtr = Xtr.astype(np.int64, copy=False)
    Xva = Xva.astype(np.int64, copy=False)
    Xt = Xt.astype(np.int64, copy=False)
    tr_cols, va_cols, t_cols = [], [], []
    for j in range(Xtr.shape[1]):
        if target_is_valid:
            allv = np.concatenate([Xtr[:, j], Xva[:, j]])
            ntr, nva = len(Xtr), len(Xva)
            _, inv = np.unique(allv, return_inverse=True)
            a = inv[:ntr]
            b = inv[ntr:ntr + nva]
            c = b
        else:
            allv = np.concatenate([Xtr[:, j], Xva[:, j], Xt[:, j]])
            ntr, nva = len(Xtr), len(Xva)
            _, inv = np.unique(allv, return_inverse=True)
            a = inv[:ntr]
            b = inv[ntr:ntr + nva]
            c = inv[ntr + nva:]
        tr_cols.append(a.astype(np.int32, copy=False))
        va_cols.append(b.astype(np.int32, copy=False))
        t_cols.append(c.astype(np.int32, copy=False))
    return np.column_stack(tr_cols), np.column_stack(va_cols), np.column_stack(t_cols)


def make_cross(a, b):
    a = a.astype(np.int64, copy=False)
    b = b.astype(np.int64, copy=False)
    base = int(b.max()) + 1
    return a * base + b


def refactor_three(a, b, c):
    vals = np.concatenate([a, b, c]).astype(np.int64, copy=False)
    n1, n2 = len(a), len(b)
    _, inv = np.unique(vals, return_inverse=True)
    return (inv[:n1].astype(np.int32, copy=False),
            inv[n1:n1 + n2].astype(np.int32, copy=False),
            inv[n1 + n2:].astype(np.int32, copy=False))


def add_cross_columns(Xtr, Xva, Xt):
    # Base columns: user_id, video_id, author_id, tab, dur_bucket.
    crosses = []
    for p, q in [(1, 3), (2, 3), (0, 1), (0, 2)]:
        a, b, c = refactor_three(make_cross(Xtr[:, p], Xtr[:, q]),
                                 make_cross(Xva[:, p], Xva[:, q]),
                                 make_cross(Xt[:, p], Xt[:, q]))
        crosses.append((a, b, c))
    if not crosses:
        return Xtr, Xva, Xt
    tr = np.column_stack([x[0] for x in crosses])
    va = np.column_stack([x[1] for x in crosses])
    tt = np.column_stack([x[2] for x in crosses])
    return (np.column_stack([Xtr, tr]).astype(np.int32, copy=False),
            np.column_stack([Xva, va]).astype(np.int32, copy=False),
            np.column_stack([Xt, tt]).astype(np.int32, copy=False))


def oof_and_target_stat(train_codes, y, va_codes, t_codes, folds, prior, alpha=50.0):
    train_codes = train_codes.astype(np.int32, copy=False)
    va_codes = va_codes.astype(np.int32, copy=False)
    t_codes = t_codes.astype(np.int32, copy=False)
    y = y.astype(np.float32, copy=False)
    max_code = int(max(train_codes.max(), va_codes.max(), t_codes.max()))
    ncat = max_code + 1
    oof_mean = np.empty(len(train_codes), dtype=np.float32)
    oof_count = np.empty(len(train_codes), dtype=np.float32)
    for f in range(folds.max() + 1):
        m = folds != f
        cnt = np.bincount(train_codes[m], minlength=ncat).astype(np.float32)
        sm = np.bincount(train_codes[m], weights=y[m], minlength=ncat).astype(np.float32)
        idx = folds == f
        cc = cnt[train_codes[idx]]
        ss = sm[train_codes[idx]]
        oof_mean[idx] = (ss + prior * alpha) / (cc + alpha)
        oof_count[idx] = np.log1p(cc)
    cnt = np.bincount(train_codes, minlength=ncat).astype(np.float32)
    sm = np.bincount(train_codes, weights=y, minlength=ncat).astype(np.float32)
    def apply(codes):
        cc = cnt[codes]
        ss = sm[codes]
        mn = (ss + prior * alpha) / (cc + alpha)
        ct = np.log1p(cc)
        return mn.astype(np.float32), ct.astype(np.float32)
    va_mean, va_count = apply(va_codes)
    t_mean, t_count = apply(t_codes)
    return oof_mean, oof_count, va_mean, va_count, t_mean, t_count


def build_minimal_matrices(splits, target):
    enc, _ = encode(splits)
    Xtr_base, ytr, _ = enc['train']
    Xva_base, yva, _ = enc['valid']
    Xt_base, _, _ = enc[target]
    target_is_valid = (target == 'valid')
    Xtr_cat, Xva_cat, Xt_cat = factorize_cols(Xtr_base, Xva_base, Xt_base, target_is_valid)
    dtr = duration_array(splits['train'])
    dva = duration_array(splits['valid'])
    dt = dva if target_is_valid else duration_array(splits[target])
    def finish(Xcat, dur):
        dur = dur.astype(np.float32)
        extra = np.column_stack([
            np.log1p(np.maximum(dur, 0.0)).astype(np.float32),
            (dur / 100000.0).astype(np.float32),
        ])
        return np.column_stack([Xcat.astype(np.float32), extra]).astype(np.float32, copy=False)
    return (finish(Xtr_cat, dtr), ytr.astype(np.int32),
            finish(Xva_cat, dva), yva.astype(np.int32),
            finish(Xt_cat, dt))


def build_stat_matrices(splits, target, seed=2027, verbose=False):
    enc, _ = encode(splits)
    Xtr_base, ytr, _ = enc['train']
    Xva_base, yva, _ = enc['valid']
    Xt_base, _, _ = enc[target]
    target_is_valid = (target == 'valid')
    Xtr_cat, Xva_cat, Xt_cat = factorize_cols(Xtr_base, Xva_base, Xt_base, target_is_valid)
    Xtr_stat_cat, Xva_stat_cat, Xt_stat_cat = add_cross_columns(Xtr_cat, Xva_cat, Xt_cat)

    dtr = duration_array(splits['train'])
    dva = duration_array(splits['valid'])
    dt = dva if target_is_valid else duration_array(splits[target])
    ytr = ytr.astype(np.int32)
    yva = yva.astype(np.int32)
    prior = float(np.mean(ytr))
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, 5, size=len(ytr), dtype=np.int16)

    tr_feats = [Xtr_cat.astype(np.float32),
                np.log1p(np.maximum(dtr, 0.0)).astype(np.float32)[:, None],
                (dtr / 100000.0).astype(np.float32)[:, None]]
    va_feats = [Xva_cat.astype(np.float32),
                np.log1p(np.maximum(dva, 0.0)).astype(np.float32)[:, None],
                (dva / 100000.0).astype(np.float32)[:, None]]
    t_feats = [Xt_cat.astype(np.float32),
               np.log1p(np.maximum(dt, 0.0)).astype(np.float32)[:, None],
               (dt / 100000.0).astype(np.float32)[:, None]]

    # Target/count encodings for base fields plus four interaction fields.
    for j in range(Xtr_stat_cat.shape[1]):
        if verbose:
            print(f"  stat column {j+1}/{Xtr_stat_cat.shape[1]}")
        a, b, c, d, e, f = oof_and_target_stat(Xtr_stat_cat[:, j], ytr,
                                               Xva_stat_cat[:, j], Xt_stat_cat[:, j],
                                               folds, prior, alpha=50.0)
        tr_feats.extend([a[:, None], b[:, None]])
        va_feats.extend([c[:, None], d[:, None]])
        t_feats.extend([e[:, None], f[:, None]])

    Ftr = np.column_stack(tr_feats).astype(np.float32, copy=False)
    Fva = np.column_stack(va_feats).astype(np.float32, copy=False)
    Ft = np.column_stack(t_feats).astype(np.float32, copy=False)
    return Ftr, ytr, Fva, yva, Ft


def fit_exact(Ftr, ytr, Fva, yva, Ft, member_seed, verbose=False):
    clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', boosting_type='gbdt',
        n_estimators=350, learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=500, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=2.0,
        random_state=int(member_seed), n_jobs=-1, verbose=-1, force_col_wise=True)
    callbacks = [lgb.early_stopping(30, first_metric_only=True, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)], categorical_feature=list(range(5)), callbacks=callbacks)
    return clf.predict(Ft, num_iteration=clf.best_iteration_, raw_score=True).astype(np.float32)


def fit_fast(Ftr, ytr, Fva, yva, Ft, member_seed, verbose=False):
    clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', boosting_type='gbdt',
        n_estimators=180, learning_rate=0.08, num_leaves=31, max_depth=-1,
        min_child_samples=300, subsample=0.80, subsample_freq=1,
        colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=1.5,
        random_state=int(member_seed), n_jobs=-1, verbose=-1, force_col_wise=True)
    callbacks = [lgb.early_stopping(20, first_metric_only=True, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)], categorical_feature=list(range(5)), callbacks=callbacks)
    return clf.predict(Ft, num_iteration=clf.best_iteration_, raw_score=True).astype(np.float32)


def fit_stat(Ftr, ytr, Fva, yva, Ft, member_seed, verbose=False):
    clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', boosting_type='gbdt',
        n_estimators=260, learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=700, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=3.0,
        random_state=int(member_seed), n_jobs=-1, verbose=-1, force_col_wise=True)
    callbacks = [lgb.early_stopping(25, first_metric_only=True, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)], categorical_feature=list(range(5)), callbacks=callbacks)
    if verbose:
        print(f"stat best_iteration={clf.best_iteration_}")
    return clf.predict(Ft, num_iteration=clf.best_iteration_, raw_score=True).astype(np.float32)


def train_incumbent_members(splits, target, split_name, verbose=True):
    os.makedirs('pred_cache', exist_ok=True)
    n_target = len(splits[target])
    exact_seeds = [0, 2]
    exact_preds = [None] * len(exact_seeds)
    need_exact = []
    for i, ms in enumerate(exact_seeds):
        paths = [os.path.join('pred_cache', f'{p}_node11_raw_split{split_name}_mseed{ms}.npy')
                 for p in ['018', '019', '020', '021', '022', '023', '024', '025']]
        for path in paths:
            if os.path.isfile(path):
                arr = np.load(path)
                if arr.shape[0] == n_target:
                    exact_preds[i] = arr.astype(np.float32, copy=False)
                    if verbose:
                        print(f"loaded exact {ms}: {path}")
                    break
        if exact_preds[i] is None:
            need_exact.append((i, ms, paths[-1]))
    fast_seed = 4
    fast_paths = [os.path.join('pred_cache', f'{p}_fast_lgbm_split{split_name}_mseed{fast_seed}.npy')
                  for p in ['022', '023', '024', '025']]
    fast_pred = None
    for path in fast_paths:
        if os.path.isfile(path):
            arr = np.load(path)
            if arr.shape[0] == n_target:
                fast_pred = arr.astype(np.float32, copy=False)
                if verbose:
                    print(f"loaded fast {fast_seed}: {path}")
                break
    need_fast = fast_pred is None
    if need_exact or need_fast:
        Ftr, ytr, Fva, yva, Ft = build_minimal_matrices(splits, target)
        for i, ms, path in need_exact:
            arr = fit_exact(Ftr, ytr, Fva, yva, Ft, ms, verbose=verbose)
            np.save(path, arr)
            exact_preds[i] = arr
        if need_fast:
            fast_pred = fit_fast(Ftr, ytr, Fva, yva, Ft, fast_seed, verbose=verbose)
            np.save(fast_paths[-1], fast_pred)
    return exact_preds, fast_pred


def train_stat_member(splits, target, split_name, verbose=True):
    os.makedirs('pred_cache', exist_ok=True)
    n_target = len(splits[target])
    path = os.path.join('pred_cache', f'025_oof_target_stats_split{split_name}_mseed6.npy')
    if os.path.isfile(path):
        arr = np.load(path)
        if arr.shape[0] == n_target:
            if verbose:
                print(f"loaded stat member: {path}")
            return arr.astype(np.float32, copy=False)
    t0 = time.time()
    Ftr, ytr, Fva, yva, Ft = build_stat_matrices(splits, target, seed=2027, verbose=verbose)
    if verbose:
        print(f"stat features train={Ftr.shape} valid={Fva.shape} target={Ft.shape} built {time.time()-t0:.1f}s")
    arr = fit_stat(Ftr, ytr, Fva, yva, Ft, 6, verbose=verbose)
    np.save(path, arr)
    return arr


def node24_scores(exact_preds, fast_pred, users, fast_weight=0.23):
    exact = np.vstack([p.astype(np.float32, copy=False) for p in exact_preds])
    fast = fast_pred.astype(np.float32, copy=False)
    users = np.asarray(users)
    out = np.zeros(exact.shape[1], dtype=np.float32)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        block = exact[:, idx]
        z_exact = ((block - block.mean(axis=1, keepdims=True)) /
                   np.maximum(block.std(axis=1, keepdims=True), 1e-6)).mean(axis=0)
        fv = fast[idx]
        fz = (fv - fv.mean()) / max(float(fv.std()), 1e-6)
        out[idx] = (1.0 - fast_weight) * z_exact + fast_weight * fz.astype(np.float32)
    return out


def blend_with_stat(inc, stat, users, stat_weight=0.30):
    users = np.asarray(users)
    inc = inc.astype(np.float32, copy=False)
    stat = stat.astype(np.float32, copy=False)
    out = np.zeros(len(inc), dtype=np.float32)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        x = inc[idx]
        z1 = (x - x.mean()) / max(float(x.std()), 1e-6)
        s = stat[idx]
        z2 = (s - s.mean()) / max(float(s.std()), 1e-6)
        out[idx] = (1.0 - stat_weight) * z1 + stat_weight * z2
    return out.astype(np.float64)


def get_target_users(splits, target):
    enc, _ = encode(splits)
    return enc[target][2]


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
    exact_preds, fast_pred = train_incumbent_members(splits, target, a.split, verbose=a.out is None)
    stat_pred = train_stat_member(splits, target, a.split, verbose=a.out is None)
    users = get_target_users(splits, target)
    inc = node24_scores(exact_preds, fast_pred, users, fast_weight=0.23)
    scores = blend_with_stat(inc, stat_pred, users, stat_weight=0.30)
    if a.out:
        np.save(a.out, scores)
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"wrote predictions in memory: {scores.shape} mean={float(np.mean(scores)):.6f}")
