"""Two exact LightGBM node-11 members plus one faster diverse member.

Node 20's two exact members with per-user z-score fusion are the current best.
This keeps those cached members unchanged and adds a cheaper LightGBM member
with smaller trees / higher learning rate, blended at 30% after per-user
z-normalisation.  If caches are absent it still trains standalone: two exact
members plus the fast member, which is intended to stay below the timeout unlike
three exact members.
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
            ntr = len(Xtr)
            nva = len(Xva)
            _, inv = np.unique(allv, return_inverse=True)
            a = inv[:ntr]
            b = inv[ntr:ntr + nva]
            c = b
        else:
            allv = np.concatenate([Xtr[:, j], Xva[:, j], Xt[:, j]])
            ntr = len(Xtr)
            nva = len(Xva)
            _, inv = np.unique(allv, return_inverse=True)
            a = inv[:ntr]
            b = inv[ntr:ntr + nva]
            c = inv[ntr + nva:]
        tr_cols.append(a.astype(np.int32, copy=False))
        va_cols.append(b.astype(np.int32, copy=False))
        t_cols.append(c.astype(np.int32, copy=False))
    return np.column_stack(tr_cols), np.column_stack(va_cols), np.column_stack(t_cols)


def build_feature_matrices(splits, target):
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


def fit_exact(Ftr, ytr, Fva, yva, Ft, member_seed, verbose=False):
    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=500,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=int(member_seed),
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
    )
    callbacks = [lgb.early_stopping(30, first_metric_only=True, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)],
            categorical_feature=list(range(5)), callbacks=callbacks)
    if verbose:
        print(f"exact_seed={member_seed} best_iteration={clf.best_iteration_}")
    return clf.predict(Ft, num_iteration=clf.best_iteration_, raw_score=True).astype(np.float32)


def fit_fast(Ftr, ytr, Fva, yva, Ft, member_seed, verbose=False):
    # A deliberately cheaper, slightly different tree ensemble.  It should be
    # weaker than an exact member but supplies another view without making the
    # script non-standalone under the 900s timeout.
    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        n_estimators=180,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=300,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=int(member_seed),
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
    )
    callbacks = [lgb.early_stopping(20, first_metric_only=True, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)],
            categorical_feature=list(range(5)), callbacks=callbacks)
    if verbose:
        print(f"fast_seed={member_seed} best_iteration={clf.best_iteration_}")
    return clf.predict(Ft, num_iteration=clf.best_iteration_, raw_score=True).astype(np.float32)


def per_user_weighted_z_fuse(exact_preds, fast_pred, users, fast_weight=0.30):
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
        mu = block.mean(axis=1, keepdims=True)
        sd = block.std(axis=1, keepdims=True)
        z_exact = ((block - mu) / np.maximum(sd, 1e-6)).mean(axis=0)
        fv = fast[idx]
        fz = (fv - fv.mean()) / max(float(fv.std()), 1e-6)
        out[idx] = (1.0 - fast_weight) * z_exact + fast_weight * fz.astype(np.float32)
    return out.astype(np.float64)


def train_members(splits, target, split_name, verbose=True):
    os.makedirs('pred_cache', exist_ok=True)
    n_target = len(splits[target])
    exact_seeds = [0, 2]
    exact_preds = [None] * len(exact_seeds)
    need_exact = []

    for i, ms in enumerate(exact_seeds):
        paths = [
            os.path.join('pred_cache', f'018_node11_raw_split{split_name}_mseed{ms}.npy'),
            os.path.join('pred_cache', f'019_node11_raw_split{split_name}_mseed{ms}.npy'),
            os.path.join('pred_cache', f'020_node11_raw_split{split_name}_mseed{ms}.npy'),
            os.path.join('pred_cache', f'021_node11_raw_split{split_name}_mseed{ms}.npy'),
            os.path.join('pred_cache', f'022_node11_raw_split{split_name}_mseed{ms}.npy'),
        ]
        loaded = False
        for path in paths:
            if os.path.isfile(path):
                arr = np.load(path)
                if arr.shape[0] == n_target:
                    exact_preds[i] = arr.astype(np.float32, copy=False)
                    loaded = True
                    if verbose:
                        print(f"loaded exact member {ms}: {path}")
                    break
        if not loaded:
            need_exact.append((i, ms, paths[-1]))

    fast_seed = 4
    fast_path = os.path.join('pred_cache', f'022_fast_lgbm_split{split_name}_mseed{fast_seed}.npy')
    fast_pred = None
    need_fast = True
    if os.path.isfile(fast_path):
        arr = np.load(fast_path)
        if arr.shape[0] == n_target:
            fast_pred = arr.astype(np.float32, copy=False)
            need_fast = False
            if verbose:
                print(f"loaded fast member {fast_seed}: {fast_path}")

    if need_exact or need_fast:
        t0 = time.time()
        Ftr, ytr, Fva, yva, Ft = build_feature_matrices(splits, target)
        if verbose:
            print(f"features train={Ftr.shape} valid={Fva.shape} target={Ft.shape} build {time.time() - t0:.1f}s")
            print(f"train pos={float(np.mean(ytr)):.4f} valid pos={float(np.mean(yva)):.4f}")
        for i, ms, path in need_exact:
            m0 = time.time()
            arr = fit_exact(Ftr, ytr, Fva, yva, Ft, ms, verbose=verbose)
            np.save(path, arr)
            exact_preds[i] = arr
            if verbose:
                print(f"trained exact member {ms} in {time.time() - m0:.1f}s -> {path}")
        if need_fast:
            m0 = time.time()
            fast_pred = fit_fast(Ftr, ytr, Fva, yva, Ft, fast_seed, verbose=verbose)
            np.save(fast_path, fast_pred)
            if verbose:
                print(f"trained fast member {fast_seed} in {time.time() - m0:.1f}s -> {fast_path}")
    return exact_preds, fast_pred


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

    exact_preds, fast_pred = train_members(splits, target=target, split_name=a.split, verbose=a.out is None)
    users = get_target_users(splits, target)
    scores = per_user_weighted_z_fuse(exact_preds, fast_pred, users, fast_weight=0.30)
    if a.out:
        np.save(a.out, scores)
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"wrote predictions in memory: {scores.shape} mean={float(np.mean(scores)):.6f}")
