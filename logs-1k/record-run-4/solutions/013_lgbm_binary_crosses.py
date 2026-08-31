"""LightGBM binary classifier with a few categorical crosses.

Node 11 is a strong minimal pointwise GBDT on the official FM fields.  This
keeps its objective and hyperparameters, but adds explicit within-user
interaction categoricals (user-author/user-video/user-tab, plus item-tab
crosses) so LightGBM can model the same kind of feature interactions that FM
gets from embeddings, without reintroducing the broken target-stat pipeline.
"""
import argparse
import os
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, FIELDS          # noqa: E402


def duration_array(rows):
    return np.asarray([float(r[5]) for r in rows], dtype=np.float32)


def raw_categorical_columns(rows):
    # tuple: (date, user_id, video_id, author_id, tab, duration_ms, label)
    u = np.asarray([str(r[1]) for r in rows], dtype=object)
    v = np.asarray([str(r[2]) for r in rows], dtype=object)
    a = np.asarray([str(r[3]) for r in rows], dtype=object)
    tab = np.asarray([str(r[4]) for r in rows], dtype=object)
    dur = duration_array(rows)
    # Match starter-kit dur_bucket approximately enough for a categorical token;
    # include raw duration transforms separately below.
    db = np.asarray(np.minimum(9, np.floor(np.log1p(np.maximum(dur, 0.0)) / 1.5)).astype(np.int32).astype(str), dtype=object)

    cols = [u, v, a, tab, db]
    # Conservative set of crosses: user-specific preference, repeated exposure,
    # and tab-conditioned item/author effects.  All are categorical IDs only.
    cols += [
        np.char.add(np.char.add(u.astype(str), '|ua|'), a.astype(str)),
        np.char.add(np.char.add(u.astype(str), '|uv|'), v.astype(str)),
        np.char.add(np.char.add(u.astype(str), '|ut|'), tab.astype(str)),
        np.char.add(np.char.add(v.astype(str), '|vt|'), tab.astype(str)),
        np.char.add(np.char.add(a.astype(str), '|at|'), tab.astype(str)),
    ]
    return cols


def factorize_object_cols(train_cols, valid_cols, target_cols, target_is_valid):
    tr_out, va_out, te_out = [], [], []
    for c_tr, c_va, c_te in zip(train_cols, valid_cols, target_cols):
        if target_is_valid:
            allv = np.concatenate([c_tr, c_va])
            ntr = len(c_tr); nva = len(c_va)
            _, inv = np.unique(allv, return_inverse=True)
            a = inv[:ntr]
            b = inv[ntr:ntr + nva]
            c = b
        else:
            allv = np.concatenate([c_tr, c_va, c_te])
            ntr = len(c_tr); nva = len(c_va)
            _, inv = np.unique(allv, return_inverse=True)
            a = inv[:ntr]
            b = inv[ntr:ntr + nva]
            c = inv[ntr + nva:]
        tr_out.append(a.astype(np.int32, copy=False))
        va_out.append(b.astype(np.int32, copy=False))
        te_out.append(c.astype(np.int32, copy=False))
    return np.column_stack(tr_out), np.column_stack(va_out), np.column_stack(te_out)


def build_feature_matrices(splits, target):
    train_rows = splits['train']
    valid_rows = splits['valid']
    target_rows = splits['valid'] if target == 'valid' else splits[target]
    target_is_valid = (target == 'valid')

    ytr = np.asarray([int(r[6]) for r in train_rows], dtype=np.int32)
    yva = np.asarray([int(r[6]) for r in valid_rows], dtype=np.int32)

    tr_cols = raw_categorical_columns(train_rows)
    va_cols = raw_categorical_columns(valid_rows)
    te_cols = va_cols if target_is_valid else raw_categorical_columns(target_rows)
    Xtr_cat, Xva_cat, Xt_cat = factorize_object_cols(tr_cols, va_cols, te_cols, target_is_valid)

    dtr = duration_array(train_rows)
    dva = duration_array(valid_rows)
    dt = dva if target_is_valid else duration_array(target_rows)

    def finish(Xcat, dur):
        dur = dur.astype(np.float32)
        extra = np.column_stack([
            np.log1p(np.maximum(dur, 0.0)).astype(np.float32),
            (dur / 100000.0).astype(np.float32),
        ])
        return np.column_stack([Xcat.astype(np.float32), extra]).astype(np.float32, copy=False)

    return (finish(Xtr_cat, dtr), ytr,
            finish(Xva_cat, dva), yva,
            finish(Xt_cat, dt))


def train_classifier(splits, target, seed=0, verbose=True):
    t0 = time.time()
    Ftr, ytr, Fva, yva, Ft = build_feature_matrices(splits, target)
    if verbose:
        print(f"features train={Ftr.shape} valid={Fva.shape} target={Ft.shape} build {time.time() - t0:.1f}s")
        print(f"train pos={float(np.mean(ytr)):.4f} valid pos={float(np.mean(yva)):.4f}")

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
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
    )
    callbacks = [lgb.early_stopping(30, first_metric_only=True, verbose=verbose),
                 lgb.log_evaluation(50 if verbose else 0)]
    clf.fit(Ftr, ytr, eval_set=[(Fva, yva)],
            categorical_feature=list(range(10)), callbacks=callbacks)
    if verbose:
        print(f"best_iteration={clf.best_iteration_} total {time.time() - t0:.1f}s")
    return clf.predict_proba(Ft, num_iteration=clf.best_iteration_)[:, 1]


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

    scores = train_classifier(splits, target=target, seed=a.seed, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"wrote predictions in memory: {scores.shape} mean={float(np.mean(scores)):.6f}")
