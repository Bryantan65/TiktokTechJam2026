"""Minimal LightGBM with dwell/play-time sample weights for training.

Builds on node 11: same features and logloss early stopping, but training rows
are weighted by raw play_time_ms (read only as an auxiliary training signal, not
as a prediction feature).  Raw long_view is never read.
"""
import argparse
import csv
import glob
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


def find_log_files(data_dir):
    # Correct KuaiRand-1K files end in _1k; keep a fallback for portability.
    p1 = glob.glob(os.path.join(data_dir, 'log_standard_4_08_to_4_21*_1k.csv'))
    p2 = glob.glob(os.path.join(data_dir, 'log_standard_4_22_to_5_08*_1k.csv'))
    if not p1:
        p1 = glob.glob(os.path.join(data_dir, 'log_standard_4_08_to_4_21*.csv'))
    if not p2:
        p2 = glob.glob(os.path.join(data_dir, 'log_standard_4_22_to_5_08*.csv'))
    return sorted(p1)[:1] + sorted(p2)[:1]


def read_play_times(data_dir, need_total):
    """Read play_time_ms in data.load row order; never read raw labels."""
    files = find_log_files(data_dir)
    vals = np.zeros(need_total, dtype=np.float32)
    if len(files) < 2:
        print('raw log files not found; using unit sample weights')
        return vals
    n = 0
    for path in files:
        if n >= need_total:
            break
        print('reading play_time_ms from', os.path.basename(path))
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                pidx = header.index('play_time_ms')
            except ValueError:
                print('play_time_ms column absent; using unit sample weights')
                return np.zeros(need_total, dtype=np.float32)
            for row in reader:
                if n >= need_total:
                    break
                try:
                    v = float(row[pidx])
                    if not np.isfinite(v) or v < 0:
                        v = 0.0
                except Exception:
                    v = 0.0
                vals[n] = v
                n += 1
    if n < need_total:
        print('raw logs shorter than expected; padding play_time_ms with zero')
    return vals


def train_play_times(data_dir, splits):
    ntr = len(splits.get('train', []))
    nva = len(splits.get('valid', []))
    nte = len(splits.get('test', []))
    allp = read_play_times(data_dir, ntr + nva + nte)
    return allp[:ntr]


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


def make_sample_weight(ytr, play_ms):
    y = ytr.astype(np.int32)
    pt = np.asarray(play_ms, dtype=np.float32)
    w = np.ones(len(y), dtype=np.float32)
    pos = (y == 1)
    if np.any(pos):
        lp = np.log1p(np.maximum(pt[pos], 0.0))
        scale = np.percentile(lp, 95) if lp.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        strength = np.clip(lp / scale, 0.0, 1.0)
        # Moderate dwell-time emphasis: positives range from 1x to 3x.
        w[pos] = 1.0 + 2.0 * strength
    # Keep average Hessian scale close to the unweighted parent.
    w *= (float(len(w)) / float(np.sum(w)))
    return w


def train_classifier(splits, target, data_dir, seed=0, verbose=True):
    t0 = time.time()
    Ftr, ytr, Fva, yva, Ft = build_feature_matrices(splits, target)
    play_tr = train_play_times(data_dir, splits)
    sw = make_sample_weight(ytr, play_tr)
    if verbose:
        print(f"features train={Ftr.shape} valid={Fva.shape} target={Ft.shape} build {time.time() - t0:.1f}s")
        print(f"train pos={float(np.mean(ytr)):.4f} valid pos={float(np.mean(yva)):.4f}")
        print(f"sample_weight mean={float(np.mean(sw)):.4f} min={float(np.min(sw)):.4f} max={float(np.max(sw)):.4f}")

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
    clf.fit(Ftr, ytr, sample_weight=sw, eval_set=[(Fva, yva)],
            categorical_feature=list(range(5)), callbacks=callbacks)
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

    scores = train_classifier(splits, target=target, data_dir=a.data_dir, seed=a.seed, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"wrote predictions in memory: {scores.shape} mean={float(np.mean(scores)):.6f}")
