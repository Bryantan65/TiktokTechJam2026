"""LightGBM binary classifier with simple time features.

Builds on node 14 (minimal categorical LightGBM, AUC early stopping) and adds
non-label time signals: date trend/day-of-week from data.load() tuples and hour
of day from the raw KuaiRand standard logs.  Raw long_view is never read.
"""
import argparse
import csv
import glob
import os
import sys
import time
import datetime as _dt

import numpy as np
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402


def duration_array(rows):
    return np.asarray([float(r[5]) for r in rows], dtype=np.float32)


def date_array(rows):
    return np.asarray([int(r[0]) for r in rows], dtype=np.int32)


def ordinal_and_dow(dates):
    uniq = np.unique(dates)
    ord_map = {}
    dow_map = {}
    for d in uniq:
        y = int(d) // 10000
        m = (int(d) // 100) % 100
        day = int(d) % 100
        try:
            dd = _dt.date(y, m, day)
            ord_map[int(d)] = dd.toordinal()
            dow_map[int(d)] = dd.weekday()
        except Exception:
            ord_map[int(d)] = int(d)
            dow_map[int(d)] = 0
    ords = np.asarray([ord_map[int(d)] for d in dates], dtype=np.int32)
    dows = np.asarray([dow_map[int(d)] for d in dates], dtype=np.int32)
    return ords, dows


def find_log_files(data_dir):
    p1 = glob.glob(os.path.join(data_dir, 'log_standard_4_08_to_4_21*_1k.csv'))
    p2 = glob.glob(os.path.join(data_dir, 'log_standard_4_22_to_5_08*_1k.csv'))
    if not p1:
        p1 = glob.glob(os.path.join(data_dir, 'log_standard_4_08_to_4_21*.csv'))
    if not p2:
        p2 = glob.glob(os.path.join(data_dir, 'log_standard_4_22_to_5_08*.csv'))
    return (sorted(p1)[:1] + sorted(p2)[:1])


def read_all_hours(data_dir, need_total):
    """Read hourmin from raw logs in the same file order as data.load()."""
    files = find_log_files(data_dir)
    if len(files) < 2:
        print('raw log files not found; hour features set to zero')
        return np.zeros(need_total, dtype=np.int16)
    hours = np.empty(need_total, dtype=np.int16)
    n = 0
    for path in files:
        if n >= need_total:
            break
        print('reading hourmin from', os.path.basename(path))
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                hidx = header.index('hourmin')
            except ValueError:
                print('hourmin column absent; hour features set to zero')
                return np.zeros(need_total, dtype=np.int16)
            for row in reader:
                if n >= need_total:
                    break
                try:
                    hm = int(row[hidx])
                    hr = hm // 100
                    if hr < 0 or hr > 23:
                        hr = 0
                except Exception:
                    hr = 0
                hours[n] = hr
                n += 1
    if n < need_total:
        print('raw logs shorter than expected; padding hour features with zero')
        hours[n:] = 0
    return hours


def split_hours(data_dir, splits):
    # The starter loader preserves row order: train, then valid, then test from
    # the two standard logs.  For devdata there may be no test; this still keeps
    # the same sequential convention and falls back safely if rows are absent.
    ntr = len(splits.get('train', []))
    nva = len(splits.get('valid', []))
    nte = len(splits.get('test', []))
    allh = read_all_hours(data_dir, ntr + nva + nte)
    out = {}
    s = 0
    for k, n in [('train', ntr), ('valid', nva), ('test', nte)]:
        if n:
            out[k] = allh[s:s+n]
        s += n
    return out


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


def build_feature_matrices(splits, target, data_dir):
    enc, _ = encode(splits)
    Xtr_base, ytr, _ = enc['train']
    Xva_base, yva, _ = enc['valid']
    Xt_base, _, _ = enc[target]
    target_is_valid = (target == 'valid')

    hours = split_hours(data_dir, splits)
    htr = hours.get('train', np.zeros(len(splits['train']), dtype=np.int16))
    hva = hours.get('valid', np.zeros(len(splits['valid']), dtype=np.int16))
    ht = hva if target_is_valid else hours.get(target, np.zeros(len(splits[target]), dtype=np.int16))

    dtr_raw = date_array(splits['train'])
    dva_raw = date_array(splits['valid'])
    dt_raw = dva_raw if target_is_valid else date_array(splits[target])
    all_dates = np.concatenate([dtr_raw, dva_raw, dt_raw]) if not target_is_valid else np.concatenate([dtr_raw, dva_raw])
    all_ord, all_dow = ordinal_and_dow(all_dates)
    ntr = len(dtr_raw)
    nva = len(dva_raw)
    otr = all_ord[:ntr]
    ova = all_ord[ntr:ntr+nva]
    dowtr = all_dow[:ntr]
    dowva = all_dow[ntr:ntr+nva]
    if target_is_valid:
        ot = ova
        dowt = dowva
    else:
        ot = all_ord[ntr+nva:]
        dowt = all_dow[ntr+nva:]
    min_ord = int(np.min(otr))

    # Categorical: original five fields plus hour-of-day and day-of-week.
    Xtr_cat_raw = np.column_stack([Xtr_base, htr, dowtr])
    Xva_cat_raw = np.column_stack([Xva_base, hva, dowva])
    Xt_cat_raw = np.column_stack([Xt_base, ht, dowt])
    Xtr_cat, Xva_cat, Xt_cat = factorize_cols(Xtr_cat_raw, Xva_cat_raw, Xt_cat_raw, target_is_valid)

    dtr = duration_array(splits['train'])
    dva = duration_array(splits['valid'])
    dt = dva if target_is_valid else duration_array(splits[target])

    def finish(Xcat, dur, ords, dows, hrs):
        dur = dur.astype(np.float32)
        day_idx = (ords.astype(np.float32) - float(min_ord))
        hour = hrs.astype(np.float32)
        dow = dows.astype(np.float32)
        extra = np.column_stack([
            np.log1p(np.maximum(dur, 0.0)).astype(np.float32),
            (dur / 100000.0).astype(np.float32),
            day_idx.astype(np.float32),
            np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32),
            np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32),
            np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32),
            np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32),
        ])
        return np.column_stack([Xcat.astype(np.float32), extra]).astype(np.float32, copy=False)

    return (finish(Xtr_cat, dtr, otr, dowtr, htr), ytr.astype(np.int32),
            finish(Xva_cat, dva, ova, dowva, hva), yva.astype(np.int32),
            finish(Xt_cat, dt, ot, dowt, ht), Xtr_cat.shape[1])


def train_classifier(splits, target, data_dir, seed=0, verbose=True):
    t0 = time.time()
    Ftr, ytr, Fva, yva, Ft, ncat = build_feature_matrices(splits, target, data_dir)
    if verbose:
        print(f"features train={Ftr.shape} valid={Fva.shape} target={Ft.shape} cat={ncat} build {time.time() - t0:.1f}s")
        print(f"train pos={float(np.mean(ytr)):.4f} valid pos={float(np.mean(yva)):.4f}")

    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        n_estimators=500,
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
            categorical_feature=list(range(ncat)), callbacks=callbacks)
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
