"""LightGBM LambdaRank with historical train-label statistics.

Debug of 003: LightGBM refuses query groups longer than 10000 rows.  Some
KuaiRand users exceed that, so this version keeps rows sorted by user but splits
oversized users into <=9000-row chunks for the ranker/eval groups.  It also uses
a smaller tree budget for the dev screen so the feature/ranker direction can be
validated before spending a full valid run.
"""
import argparse
import os
import sys
import time

import numpy as np
from lightgbm import LGBMRanker, early_stopping, log_evaluation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402


def rows_arrays(rows):
    date = np.asarray([int(r[0]) for r in rows], dtype=np.int32)
    dur = np.asarray([float(r[5]) for r in rows], dtype=np.float32)
    return date, dur


def add_1d_stat(feats, ktr, ytr, ktg, global_mean, alpha=20.0, train=False):
    n = int(max(ktr.max(initial=0), ktg.max(initial=0))) + 1
    cnt = np.bincount(ktr, minlength=n).astype(np.float32)
    sm = np.bincount(ktr, weights=ytr, minlength=n).astype(np.float32)
    if train:
        c = cnt[ktr] - 1.0
        s = sm[ktr] - ytr
    else:
        c = cnt[ktg]
        s = sm[ktg]
    rate = (s + alpha * global_mean) / (c + alpha)
    feats.append(np.log1p(np.maximum(c, 0.0)).astype(np.float32))
    feats.append(rate.astype(np.float32))


def pair_keys(a, b, nb):
    return a.astype(np.int64) * np.int64(nb) + b.astype(np.int64)


def add_pair_stat(feats, atr, btr, ytr, atg, btg, global_mean, nb,
                  alpha=20.0, train=False):
    ktr = pair_keys(atr, btr, nb)
    uniq, inv = np.unique(ktr, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float32)
    sm = np.bincount(inv, weights=ytr).astype(np.float32)
    if train:
        c = cnt[inv] - 1.0
        s = sm[inv] - ytr
    else:
        ktg = pair_keys(atg, btg, nb)
        pos = np.searchsorted(uniq, ktg)
        ok = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == ktg)
        c = np.zeros(len(ktg), dtype=np.float32)
        s = np.zeros(len(ktg), dtype=np.float32)
        if ok.any():
            c[ok] = cnt[pos[ok]]
            s[ok] = sm[pos[ok]]
    rate = (s + alpha * global_mean) / (c + alpha)
    feats.append(np.log1p(np.maximum(c, 0.0)).astype(np.float32))
    feats.append(rate.astype(np.float32))


def build_features(enc, splits, split_name):
    Xtr, ytr, utr = enc['train']
    Xtg, ytg, utg = enc[split_name]
    ytr = ytr.astype(np.float32)
    gm = float(ytr.mean())

    dtr, durtr = rows_arrays(splits['train'])
    dtg, durtg = rows_arrays(splits[split_name])
    all_dates = {d: i for i, d in enumerate(sorted(set(dtr.tolist()) | set(dtg.tolist())))}
    dtrc = np.asarray([all_dates[int(d)] for d in dtr], dtype=np.int32)
    dtgc = np.asarray([all_dates[int(d)] for d in dtg], dtype=np.int32)

    def base(X, dc, dur):
        return [X[:, 0].astype(np.float32),
                X[:, 1].astype(np.float32),
                X[:, 2].astype(np.float32),
                X[:, 3].astype(np.float32),
                X[:, 4].astype(np.float32),
                dc.astype(np.float32),
                np.log1p(dur).astype(np.float32)]

    feats = base(Xtg, dtgc, durtg)
    is_train = (split_name == 'train')

    for col in [1, 2, 3, 4]:
        add_1d_stat(feats, Xtr[:, col].astype(np.int64), ytr,
                    Xtg[:, col].astype(np.int64), gm, train=is_train)
    add_1d_stat(feats, dtrc.astype(np.int64), ytr, dtgc.astype(np.int64), gm,
                alpha=100.0, train=is_train)

    maxv = int(max(Xtr[:, 1].max(), Xtg[:, 1].max())) + 1
    maxa = int(max(Xtr[:, 2].max(), Xtg[:, 2].max())) + 1
    maxt = int(max(Xtr[:, 3].max(), Xtg[:, 3].max())) + 1
    maxd = int(max(Xtr[:, 4].max(), Xtg[:, 4].max())) + 1
    pairs = [
        (Xtr[:, 0], Xtr[:, 1], Xtg[:, 0], Xtg[:, 1], maxv, 10.0),
        (Xtr[:, 0], Xtr[:, 2], Xtg[:, 0], Xtg[:, 2], maxa, 20.0),
        (Xtr[:, 0], Xtr[:, 3], Xtg[:, 0], Xtg[:, 3], maxt, 20.0),
        (Xtr[:, 1], Xtr[:, 3], Xtg[:, 1], Xtg[:, 3], maxt, 20.0),
        (Xtr[:, 2], Xtr[:, 3], Xtg[:, 2], Xtg[:, 3], maxt, 20.0),
        (Xtr[:, 1], Xtr[:, 4], Xtg[:, 1], Xtg[:, 4], maxd, 20.0),
        (Xtr[:, 2], Xtr[:, 4], Xtg[:, 2], Xtg[:, 4], maxd, 20.0),
    ]
    for a, b, at, bt, nb, alpha in pairs:
        add_pair_stat(feats, a.astype(np.int64), b.astype(np.int64), ytr,
                      at.astype(np.int64), bt.astype(np.int64), gm, nb,
                      alpha=alpha, train=is_train)

    return np.column_stack(feats).astype(np.float32), ytg.astype(np.float32), utg


def split_counts(counts, max_group=9000):
    out = []
    for c in counts:
        c = int(c)
        while c > max_group:
            out.append(max_group)
            c -= max_group
        if c > 0:
            out.append(c)
    return out


def sort_by_user(X, y, users):
    order = np.argsort(users, kind='stable')
    su = np.asarray(users)[order]
    _, counts = np.unique(su, return_counts=True)
    groups = split_counts(counts.tolist(), max_group=9000)
    return X[order], y[order], groups, order


def train_and_predict(splits, target='valid', seed=0):
    t0 = time.time()
    enc, dim = encode(splits)
    print(f"encoded dim={dim}; building features ...")
    Xtr, ytr, utr = build_features(enc, splits, 'train')
    Xva, yva, uva = build_features(enc, splits, 'valid')
    Xtg, ytg, utg = (Xva, yva, uva) if target == 'valid' else build_features(enc, splits, target)
    print(f"features train={Xtr.shape} valid={Xva.shape} target={Xtg.shape} in {time.time() - t0:.1f}s")

    Xtrs, ytrs, gtr, _ = sort_by_user(Xtr, ytr, utr)
    Xvas, yvas, gva, _ = sort_by_user(Xva, yva, uva)
    print(f"groups train={len(gtr)} max={max(gtr)} valid={len(gva)} max={max(gva)}")

    model = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=300,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        label_gain=[0, 1],
        lambdarank_truncation_level=30,
        verbose=-1,
    )
    model.fit(
        Xtrs, ytrs,
        group=gtr,
        eval_set=[(Xvas, yvas)],
        eval_group=[gva],
        eval_at=[5],
        callbacks=[early_stopping(15, verbose=False), log_evaluation(0)],
    )
    print(f"best_iter={getattr(model, 'best_iteration_', None)} total_time={time.time() - t0:.1f}s")
    return model.predict(Xtg, num_iteration=model.best_iteration_).astype(np.float64)


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
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")

    preds = train_and_predict(splits, target=target, seed=a.seed)
    if a.out:
        np.save(a.out, preds)
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
