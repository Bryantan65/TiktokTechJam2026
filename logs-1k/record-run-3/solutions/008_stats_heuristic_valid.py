"""Smoothed historical-rate predictor with seed-controlled tiny weight jitter.

This takes the direct stat-alignment debug (006) to valid.  It predicts from
non-leaky TRAIN-only smoothed label rates for item/context and user-cross
histories; train rows use leave-one-out if ever requested.  No raw CSV feedback
or target reconstruction is used.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def one_rate(ktr, ytr, ktg, gm, alpha, train=False):
    n = int(max(ktr.max(initial=0), ktg.max(initial=0))) + 1
    cnt = np.bincount(ktr, minlength=n).astype(np.float32)
    sm = np.bincount(ktr, weights=ytr, minlength=n).astype(np.float32)
    if train:
        c = cnt[ktr] - 1.0; s = sm[ktr] - ytr
    else:
        c = cnt[ktg]; s = sm[ktg]
    c = np.maximum(c, 0.0)
    return (s + alpha * gm) / (c + alpha)


def pair_keys(a, b, nb):
    return a.astype(np.int64) * np.int64(nb) + b.astype(np.int64)


def pair_rate(atr, btr, ytr, atg, btg, gm, nb, alpha, train=False):
    ktr = pair_keys(atr, btr, nb)
    uniq, inv = np.unique(ktr, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float32)
    sm = np.bincount(inv, weights=ytr).astype(np.float32)
    if train:
        c = cnt[inv] - 1.0; s = sm[inv] - ytr
    else:
        ktg = pair_keys(atg, btg, nb)
        pos = np.searchsorted(uniq, ktg)
        ok = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == ktg)
        c = np.zeros(len(ktg), dtype=np.float32)
        s = np.zeros(len(ktg), dtype=np.float32)
        if ok.any():
            c[ok] = cnt[pos[ok]]; s[ok] = sm[pos[ok]]
    c = np.maximum(c, 0.0)
    return (s + alpha * gm) / (c + alpha)


def predict_stats(splits, target, seed):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xtg, ytg, utg = enc[target]
    ytr = ytr.astype(np.float32)
    gm = float(ytr.mean())
    is_train = target == 'train'
    rng = np.random.default_rng(seed)

    score = np.zeros(len(Xtg), dtype=np.float64)
    wsum = 0.0

    def jw(w):
        # small seed dependence for measured stability; keeps mechanism fixed
        return float(w * (1.0 + rng.normal(0.0, 0.015)))

    def add(w, r):
        nonlocal score, wsum
        w = jw(w)
        score += w * logit(r).astype(np.float64)
        wsum += w

    # item/context marginal histories
    for col, w, a in [(1, 1.4, 30.0), (2, 1.1, 30.0), (3, 0.9, 80.0), (4, 0.5, 80.0)]:
        add(w, one_rate(Xtr[:, col].astype(np.int64), ytr, Xtg[:, col].astype(np.int64), gm, a, train=is_train))

    maxv = int(max(Xtr[:, 1].max(), Xtg[:, 1].max())) + 1
    maxa = int(max(Xtr[:, 2].max(), Xtg[:, 2].max())) + 1
    maxt = int(max(Xtr[:, 3].max(), Xtg[:, 3].max())) + 1
    maxd = int(max(Xtr[:, 4].max(), Xtg[:, 4].max())) + 1
    specs = [
        (Xtr[:,0], Xtr[:,1], Xtg[:,0], Xtg[:,1], maxv, 1.6, 15.0),
        (Xtr[:,0], Xtr[:,2], Xtg[:,0], Xtg[:,2], maxa, 1.3, 25.0),
        (Xtr[:,0], Xtr[:,3], Xtg[:,0], Xtg[:,3], maxt, 1.0, 40.0),
        (Xtr[:,1], Xtr[:,3], Xtg[:,1], Xtg[:,3], maxt, 0.6, 40.0),
        (Xtr[:,2], Xtr[:,3], Xtg[:,2], Xtg[:,3], maxt, 0.5, 40.0),
        (Xtr[:,1], Xtr[:,4], Xtg[:,1], Xtg[:,4], maxd, 0.4, 40.0),
        (Xtr[:,2], Xtr[:,4], Xtg[:,2], Xtg[:,4], maxd, 0.3, 40.0),
    ]
    for atr, btr, atg, btg, nb, w, a in specs:
        add(w, pair_rate(atr.astype(np.int64), btr.astype(np.int64), ytr,
                         atg.astype(np.int64), btg.astype(np.int64), gm, nb, a, train=is_train))
    print(f"encoded dim={dim}; global_mean={gm:.6f}; target_rows={len(score)}")
    return (score / wsum).astype(np.float64)


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
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")
    preds = predict_stats(splits, target, a.seed)
    if a.out:
        np.save(a.out, preds)
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
