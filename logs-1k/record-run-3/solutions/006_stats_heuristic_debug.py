"""Debug stat feature alignment with a direct historical-rate predictor.

No model is trained.  Predictions are a fixed weighted blend of smoothed TRAIN
label rates for video/author/tab/duration and user-cross histories.  If this is
near-random, the history feature construction or row alignment is wrong; if it
is reasonable, the poor LightGBM runs are an objective/modeling issue.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def one_rate(ktr, ytr, ktg, gm, alpha=20.0, train=False):
    n = int(max(ktr.max(initial=0), ktg.max(initial=0))) + 1
    cnt = np.bincount(ktr, minlength=n).astype(np.float32)
    sm = np.bincount(ktr, weights=ytr, minlength=n).astype(np.float32)
    if train:
        c = cnt[ktr] - 1.0
        s = sm[ktr] - ytr
    else:
        c = cnt[ktg]
        s = sm[ktg]
    return (s + alpha * gm) / (np.maximum(c, 0.0) + alpha), np.maximum(c, 0.0)


def pair_keys(a, b, nb):
    return a.astype(np.int64) * np.int64(nb) + b.astype(np.int64)


def pair_rate(atr, btr, ytr, atg, btg, gm, nb, alpha=20.0, train=False):
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
    return (s + alpha * gm) / (np.maximum(c, 0.0) + alpha), np.maximum(c, 0.0)


def predict_from_stats(splits, target):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xtg, ytg, utg = enc[target]
    ytr = ytr.astype(np.float32)
    gm = float(ytr.mean())
    is_train = (target == 'train')

    score = np.zeros(len(Xtg), dtype=np.float64)
    weight_sum = 0.0

    def add(w, rate):
        nonlocal score, weight_sum
        score += w * logit(rate).astype(np.float64)
        weight_sum += w

    # Single-field histories.  Video/author identity should carry most of the
    # cross-user item quality signal; tab/duration capture large base-rate gaps.
    for col, w, a in [
        (1, 1.4, 30.0),   # video
        (2, 1.1, 30.0),   # author
        (3, 0.9, 80.0),   # tab
        (4, 0.5, 80.0),   # duration bucket
    ]:
        r, c = one_rate(Xtr[:, col].astype(np.int64), ytr,
                        Xtg[:, col].astype(np.int64), gm, alpha=a,
                        train=is_train)
        add(w, r)

    maxv = int(max(Xtr[:, 1].max(), Xtg[:, 1].max())) + 1
    maxa = int(max(Xtr[:, 2].max(), Xtg[:, 2].max())) + 1
    maxt = int(max(Xtr[:, 3].max(), Xtg[:, 3].max())) + 1
    maxd = int(max(Xtr[:, 4].max(), Xtg[:, 4].max())) + 1

    # User-conditioned histories affect ranking within a user.  Stronger prior
    # on sparse pairs avoids memorising a single old impression too hard.
    pair_specs = [
        (Xtr[:, 0], Xtr[:, 1], Xtg[:, 0], Xtg[:, 1], maxv, 1.6, 15.0),  # user-video
        (Xtr[:, 0], Xtr[:, 2], Xtg[:, 0], Xtg[:, 2], maxa, 1.3, 25.0),  # user-author
        (Xtr[:, 0], Xtr[:, 3], Xtg[:, 0], Xtg[:, 3], maxt, 1.0, 40.0),  # user-tab
        (Xtr[:, 1], Xtr[:, 3], Xtg[:, 1], Xtg[:, 3], maxt, 0.6, 40.0),  # video-tab
        (Xtr[:, 2], Xtr[:, 3], Xtg[:, 2], Xtg[:, 3], maxt, 0.5, 40.0),  # author-tab
        (Xtr[:, 1], Xtr[:, 4], Xtg[:, 1], Xtg[:, 4], maxd, 0.4, 40.0),  # video-dur
        (Xtr[:, 2], Xtr[:, 4], Xtg[:, 2], Xtg[:, 4], maxd, 0.3, 40.0),  # author-dur
    ]
    for atr, btr, atg, btg, nb, w, a in pair_specs:
        r, c = pair_rate(atr.astype(np.int64), btr.astype(np.int64), ytr,
                         atg.astype(np.int64), btg.astype(np.int64), gm, nb,
                         alpha=a, train=is_train)
        add(w, r)

    score /= max(weight_sum, 1e-9)
    print(f"encoded dim={dim}; global_mean={gm:.6f}; target_rows={len(score)}")
    return score.astype(np.float64)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)  # accepted; heuristic is deterministic
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

    preds = predict_from_stats(splits, target)
    if a.out:
        np.save(a.out, preds)
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
