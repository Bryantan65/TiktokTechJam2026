"""Improve node 28 with a tiny train-only global CTR tie-breaker.

The current best is fixed seed-0 node24 plus a 5% user-history prior.  This
keeps that full predictor unchanged and adds only a 1% within-user rank prior
from smoothed train-label video/author/context CTRs.  The aim is to resolve close
within-user calls using item/context quality that is available before valid/test
without using their labels.
"""
import argparse
import importlib.util
import os
from collections import defaultdict
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('node28_impl', os.path.join(_here, '028_seed0_history_tiebreak.py'))
node28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(node28)


def _ctr(pos, cnt, prior=0.37, alpha=20.0):
    return (pos + alpha * prior) / (cnt + alpha)


def _dur_bucket(d):
    try:
        return int(d) // 10000
    except Exception:
        return 0


def global_ctr_prior(splits, split):
    pv = defaultdict(float); cv = defaultdict(int)
    pa = defaultdict(float); ca = defaultdict(int)
    pt = defaultdict(float); ct = defaultdict(int)
    pd = defaultdict(float); cd = defaultdict(int)
    pat = defaultdict(float); cat = defaultdict(int)
    ptd = defaultdict(float); ctd = defaultdict(int)

    # Fit only on train labels.
    for row in splits['train']:
        date, u, v, a, tab, dur, y = row
        y = float(y > 0.5); db = _dur_bucket(dur)
        pv[v] += y; cv[v] += 1
        pa[a] += y; ca[a] += 1
        pt[tab] += y; ct[tab] += 1
        pd[db] += y; cd[db] += 1
        pat[(a, tab)] += y; cat[(a, tab)] += 1
        ptd[(tab, db)] += y; ctd[(tab, db)] += 1

    vals = []
    for row in splits[split]:
        date, u, v, a, tab, dur, y = row
        db = _dur_bucket(dur)
        # Strong shrinkage: this is only a close-score tie-breaker, not a second model.
        rv = _ctr(pv[v], cv[v], 0.37, 30.0)
        ra = _ctr(pa[a], ca[a], 0.37, 30.0)
        rt = _ctr(pt[tab], ct[tab], 0.37, 50.0)
        rd = _ctr(pd[db], cd[db], 0.37, 50.0)
        rat = _ctr(pat[(a, tab)], cat[(a, tab)], 0.37, 40.0)
        rtd = _ctr(ptd[(tab, db)], ctd[(tab, db)], 0.37, 40.0)
        vals.append(0.30 * rat + 0.24 * ra + 0.18 * rv + 0.12 * rtd + 0.10 * rt + 0.06 * rd)
    return np.asarray(vals, dtype=np.float64)


def run_predict(splits, data_dir, split='valid', seed=0, device='cpu', verbose=False):
    base = node28.run_predict(splits, data_dir, split=split, seed=seed, device=device, verbose=verbose).astype(np.float64)
    users = np.asarray([r[1] for r in splits[split]], dtype=np.int64)
    br = node28.impl.within_user_ranks(base, users)
    gr = node28.impl.within_user_ranks(global_ctr_prior(splits, split), users)
    return 0.99 * br + 0.01 * gr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    print(f'loading {a.data_dir} ...')
    splits = node28.impl.load(a.data_dir)
    print({k: len(v) for k, v in splits.items()}, 'fields=node28_plus_tiny_global_ctr_prior')
    scores = run_predict(splits, a.data_dir, split=a.split, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
