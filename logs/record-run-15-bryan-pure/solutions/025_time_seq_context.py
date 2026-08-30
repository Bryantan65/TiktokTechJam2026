"""Improve node 22: make the learned sequence-context member time-aware.

Node 22's sequence BPR member helped, but it only learned prior user context on
item/author/tab/duration while node20's time signal lives in separate members.
This keeps node20 and the node22 member unchanged via caches and adds a 30%
readable blend member whose history buckets also include hour/dow/tab-hour and
per-user prior positives/negatives in the current hour bucket.
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'kuairand-starter-kit'))
import importlib.util
spec22 = importlib.util.spec_from_file_location('node22', os.path.join(HERE, '022_seq_context_fm.py'))
node22 = importlib.util.module_from_spec(spec22); spec22.loader.exec_module(node22)
node20 = node22.node20
from data import load, encode


def dur_bucket(ms):
    try: x = float(ms)
    except Exception: x = 0.0
    if x < 5000: return 0
    if x < 10000: return 1
    if x < 20000: return 2
    if x < 30000: return 3
    if x < 60000: return 4
    if x < 120000: return 5
    return 6


def cb(x):
    x = int(x)
    if x <= 0: return '0'
    if x == 1: return '1'
    if x <= 3: return '2-3'
    if x <= 7: return '4-7'
    return '8+'


def rb(age):
    if age is None: return 'none'
    if age <= 1: return '1'
    if age <= 3: return '2-3'
    if age <= 10: return '4-10'
    if age <= 50: return '11-50'
    return '50+'


def empty_hist():
    return {'pv':defaultdict(int),'nv':defaultdict(int),'pa':defaultdict(int),'na':defaultdict(int),
            'pt':defaultdict(int),'nt':defaultdict(int),'pd':defaultdict(int),'nd':defaultdict(int),
            'ph':defaultdict(int),'nh':defaultdict(int),'pth':defaultdict(int),'nth':defaultdict(int),
            'lv':{},'la':{},'lh':{},'cnt':0,'pos':0,'neg':0}


def h4_of(hour):
    h = int(hour)
    return 'miss' if h >= 24 else str(h // 4)


def hist_vals(h, r, hour):
    v, a, t, d, hb = str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])), h4_of(hour)
    th = t + '_' + hb
    c = h['cnt']
    return (
        cb(h['pv'].get(v,0)), cb(h['nv'].get(v,0)),
        cb(h['pa'].get(a,0)), cb(h['na'].get(a,0)),
        cb(h['pt'].get(t,0)), cb(h['nt'].get(t,0)),
        cb(h['pd'].get(d,0)), cb(h['nd'].get(d,0)),
        cb(h['ph'].get(hb,0)), cb(h['nh'].get(hb,0)),
        cb(h['pth'].get(th,0)), cb(h['nth'].get(th,0)),
        rb(None if v not in h['lv'] else c - h['lv'][v]),
        rb(None if a not in h['la'] else c - h['la'][a]),
        rb(None if hb not in h['lh'] else c - h['lh'][hb]),
        cb(h['pos']), cb(h['neg'])
    )


def add_hist(h, r, hour):
    v, a, t, d, hb, y = str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])), h4_of(hour), int(r[6])
    th = t + '_' + hb
    h['cnt'] += 1; c = h['cnt']
    if y > 0:
        h['pv'][v] += 1; h['pa'][a] += 1; h['pt'][t] += 1; h['pd'][d] += 1
        h['ph'][hb] += 1; h['pth'][th] += 1
        h['lv'][v] = c; h['la'][a] = c; h['lh'][hb] = c; h['pos'] += 1
    else:
        h['nv'][v] += 1; h['na'][a] += 1; h['nt'][t] += 1; h['nd'][d] += 1
        h['nh'][hb] += 1; h['nth'][th] += 1; h['neg'] += 1


def build_time_seq_rows(splits, hours):
    vals_by = {}
    maps = None
    # base: user, video, author, tab, dur, hour4, dow, tab_hour + 17 history fields
    nfields = 25
    maps = [dict() for _ in range(nfields)]
    hist = defaultdict(empty_hist); vals = []
    htrain = hours.get('train', np.full(len(splits['train']), 24, dtype=np.int64))
    for i, r in enumerate(splits['train']):
        hb = h4_of(htrain[i]); dow = str(node20.day_of_week(r[0])); tab = str(r[4])
        ubase = (str(r[1]), str(r[2]), str(r[3]), tab, str(dur_bucket(r[5])), hb, dow, tab + '_' + hb)
        u = str(r[1]); vr = ubase + hist_vals(hist[u], r, htrain[i])
        vals.append(vr); add_hist(hist[u], r, htrain[i])
    vals_by['train'] = vals
    for vr in vals:
        for j, v in enumerate(vr):
            if v not in maps[j]: maps[j][v] = len(maps[j])
    base_hist = hist
    for sp in splits:
        if sp == 'train': continue
        vals = []; hs = hours.get(sp, np.full(len(splits[sp]), 24, dtype=np.int64))
        for i, r in enumerate(splits[sp]):
            hb = h4_of(hs[i]); dow = str(node20.day_of_week(r[0])); tab = str(r[4])
            ubase = (str(r[1]), str(r[2]), str(r[3]), tab, str(dur_bucket(r[5])), hb, dow, tab + '_' + hb)
            u = str(r[1]); vals.append(ubase + hist_vals(base_hist[u], r, hs[i]))
        vals_by[sp] = vals
        for vr in vals:
            for j, v in enumerate(vr):
                if v not in maps[j]: maps[j][v] = len(maps[j])
    offsets=[]; off=0
    for m in maps:
        offsets.append(off); off += len(m)
    enc = {}
    for sp, vals in vals_by.items():
        X = np.zeros((len(vals), len(maps)), dtype=np.int64)
        for i, vr in enumerate(vals):
            for j, v in enumerate(vr): X[i, j] = offsets[j] + maps[j][v]
        enc[sp] = (X, np.asarray([r[6] for r in splits[sp]], dtype=np.float32), np.asarray([r[1] for r in splits[sp]], dtype=object))
    return enc, off


def per_user_zscore(scores, users):
    return node22.per_user_zscore(scores, users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args()
    print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, 'fields=node22+time_seq_context')

    # Reconstruct node22 incumbent exactly (node20 base + node22 sequence member).
    base = node20.run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    enc0, _ = encode(splits); users = enc0[target][2]
    senc, sdim = node22.build_seq_rows(splits)
    member_seeds = [a.seed + 1000 * m for m in range(3)]
    seq_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(senc, sdim, target, a.split, ms, a.device, a.out is None,
                                      '022_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        seq_members.append(per_user_zscore(p, users))
    seq = per_user_zscore(np.mean(np.vstack(seq_members), axis=0), users)
    incumbent = 0.70 * per_user_zscore(base, users) + 0.30 * seq

    hours = node20.load_hours(a.data_dir, splits, verbose=a.out is None)
    tsenc, tsdim = build_time_seq_rows(splits, hours)
    tseq_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(tsenc, tsdim, target, a.split, ms, a.device, a.out is None,
                                      '025_time_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        tseq_members.append(per_user_zscore(p, users))
    tseq = per_user_zscore(np.mean(np.vstack(tseq_members), axis=0), users)
    scores = 0.70 * per_user_zscore(incumbent, users) + 0.30 * tseq
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}')

if __name__ == '__main__':
    main()
