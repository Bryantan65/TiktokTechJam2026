"""Debug node 21: learn the sequence/history signal instead of hard-coding it.

Node 21 blended a hand-weighted history-overlap score and dropped badly.  This
keeps node 20 unchanged, builds leakage-safe online prior-history buckets for
train and train-history buckets for valid/test, trains a same-user BPR FM member
on those sequence context fields, and blends it at a readable 30% weight.
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'kuairand-starter-kit'))
import importlib.util
spec = importlib.util.spec_from_file_location('node20', os.path.join(HERE, '020_dual_time_blend40.py'))
node20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(node20)
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
            'lv':{},'la':{},'cnt':0,'pos':0,'neg':0}


def hist_vals(h, r):
    v, a, t, d = str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5]))
    c = h['cnt']
    return (
        cb(h['pv'].get(v,0)), cb(h['nv'].get(v,0)),
        cb(h['pa'].get(a,0)), cb(h['na'].get(a,0)),
        cb(h['pt'].get(t,0)), cb(h['nt'].get(t,0)),
        cb(h['pd'].get(d,0)), cb(h['nd'].get(d,0)),
        rb(None if v not in h['lv'] else c - h['lv'][v]),
        rb(None if a not in h['la'] else c - h['la'][a]),
        cb(h['pos']), cb(h['neg'])
    )


def add_hist(h, r):
    v, a, t, d, y = str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])), int(r[6])
    h['cnt'] += 1; c = h['cnt']
    if y > 0:
        h['pv'][v] += 1; h['pa'][a] += 1; h['pt'][t] += 1; h['pd'][d] += 1
        h['lv'][v] = c; h['la'][a] = c; h['pos'] += 1
    else:
        h['nv'][v] += 1; h['na'][a] += 1; h['nt'][t] += 1; h['nd'][d] += 1; h['neg'] += 1


def build_seq_rows(splits):
    vals_by = {}
    maps = [dict() for _ in range(17)]
    # Train features use only preceding rows for that user.
    hist = defaultdict(empty_hist); vals = []
    for r in splits['train']:
        ubase = (str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])))
        u = str(r[1]); vr = ubase + hist_vals(hist[u], r)
        vals.append(vr); add_hist(hist[u], r)
    vals_by['train'] = vals
    for vr in vals:
        for j, v in enumerate(vr):
            if v not in maps[j]: maps[j][v] = len(maps[j])
    # Valid/test features use full train history only, avoiding target self-label leakage.
    base_hist = hist
    for sp in splits:
        if sp == 'train': continue
        vals = []
        for r in splits[sp]:
            ubase = (str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])))
            u = str(r[1]); vr = ubase + hist_vals(base_hist[u], r)
            vals.append(vr)
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
    scores = scores.astype(np.float64, copy=True); groups = defaultdict(list)
    for i, u in enumerate(users): groups[u].append(i)
    for idxs in groups.values():
        idx = np.asarray(idxs, dtype=np.int64); vals = scores[idx]; sd = vals.std()
        scores[idx] = (vals - vals.mean()) / sd if sd > 1e-12 else 0.0
    return scores


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
    print({k: len(v) for k, v in splits.items()}, 'fields=node20+learned_seq_context')
    base = node20.run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    enc0, _ = encode(splits); users = enc0[target][2]
    senc, sdim = build_seq_rows(splits)
    member_seeds = [a.seed + 1000 * m for m in range(3)]
    seq_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(senc, sdim, target, a.split, ms, a.device, a.out is None,
                                      '022_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        seq_members.append(per_user_zscore(p, users))
    seq = per_user_zscore(np.mean(np.vstack(seq_members), axis=0), users)
    scores = 0.70 * per_user_zscore(base, users) + 0.30 * seq
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}')

if __name__ == '__main__':
    main()
