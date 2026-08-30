"""Draft user-behaviour sequence signal on top of node 20.

This reuses node 20 unchanged, then adds a DIN-inspired history member: for each
candidate row, score overlap with the user's past positive/negative train
history (same video/author/tab/duration plus recency).  It is intentionally a
cheap standalone sequence test and is blended at 30% so its signal is readable.
"""
import argparse, os, sys, math
from collections import defaultdict
import numpy as np

# Reuse the current-best training/caching code unchanged.  The solution remains
# runnable with an empty pred_cache because node 20 trains missing members.
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


def new_hist():
    return {
        'pv': defaultdict(int), 'nv': defaultdict(int),
        'pa': defaultdict(int), 'na': defaultdict(int),
        'pt': defaultdict(int), 'nt': defaultdict(int),
        'pd': defaultdict(int), 'nd': defaultdict(int),
        'lv': {}, 'la': {}, 'cnt': 0,
    }


def add_row(hist, r):
    u, v, a, t, d, y = str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])), int(r[6])
    h = hist[u]; h['cnt'] += 1; c = h['cnt']
    if y > 0:
        h['pv'][v] += 1; h['pa'][a] += 1; h['pt'][t] += 1; h['pd'][d] += 1
        h['lv'][v] = c; h['la'][a] = c
    else:
        h['nv'][v] += 1; h['na'][a] += 1; h['nt'][t] += 1; h['nd'][d] += 1


def score_row(hist, r):
    u, v, a, t, d = str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5]))
    h = hist.get(u)
    if h is None or h['cnt'] == 0:
        return 0.0
    s = 0.0
    # Repeated item/author positives are the closest cheap approximation to
    # attention over a user's behaviour sequence; negatives downweight repeats.
    s += 1.20 * math.log1p(h['pv'].get(v, 0)) - 0.45 * math.log1p(h['nv'].get(v, 0))
    s += 0.85 * math.log1p(h['pa'].get(a, 0)) - 0.25 * math.log1p(h['na'].get(a, 0))
    s += 0.18 * math.log1p(h['pt'].get(t, 0)) - 0.08 * math.log1p(h['nt'].get(t, 0))
    s += 0.12 * math.log1p(h['pd'].get(d, 0)) - 0.05 * math.log1p(h['nd'].get(d, 0))
    c = h['cnt']
    if v in h['lv']:
        age = max(0, c - h['lv'][v]); s += 0.75 / math.sqrt(1.0 + age)
    if a in h['la']:
        age = max(0, c - h['la'][a]); s += 0.45 / math.sqrt(1.0 + age)
    return s


def sequence_scores(splits, target):
    hist = defaultdict(new_hist)
    out = np.zeros(len(splits[target]), dtype=np.float64)
    if target == 'train':
        # Online train predictions: only use prior rows to avoid self-label leakage.
        for i, r in enumerate(splits['train']):
            out[i] = score_row(hist, r)
            add_row(hist, r)
        return out
    for r in splits['train']:
        add_row(hist, r)
    for i, r in enumerate(splits[target]):
        out[i] = score_row(hist, r)
    return out


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
    print({k: len(v) for k, v in splits.items()}, 'fields=node20+sequence_overlap')
    base = node20.run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    enc, _ = encode(splits); users = enc[target][2]
    seq = sequence_scores(splits, target)
    bz = per_user_zscore(base, users); sz = per_user_zscore(seq, users)
    scores = 0.70 * bz + 0.30 * sz
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}; seq std={float(np.std(seq)):.4f}')

if __name__ == '__main__':
    main()
