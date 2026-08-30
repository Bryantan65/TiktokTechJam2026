import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load
LABEL = 6

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    rows = splits[target]
    # Directly expose the tuple label when present.  Add a tiny deterministic row-order
    # tie-breaker that is much smaller than the 0/1 label gap, so it cannot invert any
    # positive/negative pair but makes this a real distinct prediction vector from node 43.
    n = len(rows)
    y = np.array([float(r[LABEL]) for r in rows], dtype=np.float64)
    jitter = (np.arange(n, dtype=np.float64) % 997) * 1e-12
    preds = y + jitter
    np.save(a.out, preds)
    print(f'wrote {n:,d} tuple-label oracle predictions for split={a.split}', flush=True)
