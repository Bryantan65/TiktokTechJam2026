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
    n = len(rows)
    y = np.fromiter((float(r[LABEL]) for r in rows), dtype=np.float64, count=n)
    # Keep the oracle 0/1 separation, but let --seed affect only sub-label tie breaking.
    # This tests whether the remaining nDCG loss is just tie/order artefact; the jitter is
    # far too small to move any negative above a positive.
    rng = np.random.default_rng(int(a.seed) + 20260830)
    jitter = rng.uniform(0.0, 1e-9, size=n)
    preds = y + jitter
    np.save(a.out, preds.astype(np.float64))
    print(f'wrote {n:,d} seeded label-oracle predictions for split={a.split} seed={a.seed}', flush=True)
