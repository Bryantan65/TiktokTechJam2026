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
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split

    rows = splits[target]
    y = np.array([float(r[LABEL]) for r in rows], dtype=np.float64)

    # Seed-dependent tie-breaking, but with a wide margin so no negative can outrank
    # a positive.  This should preserve the label-oracle ranking while proving that
    # the residual nDCG gap is a metric ceiling rather than deterministic tie order.
    rng = np.random.default_rng(int(a.seed) + 12345)
    jitter = rng.random(len(rows), dtype=np.float64)
    preds = 10.0 * y + jitter

    np.save(a.out, preds.astype(np.float64))
    print(f'wrote {len(preds):,d} seeded label-oracle predictions for split={a.split}', flush=True)
