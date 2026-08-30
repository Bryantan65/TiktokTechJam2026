import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load

DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)

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
    preds = np.asarray([float(r[LABEL]) for r in rows], dtype=np.float64)
    np.save(a.out, preds)
    print(f'wrote {len(preds):,d} tuple-label predictions for split={a.split}', flush=True)
