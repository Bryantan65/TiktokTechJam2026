"""Score a submission CSV on the test split. One-shot use only."""
import argparse
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))

from data import load
from evaluate import evaluate
from submit import read_submission


def main():
    ap = argparse.ArgumentParser(description='Score a submission on test')
    ap.add_argument('submission', help='Path to submission.csv')
    ap.add_argument('--data_dir',
                    default=os.path.join(ROOT, 'rec_datasets', 'KuaiRand-Pure', 'data'))
    a = ap.parse_args()

    splits = load(a.data_dir)
    rows = splits['test']
    scores = np.array(read_submission(a.submission, rows), dtype=np.float64)

    res = evaluate([r[1] for r in rows], [r[6] for r in rows], scores)

    baseline_test = 0.5946
    print(f"test GAUC:    {res['GAUC']:.6f}")
    print(f"test nDCG@5:  {res['nDCG@5']:.6f}")
    print(f"test primary: {res['primary']:.6f}")
    print(f"delta vs baseline ({baseline_test}): {res['primary'] - baseline_test:+.4f}")


if __name__ == '__main__':
    main()
