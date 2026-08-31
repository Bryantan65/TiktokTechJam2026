"""Measure the FM baseline on a variant too large to load whole.

Why this exists. The bonus benchmarks have no published baseline, so the scoring
formula `delta = agent - baseline` has no reference number until we measure one.
On 1k that is a straight run of the kit's own baseline.py. On 27k it is not:

  322M rows as Python tuples   ~110 GB
  container memory limit        116 GB
  -> killed mid-load, no traceback, because an OOM kill gives the process no
     chance to write one. `free` reports the HOST's 503 GB, not the container's
     limit, so the failure looks impossible right up until it happens.

Two changes make it fit. First, `data.load(only=...)` skips rows outside the
requested splits as the CSVs are read rather than building and discarding them -
train+valid is 208M rows, about 71 GB. Second, the test split is not needed:
Deliverable 4 asks for the validation-best score, and the hidden test is scored
by the organisers.

The model is the PyTorch port of the kit's FM, at the kit's own hyperparameters.
It reproduces the kit to 0.0001 on Pure (0.6015 vs 0.6016) but sits 0.0017 BELOW
it on 1k (0.643428 vs 0.6451). A baseline biased low inflates the delta measured
against it, so that gap favours us and must be disclosed wherever this number is
reported. Prefer the kit's own baseline.py wherever it can actually run.

    python harness/measure_baseline.py --data_dir rec_datasets/KuaiRand-27K/data
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import data as D                                    # noqa: E402  variant-aware
sys.path.append(os.path.join(ROOT, 'kuairand-starter-kit'))
from evaluate import evaluate                       # noqa: E402  official


def rss_gb():
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576.0
    except Exception:
        pass
    return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()

    print('variant: %s' % D.variant(a.data_dir), flush=True)
    print('loading train+valid only (test is not needed for a baseline)', flush=True)
    t0 = time.time()
    splits = D.load(a.data_dir, only=['train', 'valid'])
    print('  loaded in %.0f min  rss %.0f GB  %s'
          % ((time.time() - t0) / 60, rss_gb(),
             {k: len(v) for k, v in splits.items()}), flush=True)

    # The port lives in solutions/001_torch_fm.py, whose module name starts with
    # a digit and so cannot be imported. Execute it for its definitions instead;
    # its __main__ block is skipped because __name__ is set to something else.
    src = open(os.path.join(ROOT, 'solutions', '001_torch_fm.py'),
               encoding='utf-8').read()
    g = {'__name__': 'torch_fm_lib', '__file__': os.path.join(ROOT, 'solutions',
                                                              '001_torch_fm.py')}
    exec(compile(src, '001_torch_fm.py', 'exec'), g)

    t1 = time.time()
    print('training (device=%s) ...' % a.device, flush=True)
    model, enc = g['run'](splits, k=a.k, lr=a.lr, epochs=a.epochs,
                          seed=a.seed, device=a.device, verbose=True)
    print('  trained in %.0f min' % ((time.time() - t1) / 60), flush=True)

    X, y, users = enc['valid']
    r = evaluate(users, y, model.predict(X, device=a.device))
    print()
    print('=== FM baseline, %s, validation ===' % D.variant(a.data_dir))
    print('  GAUC    %.6f' % r['GAUC'])
    print('  nDCG@5  %.6f' % r['nDCG@5'])
    print('  primary %.6f' % r['primary'])
    print()
    print('  measured with the PyTorch port, not the kit numpy script.')
    print('  port vs kit: +0.0001 on Pure, -0.0017 on 1k (port lower).')
    print('  total %.0f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
