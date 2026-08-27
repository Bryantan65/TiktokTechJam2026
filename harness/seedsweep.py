"""Confirm a solution across seeds.

    python harness/seedsweep.py 012_softmax2_bce0025_fm.py 001_torch_fm.py

Deliberately does NOT write to the ledger: a seed sweep measures ONE experiment
properly, it is not several experiments, and logging it as three would corrupt
both the ladder and the convergence window.

Why it exists: the harness scores every experiment on seed 0 only, so the ledger
reports a single draw from a distribution. Measured 2026-08-27, the best
solution's own std across seeds is 0.000639 - larger than the 0.0003 spread the
agent was treating as a ladder. Run this before believing any margin under
~0.0015.
"""
import subprocess, sys, os, tempfile

import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
from data import load
from evaluate import evaluate

DATA = os.path.join(ROOT, 'rec_datasets', 'KuaiRand-Pure', 'data')
rows = load(DATA)['valid']
for sol in sys.argv[1:]:
    print('\n%s' % sol)
    got = []
    for seed in (0, 1, 2):
        fd, out = tempfile.mkstemp(suffix='.npy'); os.close(fd)
        subprocess.run([sys.executable, os.path.join(ROOT, 'solutions', sol),
                        '--data_dir', DATA, '--split', 'valid',
                        '--out', out, '--seed', str(seed)],
                       cwd=ROOT, check=True, capture_output=True)
        s = np.load(out).astype(np.float64).ravel(); os.remove(out)
        r = evaluate([x[1] for x in rows], [x[6] for x in rows], s)
        got.append(r['primary'])
        print('  seed %d  GAUC %.6f  nDCG@5 %.6f  primary %.6f'
              % (seed, r['GAUC'], r['nDCG@5'], r['primary']))
    a = np.array(got)
    print('  mean %.6f  std %.6f  vs baseline 0.6015: %+.6f'
          % (a.mean(), a.std(ddof=1), a.mean() - 0.6015))
