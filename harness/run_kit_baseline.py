"""Run the Starter Kit's own baseline.py on any KuaiRand variant.

Why this exists. `kuairand-starter-kit/data.py` hardcodes Pure's filenames, and
the kit is read-only, so `harness/data.py` provides a variant-aware `load()`
instead. Putting `harness/` on PYTHONPATH is NOT enough to make the kit pick it
up:

    PYTHONPATH=harness;kuairand-starter-kit \
      python kuairand-starter-kit/baseline.py --data_dir .../KuaiRand-1K/data
    -> FileNotFoundError: video_features_basic_pure.csv

Python puts the *script's own directory* at sys.path[0], ahead of everything on
PYTHONPATH. Running `kuairand-starter-kit/baseline.py` therefore guarantees the
kit's own data.py wins, whatever PYTHONPATH says. That command was in
docs/bonus-baselines.md and could never have worked on a non-Pure variant.

The fix is to import harness/data.py FIRST, under the name `data`. Once it is in
sys.modules, the kit's `from data import load` finds it there and never searches
the path at all. The kit stays byte-identical; nothing is patched.

    python harness/run_kit_baseline.py \
      --data_dir rec_datasets/KuaiRand-1K/data --model fm --seed 0

All arguments are passed through to baseline.py unchanged.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'kuairand-starter-kit')

# 1. Bind the variant-aware loader as `data` before anything else can.
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import data                                    # noqa: E402,F401  now sys.modules['data']

# 2. Let the kit find evaluate.py and its other siblings, but too late to matter
#    for `data` - that name is already resolved.
sys.path.insert(1, KIT)

if __name__ == '__main__':
    print('loader in use: %s' % data.__file__)
    print('variant detected: %s' % data.variant(
        sys.argv[sys.argv.index('--data_dir') + 1]
        if '--data_dir' in sys.argv else os.path.join(
            ROOT, 'rec_datasets', 'KuaiRand-Pure', 'data')))
    print()
    sys.argv[0] = os.path.join(KIT, 'baseline.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
