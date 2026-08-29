"""Improve node 31 by slightly increasing the global CTR tie-breaker weight.

Node 31 showed that a train-only global video/author/tab CTR prior corrects a few
close within-user mistakes at 1% weight.  This keeps the identical fixed seed-0
node-28 ensemble and the same prior, but tests whether 2% is closer to the
post-hoc tie-breaker optimum before this branch is exhausted.
"""
import argparse
import importlib.util
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec31 = importlib.util.spec_from_file_location('node31_impl', os.path.join(_here, '031_global_ctr_tiebreak.py'))
node31 = importlib.util.module_from_spec(_spec31)
_spec31.loader.exec_module(node31)


def run_predict(splits, data_dir, split='valid', seed=0, device='cpu', verbose=False):
    # Reconstruct node31's components but change only the global-CTR prior weight.
    base = node31.node28.run_predict(splits, data_dir, split=split, seed=seed, device=device, verbose=verbose).astype(np.float64)
    users = np.asarray([r[1] for r in splits[split]], dtype=np.int64)
    br = node31.node28.impl.within_user_ranks(base, users)
    gr = node31.node28.impl.within_user_ranks(node31.global_ctr_prior(splits, split), users)
    return 0.98 * br + 0.02 * gr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    print(f'loading {a.data_dir} ...')
    splits = node31.node28.impl.load(a.data_dir)
    print({k: len(v) for k, v in splits.items()}, 'fields=node28_plus_2pct_global_ctr_prior')
    scores = run_predict(splits, a.data_dir, split=a.split, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
