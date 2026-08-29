"""Combine the two positive deterministic tie-breaker priors.

Node 31's 1% global CTR prior and node 30's 5% train-only watch/completion
prior each improved the fixed node-28 seed-0 ensemble slightly.  This keeps the
same unchanged base predictor and tests whether the two priors are complementary
when blended at their previously useful weights.
"""
import argparse
import importlib.util
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec31 = importlib.util.spec_from_file_location('node31_impl', os.path.join(_here, '031_global_ctr_tiebreak.py'))
node31 = importlib.util.module_from_spec(_spec31)
_spec31.loader.exec_module(node31)

_spec30 = importlib.util.spec_from_file_location('node30_impl', os.path.join(_here, '030_watchtime_prior_tiebreak.py'))
node30 = importlib.util.module_from_spec(_spec30)
_spec30.loader.exec_module(node30)


def run_predict(splits, data_dir, split='valid', seed=0, device='cpu', verbose=False):
    # Start from the common node-28 fixed seed-0 ensemble, not from node31/node30 outputs,
    # so the final weights are exactly: 94% base rank, 5% watch prior, 1% global CTR prior.
    base = node31.node28.run_predict(splits, data_dir, split=split, seed=seed, device=device, verbose=verbose).astype(np.float64)
    users = np.asarray([r[1] for r in splits[split]], dtype=np.int64)
    ranker = node31.node28.impl.within_user_ranks
    br = ranker(base, users)
    wr = ranker(node30.watchtime_prior(splits, data_dir, split), users)
    gr = ranker(node31.global_ctr_prior(splits, split), users)
    return 0.94 * br + 0.05 * wr + 0.01 * gr


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
    print({k: len(v) for k, v in splits.items()}, 'fields=node28_plus_ctr_and_watch_priors')
    scores = run_predict(splits, a.data_dir, split=a.split, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
