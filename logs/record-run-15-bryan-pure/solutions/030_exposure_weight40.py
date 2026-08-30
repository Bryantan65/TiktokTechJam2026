"""Improve node 29: increase the readable weight of the exposure-context member.

Node29's label-free exposure/recency member improved both GAUC and nDCG at 30%.
Keep all member training and caches unchanged, and test whether the stronger
exposure-context signal should get 40% of the final blend.
"""
import argparse, os, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'kuairand-starter-kit'))

spec29 = importlib.util.spec_from_file_location('node29', os.path.join(HERE, '029_time_seq_exposure_context.py'))
node29 = importlib.util.module_from_spec(spec29); spec29.loader.exec_module(node29)
node22 = node29.node22
node20 = node29.node20
from data import load, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args()
    print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, 'fields=node22+time_seq_exposure_context_w40')

    pz = node22.per_user_zscore
    enc0, _ = encode(splits); users = enc0[target][2]
    member_seeds = [a.seed + 1000 * m for m in range(3)]

    base = node20.run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr,
                               epochs=a.epochs, seed=a.seed, device=a.device,
                               verbose=a.out is None)
    senc, sdim = node22.build_seq_rows(splits)
    seq_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(senc, sdim, target, a.split, ms, a.device, a.out is None,
                                      '022_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        seq_members.append(pz(p, users))
    seq = pz(np.mean(np.vstack(seq_members), axis=0), users)
    incumbent = 0.70 * pz(base, users) + 0.30 * seq

    hours = node20.load_hours(a.data_dir, splits, verbose=a.out is None)
    eenc, edim = node29.build_time_seq_exposure_rows(splits, hours)
    exp_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(eenc, edim, target, a.split, ms, a.device, a.out is None,
                                      '029_time_seq_exposure_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        exp_members.append(pz(p, users))
    expseq = pz(np.mean(np.vstack(exp_members), axis=0), users)
    scores = 0.60 * pz(incumbent, users) + 0.40 * expseq

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}')

if __name__ == '__main__':
    main()
