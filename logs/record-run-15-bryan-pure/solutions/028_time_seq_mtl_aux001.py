"""Improve/debug node 24: auxiliary MTL on the time-aware sequence member.

Node24's auxiliary-feedback member hurt when applied to the plain sequence rows
with aux_lambda=0.05.  This keeps the successful node25 blend structure but
replaces the 30% time-sequence BPR member by a time-sequence MTL member with a
much smaller aux loss (0.01), so the auxiliary raw feedback can regularise
rather than dominate the ranking objective.
"""
import argparse, os, sys, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'kuairand-starter-kit'))

spec25 = importlib.util.spec_from_file_location('node25', os.path.join(HERE, '025_time_seq_context.py'))
node25 = importlib.util.module_from_spec(spec25); spec25.loader.exec_module(node25)
node22 = node25.node22
node20 = node25.node20
spec24 = importlib.util.spec_from_file_location('node24', os.path.join(HERE, '024_multitask_seq_aux.py'))
node24 = importlib.util.module_from_spec(spec24); spec24.loader.exec_module(node24)
from data import load, encode


def cached_time_mtl_predictions(enc, dim, aux, target, split_name, seed, device, verbose,
                                k=16, lr=0.001, epochs=50):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'028_time_seq_mtl_aux001_member_seed{seed}_{split_name}_{target}.npy')
    if os.path.isfile(path):
        if verbose:
            print(f'loading cached member: {path}')
        return np.load(path)
    model = node24.train_mtl_member(enc, dim, aux, k=k, lr=lr, epochs=epochs, seed=seed,
                                    device=device, verbose=verbose, nneg=3, aux_lambda=0.01)
    preds = model.predict(enc[target][0], device=device).astype(np.float64)
    np.save(path, preds)
    return preds


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
    torch.manual_seed(a.seed)
    print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, 'fields=node22+time_seq_mtl_aux001')

    pz = node22.per_user_zscore
    enc0, _ = encode(splits)
    users = enc0[target][2]
    member_seeds = [a.seed + 1000 * m for m in range(3)]

    # Reconstruct the node22 incumbent exactly: node20 base + node22 learned sequence member.
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

    # Train the debugged MTL variant on the same time-aware sequence rows as node25.
    hours = node20.load_hours(a.data_dir, splits, verbose=a.out is None)
    tsenc, tsdim = node25.build_time_seq_rows(splits, hours)
    aux, aux_names = node24.load_aux_targets(a.data_dir, splits, verbose=a.out is None)
    mtl_members = []
    for ms in member_seeds:
        p = cached_time_mtl_predictions(tsenc, tsdim, aux, target, a.split, ms, a.device,
                                        a.out is None, k=a.k, lr=a.lr, epochs=a.epochs)
        mtl_members.append(pz(p, users))
    time_mtl = pz(np.mean(np.vstack(mtl_members), axis=0), users)

    scores = 0.70 * pz(incumbent, users) + 0.30 * time_mtl
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}; aux={aux_names}')

if __name__ == '__main__':
    main()
