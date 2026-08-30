"""Improve node 25: make the time-aware sequence blend user-history dependent.

Node25's 30% time-aware sequence member improved, while node26's uniform 40%
overweighted it.  This keeps all member training/caches unchanged and gives the
new member less weight for cold users and a little more for users with enough
training history for the time/history buckets to be meaningful.
"""
import argparse, os, sys, importlib.util
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'kuairand-starter-kit'))
spec25 = importlib.util.spec_from_file_location('node25', os.path.join(HERE, '025_time_seq_context.py'))
node25 = importlib.util.module_from_spec(spec25); spec25.loader.exec_module(node25)
node22 = node25.node22
node20 = node25.node20
from data import load, encode


def history_weight(counts, users):
    """Per-row final weight for the node25 time-aware sequence member.

    Keep the global average near node25's successful 0.30, but avoid letting a
    history-heavy member dominate users for whom all prior-count fields are cold.
    """
    w = np.empty(len(users), dtype=np.float64)
    for i, u in enumerate(users):
        c = counts.get(str(u), 0)
        if c <= 1:
            w[i] = 0.18
        elif c <= 5:
            w[i] = 0.24
        elif c <= 20:
            w[i] = 0.29
        elif c <= 80:
            w[i] = 0.31
        else:
            w[i] = 0.33
    return w


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
    print({k: len(v) for k, v in splits.items()}, 'fields=node25_adaptive_tseq_weight')

    enc0, _ = encode(splits); users = enc0[target][2]
    pz = node25.per_user_zscore
    member_seeds = [a.seed + 1000 * m for m in range(3)]

    # Recreate the node22 incumbent exactly: node20 base + unchanged sequence member.
    base = node20.run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr,
                               epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    senc, sdim = node22.build_seq_rows(splits)
    seq_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(senc, sdim, target, a.split, ms, a.device, a.out is None,
                                      '022_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        seq_members.append(pz(p, users))
    seq = pz(np.mean(np.vstack(seq_members), axis=0), users)
    incumbent = 0.70 * pz(base, users) + 0.30 * seq

    # Unchanged node25 time-aware sequence member.
    hours = node20.load_hours(a.data_dir, splits, verbose=a.out is None)
    tsenc, tsdim = node25.build_time_seq_rows(splits, hours)
    tseq_members = []
    for ms in member_seeds:
        p = node20.cached_predictions(tsenc, tsdim, target, a.split, ms, a.device, a.out is None,
                                      '025_time_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        tseq_members.append(pz(p, users))
    tseq = pz(np.mean(np.vstack(tseq_members), axis=0), users)

    counts = defaultdict(int)
    for r in splits['train']:
        counts[str(r[1])] += 1
    w = history_weight(counts, users)
    incz = pz(incumbent, users)
    scores = (1.0 - w) * incz + w * tseq

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}')

if __name__ == '__main__':
    main()
