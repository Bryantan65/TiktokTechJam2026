"""Full-strength ensemble of independent node-20 rankers.

Rather than adding a weak low-weight member, train three complete node-20 models
with different seeds and average their within-user standardized scores.  The
metric is per-user ranking, so per-user centering/scaling makes the combination
focus on ranking disagreements and removes seed-to-seed score-scale wobble.
"""
import argparse
import importlib.util
import os
import sys
from collections import defaultdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, '020_lambdamart_residual.py')
spec = importlib.util.spec_from_file_location('node20', PARENT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def user_z(scores, users):
    scores = np.asarray(scores, dtype=np.float32)
    out = np.empty_like(scores, dtype=np.float32)
    mp = defaultdict(list)
    for i, u in enumerate(users):
        mp[u].append(i)
    for idxs in mp.values():
        idx = np.asarray(idxs, dtype=np.int64)
        s = scores[idx]
        sd = float(s.std())
        if sd < 1e-6:
            out[idx] = s - float(s.mean())
        else:
            out[idx] = (s - float(s.mean())) / sd
    return out


def predict_one(splits, data_dir, target, seed, k, lr, epochs, device):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    model, resid, enc, dlogs, calfeats = m.run(
        splits, data_dir, seed=int(seed), k=k, lr=lr, epochs=epochs, device=device, verbose=False
    )
    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=device)
    scores = resid.predict(base, u, X, calfeats[target]).astype(np.float32)
    return user_z(scores, u), u


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = m.load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, "node20 x3 seed ensemble, per-user z average")

    # Non-overlapping internal seeds for each harness seed; this preserves seed dependence
    # while making each submitted prediction a real ensemble.
    internal = [int(a.seed) * 100 + 11, int(a.seed) * 100 + 37, int(a.seed) * 100 + 73]
    preds = []
    users_ref = None
    for j, s in enumerate(internal, 1):
        print(f"ensemble member {j}/3 seed={s}")
        p, users = predict_one(splits, a.data_dir, target, s, a.k, a.lr, a.epochs, a.device)
        preds.append(p)
        if users_ref is None:
            users_ref = users
    scores = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float64)

    if a.out:
        np.save(a.out, scores)
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        # Do not compute metrics in harness mode.  This branch is for local smoke checks only.
        print(scores[:10])
