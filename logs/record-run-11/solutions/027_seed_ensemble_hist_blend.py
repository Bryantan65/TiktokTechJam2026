"""Three-seed ensemble blending node20 and history-aware residuals.

Node25 reduced variance by averaging three independent node20 rankers.  Node23's
history/target-encoding residual was slightly weaker standalone but uses different
collaborative-memory features, so here each neural seed trains both residuals on
the same frozen base model and blends their per-user standardized rankings at a
readable 40% history weight before averaging seeds.
"""
import argparse
import importlib.util
import os
from collections import defaultdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P20 = os.path.join(HERE, '020_lambdamart_residual.py')
P23 = os.path.join(HERE, '023_hist_lambdamart.py')
s20 = importlib.util.spec_from_file_location('node20', P20)
m20 = importlib.util.module_from_spec(s20)
s20.loader.exec_module(m20)
s23 = importlib.util.spec_from_file_location('node23', P23)
m23 = importlib.util.module_from_spec(s23)
s23.loader.exec_module(m23)


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


def predict_one(splits, data_dir, target, seed, k, lr, epochs, device, hist_weight=0.40):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    model, resid20, enc, dlogs, calfeats = m20.run(
        splits, data_dir, seed=int(seed), k=k, lr=lr, epochs=epochs, device=device, verbose=False
    )
    Xtr, ytr, utr = enc['train']
    base_tr = model.predict(Xtr, dlogs['train'], device=device)
    resid_hist = m23.HistLambdaResidual(seed=int(seed), weight=0.45).fit(
        base_tr, ytr, utr, Xtr, calfeats['train'])

    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=device)
    s20 = resid20.predict(base, u, X, calfeats[target]).astype(np.float32)
    sh = resid_hist.predict(base, u, X, calfeats[target]).astype(np.float32)
    blended = (1.0 - hist_weight) * user_z(s20, u) + hist_weight * user_z(sh, u)
    return user_z(blended, u), u


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
    ap.add_argument('--hist_weight', type=float, default=0.40)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = m20.load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"node20 x3 + hist residual blend w={a.hist_weight}")

    internal = [int(a.seed) * 100 + 11, int(a.seed) * 100 + 37, int(a.seed) * 100 + 73]
    preds = []
    for j, s in enumerate(internal, 1):
        print(f"ensemble member {j}/3 seed={s}")
        p, _users = predict_one(splits, a.data_dir, target, s, a.k, a.lr, a.epochs, a.device, hist_weight=a.hist_weight)
        preds.append(p)
    scores = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float64)

    if a.out:
        np.save(a.out, scores)
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(scores[:10])
