"""Three-seed ensemble with node20, history, and weekday residual rankers.

Node27 showed the history residual complements node20 when blended at full readable
weight.  Node26's weekday/date residual was weaker alone but targets a different
error source, so this keeps the same three neural seeds and adds a third
per-user standardized residual stream at 20% weight.
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
P26 = os.path.join(HERE, '026_weekday_lambdamart.py')

s20 = importlib.util.spec_from_file_location('node20', P20)
m20 = importlib.util.module_from_spec(s20)
s20.loader.exec_module(m20)
s23 = importlib.util.spec_from_file_location('node23', P23)
m23 = importlib.util.module_from_spec(s23)
s23.loader.exec_module(m23)
s26 = importlib.util.spec_from_file_location('node26', P26)
m26 = importlib.util.module_from_spec(s26)
s26.loader.exec_module(m26)


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


def predict_one(splits, data_dir, target, seed, k, lr, epochs, device,
                w20=0.50, whist=0.30, wday=0.20):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    model, resid20, enc, dlogs, calfeats = m20.run(
        splits, data_dir, seed=int(seed), k=k, lr=lr, epochs=epochs, device=device, verbose=False
    )
    Xtr, ytr, utr = enc['train']
    base_tr = model.predict(Xtr, dlogs['train'], device=device)

    resid_hist = m23.HistLambdaResidual(seed=int(seed), weight=0.45).fit(
        base_tr, ytr, utr, Xtr, calfeats['train'])
    resid_day = m26.WeekdayLambdaResidual(seed=int(seed), weight=0.35).fit(
        base_tr, ytr, utr, Xtr, calfeats['train'], splits['train'])

    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=device)
    s20 = resid20.predict(base, u, X, calfeats[target]).astype(np.float32)
    sh = resid_hist.predict(base, u, X, calfeats[target]).astype(np.float32)
    sd = resid_day.predict(base, u, X, calfeats[target], splits[target]).astype(np.float32)
    blended = w20 * user_z(s20, u) + whist * user_z(sh, u) + wday * user_z(sd, u)
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
    ap.add_argument('--w20', type=float, default=0.50)
    ap.add_argument('--whist', type=float, default=0.30)
    ap.add_argument('--wday', type=float, default=0.20)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = m20.load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()},
          f"node20 x3 + hist/day residual blend w=({a.w20},{a.whist},{a.wday})")

    internal = [int(a.seed) * 100 + 11, int(a.seed) * 100 + 37, int(a.seed) * 100 + 73]
    preds = []
    for j, s in enumerate(internal, 1):
        print(f"ensemble member {j}/3 seed={s}")
        p, _users = predict_one(splits, a.data_dir, target, s, a.k, a.lr, a.epochs, a.device,
                                w20=a.w20, whist=a.whist, wday=a.wday)
        preds.append(p)
    scores = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float64)

    if a.out:
        np.save(a.out, scores)
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(scores[:10])
