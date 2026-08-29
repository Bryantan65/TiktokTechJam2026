"""Node 20 refinement: user-normalized LambdaMART residual blend.

This reuses the node-20 neural training code, but replaces the post-hoc residual
combiner.  LambdaMART is still trained on frozen train scores, while prediction
centers/scales its contribution separately inside each user group, matching the
within-user metrics and preventing global between-user score variance from
setting the blend strength.
"""
import argparse, importlib.util, os, sys
from collections import defaultdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, '020_lambdamart_residual.py')
spec = importlib.util.spec_from_file_location('node20', PARENT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class UserNormLambdaResidual:
    def __init__(self, seed=0, weight=0.30):
        self.seed = seed
        self.weight = float(weight)
        self.model = None
        self.global_raw_mean = 0.0
        self.global_raw_std = 1.0
        self.global_base_std = 1.0

    def fit(self, base, y, users, X, calfeat):
        import lightgbm as lgb
        F = m.make_lgb_features(base, users, X, calfeat)
        order = np.argsort(users, kind='mergesort')
        us = np.asarray(users)[order]
        _, counts = np.unique(us, return_counts=True)
        Fs = F[order]
        ys = y[order].astype(np.int32)
        self.model = lgb.LGBMRanker(
            objective='lambdarank', metric='ndcg', eval_at=[5], label_gain=[0, 1],
            n_estimators=90, learning_rate=0.045, num_leaves=31, max_depth=-1,
            min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.90, reg_lambda=1.5, random_state=self.seed,
            n_jobs=2, verbosity=-1)
        self.model.fit(Fs, ys, group=counts.tolist(), categorical_feature=[3,4,5,6,7,8,9,10])
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        self.global_raw_mean = float(raw.mean())
        self.global_raw_std = float(raw.std() + 1e-6)
        self.global_base_std = float(np.std(base) + 1e-6)
        print(f"UserNormLambdaResidual: raw_std={self.global_raw_std:.5f} base_std={self.global_base_std:.5f} weight={self.weight:.3f}")
        return self

    def predict(self, base, users, X, calfeat):
        if self.model is None:
            return base
        F = m.make_lgb_features(base, users, X, calfeat)
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        out = base.astype(np.float32).copy()
        mp = defaultdict(list)
        for i, u in enumerate(users):
            mp[u].append(i)
        for idxs in mp.values():
            idx = np.asarray(idxs, dtype=np.int64)
            if len(idx) <= 1:
                continue
            b = base[idx].astype(np.float32)
            r = raw[idx].astype(np.float32)
            bs = float(b.std())
            rs = float(r.std())
            if bs < 1e-6 or rs < 1e-6:
                # Fallback to the global scaling used by node 20 for degenerate groups.
                out[idx] = b + self.weight * self.global_base_std / self.global_raw_std * (r - self.global_raw_mean)
            else:
                out[idx] = b + self.weight * bs * (r - float(r.mean())) / (rs + 1e-6)
        return out


m.LambdaResidual = UserNormLambdaResidual

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = m.load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={m.FIELDS}+time aux+CWM+usernorm-LambdaMART-residual")
    model, resid, enc, dlogs, calfeats = m.run(
        splits, a.data_dir, seed=a.seed, k=a.k, lr=a.lr, epochs=a.epochs,
        device=a.device, verbose=a.out is None)
    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=a.device)
    scores = resid.predict(base, u, X, calfeats[target])
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        for sp in ('valid','test'):
            Xs, ys, us = enc[sp]
            b = model.predict(Xs, dlogs[sp], device=a.device)
            r = m.evaluate(us, ys, resid.predict(b, us, Xs, calfeats[sp]))
            print(sp, r)
