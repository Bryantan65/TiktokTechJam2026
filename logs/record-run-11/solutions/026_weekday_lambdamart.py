"""Weekday-aware LambdaMART residual on top of node 20.

Node 20's residual uses base score, per-user rank/z, item ids, tab/hour/daypart and
duration, but not weekday/date.  The neural model has weekday embeddings already;
this tests whether the post-hoc LambdaRank stage can use weekday and simple
weekday interactions to correct top-5 ordering without changing the neural loss.
"""
import argparse
import importlib.util
import os
from collections import defaultdict
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, '020_lambdamart_residual.py')
spec = importlib.util.spec_from_file_location('node20', PARENT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def weekday(date):
    try:
        return datetime.strptime(str(int(date)), '%Y%m%d').weekday()
    except Exception:
        return 7


def day_number(date):
    try:
        dt = datetime.strptime(str(int(date)), '%Y%m%d')
        return dt.toordinal()
    except Exception:
        return 0


class WeekdayLambdaResidual:
    def __init__(self, seed=0, weight=0.35):
        self.seed = int(seed)
        self.weight = float(weight)
        self.model = None
        self.mean = 0.0
        self.scale = 0.0
        self.day0 = 0.0

    def _extra(self, dates, X, calfeat):
        wd = np.asarray([weekday(r[0] if isinstance(r, tuple) else r) for r in dates], dtype=np.int32)
        dn = np.asarray([day_number(r[0] if isinstance(r, tuple) else r) for r in dates], dtype=np.float32)
        if self.day0 == 0.0:
            self.day0 = float(dn[dn > 0].min()) if np.any(dn > 0) else 0.0
        drel = (dn - self.day0).astype(np.float32)
        hour = calfeat[:, 1].astype(np.int32)
        tab = X[:, 3].astype(np.int32)
        # continuous relative day plus categorical weekday interactions
        return np.column_stack([
            drel,
            wd.astype(np.float32),
            (wd * 25 + hour).astype(np.float32),
            (wd * 8 + tab).astype(np.float32),
        ]).astype(np.float32)

    def _features(self, base, users, X, calfeat, rows):
        F = m.make_lgb_features(base, users, X, calfeat)
        E = self._extra(rows, X, calfeat)
        return np.column_stack([F, E]).astype(np.float32)

    def fit(self, base, y, users, X, calfeat, rows):
        import lightgbm as lgb
        F = self._features(base, users, X, calfeat, rows)
        order = np.argsort(users, kind='mergesort')
        us = np.asarray(users)[order]
        _, counts = np.unique(us, return_counts=True)
        Fs = F[order]
        ys = y[order].astype(np.int32)
        # Node20 categorical columns are 3..10. Added: weekday, weekday*hour, weekday*tab.
        cat = [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
        self.model = lgb.LGBMRanker(
            objective='lambdarank', metric='ndcg', eval_at=[5], label_gain=[0, 1],
            n_estimators=110, learning_rate=0.04, num_leaves=31, max_depth=-1,
            min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.90, reg_lambda=1.8, random_state=self.seed,
            n_jobs=2, verbosity=-1)
        self.model.fit(Fs, ys, group=counts.tolist(), categorical_feature=cat)
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        self.mean = float(raw.mean())
        sb = float(np.std(base)); sr = float(np.std(raw))
        self.scale = self.weight * sb / (sr + 1e-6)
        print(f"WeekdayLambdaResidual: raw_std={sr:.5f} base_std={sb:.5f} scale={self.scale:.5f} day0={self.day0:.0f}")
        return self

    def predict(self, base, users, X, calfeat, rows):
        F = self._features(base, users, X, calfeat, rows)
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        return base + self.scale * (raw - self.mean)


def run(splits, data_dir, seed=0, k=16, lr=0.001, epochs=40, device='cpu', verbose=True):
    model, _old_resid, enc, dlogs, calfeats = m.run(
        splits, data_dir, seed=seed, k=k, lr=lr, epochs=epochs, device=device, verbose=verbose)
    Xtr, ytr, utr = enc['train']
    base_tr = model.predict(Xtr, dlogs['train'], device=device)
    resid = WeekdayLambdaResidual(seed=seed, weight=0.35).fit(
        base_tr, ytr, utr, Xtr, calfeats['train'], splits['train'])
    return model, resid, enc, dlogs, calfeats


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
    m.torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = m.load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={m.FIELDS}+time aux+CWM+weekday-LambdaMART")
    model, resid, enc, dlogs, calfeats = run(
        splits, a.data_dir, seed=a.seed, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=a.out is None)
    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=a.device)
    scores = resid.predict(base, u, X, calfeats[target], splits[target])
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                b = model.predict(Xs, dlogs[sp], device=a.device)
                print(sp, m.evaluate(us, ys, resid.predict(b, us, Xs, calfeats[sp], splits[sp])))
