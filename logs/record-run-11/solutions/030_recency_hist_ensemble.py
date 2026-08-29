"""Three-seed node20 ensemble with recency-aware history residuals.

Node27 showed that a history/target-encoding LambdaMART residual complements the
node20 residual.  This variant keeps the same 60/40 blend but augments the
history residual with chronological prior-count/rate and days-since-seen features
for user-video/user-author/video/author, so repeated exposures and stale memories
can be ranked differently from global aggregate CTRs.
"""
import argparse
import importlib.util
import os
from collections import defaultdict
from datetime import date
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


def daynum(d):
    d = int(d)
    y, m, dd = d // 10000, (d // 100) % 100, d % 100
    return date(y, m, dd).toordinal()


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


class RecencyHistLambdaResidual(m23.HistLambdaResidual):
    def __init__(self, seed=0, weight=0.45, alpha=20.0):
        super().__init__(seed=seed, weight=weight, alpha=alpha)
        self.final_seq = {}

    def _seq_keys(self, X):
        return [
            ('u_vid', list(zip(X[:, 0].tolist(), X[:, 1].tolist()))),
            ('u_author', list(zip(X[:, 0].tolist(), X[:, 2].tolist()))),
            ('vid', X[:, 1].tolist()),
            ('author', X[:, 2].tolist()),
        ]

    def _seq_train_features(self, X, y, rows):
        n = len(y)
        days = np.asarray([daynum(r[0]) for r in rows], dtype=np.int32)
        cols = []
        self.final_seq = {}
        g = float(np.mean(y))
        a = float(self.alpha)
        for name, keys in self._seq_keys(X):
            d = {}
            rate = np.empty(n, dtype=np.float32)
            lcnt = np.empty(n, dtype=np.float32)
            ldays = np.empty(n, dtype=np.float32)
            seen = np.empty(n, dtype=np.float32)
            for i, (k, yy, dn) in enumerate(zip(keys, y, days)):
                c, s, last = d.get(k, (0, 0.0, -1))
                rate[i] = (s + a * g) / (c + a)
                lcnt[i] = np.log1p(c)
                if last < 0:
                    ldays[i] = np.log1p(99.0)
                    seen[i] = 0.0
                else:
                    ldays[i] = np.log1p(max(0, int(dn) - int(last)))
                    seen[i] = 1.0
                d[k] = (c + 1, s + float(yy), int(dn))
            self.final_seq[name] = d
            cols.extend([rate, lcnt, ldays, seen])
        return np.column_stack(cols).astype(np.float32)

    def _seq_pred_features(self, X, rows):
        n = X.shape[0]
        days = np.asarray([daynum(r[0]) for r in rows], dtype=np.int32)
        cols = []
        g = float(self.global_rate)
        a = float(self.alpha)
        for name, keys in self._seq_keys(X):
            # Copy so repeated target impressions can update recency/count without
            # mutating the fitted train state across repeated predict() calls.
            d = dict(self.final_seq.get(name, {}))
            rate = np.empty(n, dtype=np.float32)
            lcnt = np.empty(n, dtype=np.float32)
            ldays = np.empty(n, dtype=np.float32)
            seen = np.empty(n, dtype=np.float32)
            for i, (k, dn) in enumerate(zip(keys, days)):
                c, s, last = d.get(k, (0, 0.0, -1))
                rate[i] = (s + a * g) / (c + a)
                lcnt[i] = np.log1p(c)
                if last < 0:
                    ldays[i] = np.log1p(99.0)
                    seen[i] = 0.0
                else:
                    ldays[i] = np.log1p(max(0, int(dn) - int(last)))
                    seen[i] = 1.0
                # Labels are unavailable at prediction time; update only exposure
                # count and last-seen date, preserving the train positive evidence.
                d[k] = (c + 1, s, int(dn))
            cols.extend([rate, lcnt, ldays, seen])
        return np.column_stack(cols).astype(np.float32)

    def _merge_features(self, baseF, extra):
        # Parent history features end with 8 categorical columns.  Insert the new
        # continuous recency columns before those cats so categorical indices stay clear.
        return np.column_stack([baseF[:, :-8], extra, baseF[:, -8:]]).astype(np.float32)

    def fit(self, base, y, users, X, calfeat, train_rows):
        import lightgbm as lgb
        self._build_stats(X, y)
        baseF = self._features(base, users, X, calfeat, y=y, train=True)
        extra = self._seq_train_features(X, y, train_rows)
        F = self._merge_features(baseF, extra)
        order = np.argsort(users, kind='mergesort')
        us = np.asarray(users)[order]
        _, counts = np.unique(us, return_counts=True)
        Fs = F[order]
        ys = y[order].astype(np.int32)
        cat = list(range(F.shape[1] - 8, F.shape[1]))
        self.model = lgb.LGBMRanker(
            objective='lambdarank', metric='ndcg', eval_at=[5], label_gain=[0, 1],
            n_estimators=140, learning_rate=0.04, num_leaves=31, max_depth=-1,
            min_child_samples=90, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.90, reg_lambda=2.0, random_state=self.seed,
            n_jobs=2, verbosity=-1)
        self.model.fit(Fs, ys, group=counts.tolist(), categorical_feature=cat)
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        self.mean = float(raw.mean())
        sb = float(np.std(base)); sr = float(np.std(raw))
        self.scale = self.weight * sb / (sr + 1e-6)
        print(f"RecencyHistLambdaResidual: raw_std={sr:.5f} base_std={sb:.5f} scale={self.scale:.5f} global={self.global_rate:.4f}")
        return self

    def predict(self, base, users, X, calfeat, rows):
        baseF = self._features(base, users, X, calfeat, y=None, train=False)
        extra = self._seq_pred_features(X, rows)
        F = self._merge_features(baseF, extra)
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        return base + self.scale * (raw - self.mean)


def predict_one(splits, data_dir, target, seed, k, lr, epochs, device, hist_weight=0.40):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    model, resid20, enc, dlogs, calfeats = m20.run(
        splits, data_dir, seed=int(seed), k=k, lr=lr, epochs=epochs, device=device, verbose=False
    )
    Xtr, ytr, utr = enc['train']
    base_tr = model.predict(Xtr, dlogs['train'], device=device)
    resid_hist = RecencyHistLambdaResidual(seed=int(seed), weight=0.45).fit(
        base_tr, ytr, utr, Xtr, calfeats['train'], splits['train'])

    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=device)
    s20p = resid20.predict(base, u, X, calfeats[target]).astype(np.float32)
    shp = resid_hist.predict(base, u, X, calfeats[target], splits[target]).astype(np.float32)
    blended = (1.0 - hist_weight) * user_z(s20p, u) + hist_weight * user_z(shp, u)
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
    print({k: len(v) for k, v in splits.items()}, f"node20 x3 + recency-hist residual blend w={a.hist_weight}")

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
