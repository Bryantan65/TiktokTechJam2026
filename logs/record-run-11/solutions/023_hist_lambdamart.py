"""History target-encoding LambdaMART residual on top of the node-20 neural stack.

This revisits the history direction with explicit train-only collaborative memory
features (video/author/user-author/user-video CTR and counts, with leave-one-out
features on train) instead of coarse history buckets inside the FM.
"""
import argparse, importlib.util, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, '020_lambdamart_residual.py')
spec = importlib.util.spec_from_file_location('node20', PARENT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HistLambdaResidual:
    def __init__(self, seed=0, weight=0.45, alpha=20.0):
        self.seed = int(seed)
        self.weight = float(weight)
        self.alpha = float(alpha)
        self.model = None
        self.scale = 0.0
        self.mean = 0.0
        self.global_rate = 0.0
        self.stats = {}

    def _build_stats(self, X, y):
        y = y.astype(np.float32)
        self.global_rate = float(y.mean())
        specs = {
            'vid': X[:, 1],
            'author': X[:, 2],
            'tab': X[:, 3],
            'dur': X[:, 4],
            'u_vid': list(zip(X[:, 0].tolist(), X[:, 1].tolist())),
            'u_author': list(zip(X[:, 0].tolist(), X[:, 2].tolist())),
            'u_tab': list(zip(X[:, 0].tolist(), X[:, 3].tolist())),
        }
        for name, keys in specs.items():
            d = {}
            for k, yy in zip(keys, y):
                if k in d:
                    c, s = d[k]; d[k] = (c + 1, s + float(yy))
                else:
                    d[k] = (1, float(yy))
            self.stats[name] = d

    def _rate_count(self, name, keys, y=None, loo=False):
        d = self.stats[name]
        n = len(keys)
        rate = np.empty(n, dtype=np.float32)
        lcnt = np.empty(n, dtype=np.float32)
        g = self.global_rate
        a = self.alpha
        if loo and y is not None:
            for i, (k, yy) in enumerate(zip(keys, y)):
                c, s = d.get(k, (0, 0.0))
                c -= 1; s -= float(yy)
                if c < 0:
                    c = 0; s = 0.0
                rate[i] = (s + a * g) / (c + a)
                lcnt[i] = np.log1p(c)
        else:
            for i, k in enumerate(keys):
                c, s = d.get(k, (0, 0.0))
                rate[i] = (s + a * g) / (c + a)
                lcnt[i] = np.log1p(c)
        return rate, lcnt

    def _features(self, base, users, X, calfeat, y=None, train=False):
        base = base.astype(np.float32)
        n = len(base)
        z = np.zeros(n, dtype=np.float32)
        rk = np.zeros(n, dtype=np.float32)
        mp = defaultdict(list)
        for i, u in enumerate(users):
            mp[u].append(i)
        for idxs in mp.values():
            idx = np.asarray(idxs, dtype=np.int64)
            b = base[idx]
            z[idx] = (b - float(b.mean())) / (float(b.std()) + 1e-6)
            o = np.argsort(-b, kind='mergesort')
            rr = np.empty(len(idx), dtype=np.float32)
            rr[o] = np.arange(len(idx), dtype=np.float32) / max(len(idx) - 1, 1)
            rk[idx] = rr

        loo = bool(train and y is not None)
        cols = [base, z, rk]
        key_specs = [
            ('vid', X[:, 1]),
            ('author', X[:, 2]),
            ('tab', X[:, 3]),
            ('dur', X[:, 4]),
            ('u_vid', list(zip(X[:, 0].tolist(), X[:, 1].tolist()))),
            ('u_author', list(zip(X[:, 0].tolist(), X[:, 2].tolist()))),
            ('u_tab', list(zip(X[:, 0].tolist(), X[:, 3].tolist()))),
        ]
        for name, keys in key_specs:
            r, c = self._rate_count(name, keys, y=y, loo=loo)
            cols.extend([r, c])
        # Simple interactions that let the tree use memory mostly near the top of a user's slate.
        cols.append(z * cols[3])      # user-normalized base * video CTR
        cols.append(z * cols[11])     # user-normalized base * user-video CTR
        cols.append(z * cols[13])     # user-normalized base * user-author CTR
        cont = np.column_stack(cols).astype(np.float32)
        cats = np.column_stack([
            X[:, 1].astype(np.int32), X[:, 2].astype(np.int32), X[:, 3].astype(np.int32), X[:, 4].astype(np.int32),
            calfeat[:, 0].astype(np.int32), calfeat[:, 1].astype(np.int32), calfeat[:, 2].astype(np.int32), calfeat[:, 3].astype(np.int32)
        ]).astype(np.float32)
        return np.column_stack([cont, cats]).astype(np.float32)

    def fit(self, base, y, users, X, calfeat):
        import lightgbm as lgb
        self._build_stats(X, y)
        F = self._features(base, users, X, calfeat, y=y, train=True)
        order = np.argsort(users, kind='mergesort')
        us = np.asarray(users)[order]
        _, counts = np.unique(us, return_counts=True)
        Fs = F[order]
        ys = y[order].astype(np.int32)
        n_cont = F.shape[1] - 8
        cat = list(range(n_cont, F.shape[1]))
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
        print(f"HistLambdaResidual: raw_std={sr:.5f} base_std={sb:.5f} scale={self.scale:.5f} global={self.global_rate:.4f}")
        return self

    def predict(self, base, users, X, calfeat):
        F = self._features(base, users, X, calfeat, y=None, train=False)
        raw = self.model.predict(F, num_iteration=self.model.best_iteration_).astype(np.float32)
        return base + self.scale * (raw - self.mean)


def run(splits, data_dir, seed=0, k=16, lr=0.001, epochs=40, device='cpu', verbose=True):
    # Reuse node-20 neural training exactly, but discard its coarse residual and
    # fit the history-aware residual in its place.
    model, _old_resid, enc, dlogs, calfeats = m.run(
        splits, data_dir, seed=seed, k=k, lr=lr, epochs=epochs, device=device, verbose=verbose)
    Xtr, ytr, utr = enc['train']
    base_tr = model.predict(Xtr, dlogs['train'], device=device)
    resid = HistLambdaResidual(seed=seed, weight=0.45).fit(base_tr, ytr, utr, Xtr, calfeats['train'])
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
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = m.load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={m.FIELDS}+time aux+CWM+hist-LambdaMART")
    model, resid, enc, dlogs, calfeats = run(splits, a.data_dir, seed=a.seed, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=a.out is None)
    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=a.device)
    scores = resid.predict(base, u, X, calfeats[target])
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                b = model.predict(Xs, dlogs[sp], device=a.device)
                print(sp, m.evaluate(us, ys, resid.predict(b, us, Xs, calfeats[sp])))
