"""FM + LambdaMART blend for within-user ranking.

A new model family is tested at readable weight: LightGBM LambdaRank is trained
with user_id as the query group and historical train-only target statistics as
features, then blended 50/50 with the stable BCE FM after per-user z-scoring.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


# ----------------------------- FM baseline member --------------------------
class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def train_fm_predict(splits, target, seed=0, device='cpu'):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, _, _ = enc[target]

    model = TorchFM(dim, k=16, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': 1e-6},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    bs = 8192
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        model.train()
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if va['primary'] > best + 1e-5:
            best = va['primary']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    return model.predict(Xte, device=device)


# ----------------------------- LightGBM features ---------------------------
def rows_to_columns(rows):
    n = len(rows)
    cols = {
        'date': np.empty(n, dtype=np.int32),
        'user': np.empty(n, dtype=object),
        'video': np.empty(n, dtype=object),
        'author': np.empty(n, dtype=object),
        'tab': np.empty(n, dtype=np.int32),
        'duration': np.empty(n, dtype=np.float32),
        'label': np.empty(n, dtype=np.float32),
    }
    for i, r in enumerate(rows):
        cols['date'][i] = int(r[0])
        cols['user'][i] = r[1]
        cols['video'][i] = r[2]
        cols['author'][i] = r[3]
        cols['tab'][i] = int(r[4])
        cols['duration'][i] = float(r[5])
        cols['label'][i] = float(r[6])
    return cols


def fit_map(values):
    mp = {}
    nxt = 1  # 0 is unknown
    for v in values:
        if v not in mp:
            mp[v] = nxt
            nxt += 1
    return mp


def apply_map(values, mp):
    out = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        out[i] = mp.get(v, 0)
    return out


def make_bucket_edges(x, nb=20):
    qs = np.linspace(0, 1, nb + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs))
    return edges.astype(np.float32)


def apply_bucket(x, edges):
    return np.searchsorted(edges, x, side='right').astype(np.int32) + 1


def build_stat_maps(keys, y):
    cnt = defaultdict(int)
    sm = defaultdict(float)
    for k, yy in zip(keys, y):
        cnt[k] += 1
        sm[k] += float(yy)
    return cnt, sm


def stat_values(keys_train, keys_target, ytrain, target_is_train, alpha=20.0):
    """Smoothed mean/count; leave-one-out for train rows, train-only for others."""
    g = float(np.mean(ytrain))
    cnt, sm = build_stat_maps(keys_train, ytrain)
    mean = np.empty(len(keys_target), dtype=np.float32)
    lcnt = np.empty(len(keys_target), dtype=np.float32)
    if target_is_train:
        for i, (k, yy) in enumerate(zip(keys_target, ytrain)):
            c = cnt.get(k, 0) - 1
            s = sm.get(k, 0.0) - float(yy)
            mean[i] = (s + alpha * g) / (c + alpha)
            lcnt[i] = np.log1p(max(c, 0))
    else:
        for i, k in enumerate(keys_target):
            c = cnt.get(k, 0)
            s = sm.get(k, 0.0)
            mean[i] = (s + alpha * g) / (c + alpha)
            lcnt[i] = np.log1p(c)
    return mean, lcnt


def pair_keys(a, b):
    return list(zip(a, b))


def make_lgb_features(splits):
    cols = {sp: rows_to_columns(rows) for sp, rows in splits.items()}
    tr = cols['train']
    ytr = tr['label'].astype(np.float32)

    # Category encoders are fitted on train only.  Unseen valid/test categories
    # become 0 rather than leaking any label information.
    maps = {
        'user': fit_map(tr['user']),
        'video': fit_map(tr['video']),
        'author': fit_map(tr['author']),
        'date': fit_map(tr['date']),
        'tab': fit_map(tr['tab']),
    }
    dur_edges = make_bucket_edges(tr['duration'], 20)

    feats = {}
    users = {}
    labels = {}
    cat_idx = [0, 1, 2, 3, 4, 5]
    for sp, c in cols.items():
        is_train = (sp == 'train')
        user_c = apply_map(c['user'], maps['user'])
        video_c = apply_map(c['video'], maps['video'])
        author_c = apply_map(c['author'], maps['author'])
        date_c = apply_map(c['date'], maps['date'])
        tab_c = apply_map(c['tab'], maps['tab'])
        dur_b = apply_bucket(c['duration'], dur_edges)

        num_parts = [
            np.log1p(c['duration']).astype(np.float32),
            ((c['date'].astype(np.int32) % 100).astype(np.float32)),
        ]
        # Train-only historical relevance features.  User crossed stats are the
        # main addition over FM: they tell LambdaMART which authors/tabs/buckets
        # a user has previously watched without using valid/test labels.
        stat_defs = [
            tr['video'], c['video'],
            tr['author'], c['author'],
            tr['tab'], c['tab'],
            apply_bucket(tr['duration'], dur_edges), dur_b,
            pair_keys(tr['user'], tr['author']), pair_keys(c['user'], c['author']),
            pair_keys(tr['user'], tr['tab']), pair_keys(c['user'], c['tab']),
            pair_keys(tr['user'], apply_bucket(tr['duration'], dur_edges)), pair_keys(c['user'], dur_b),
        ]
        for j in range(0, len(stat_defs), 2):
            m, lc = stat_values(stat_defs[j], stat_defs[j + 1], ytr, is_train)
            num_parts.extend([m, lc])

        X = np.column_stack([
            user_c, video_c, author_c, date_c, tab_c, dur_b,
            *num_parts
        ]).astype(np.float32)
        feats[sp] = X
        users[sp] = user_c
        labels[sp] = c['label'].astype(np.float32)
    return feats, labels, users, cat_idx


def sort_by_user(X, y, user_code):
    order = np.argsort(user_code, kind='mergesort')
    Xs = X[order]
    ys = y[order]
    us = user_code[order]
    _, counts = np.unique(us, return_counts=True)
    return Xs, ys, counts.tolist(), order


def train_lgbm_rank_predict(splits, target, seed=0):
    feats, labels, users, cat_idx = make_lgb_features(splits)
    Xtr, ytr, gtr, _ = sort_by_user(feats['train'], labels['train'], users['train'])
    Xva, yva, gva, _ = sort_by_user(feats['valid'], labels['valid'], users['valid'])
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, categorical_feature=cat_idx,
                         free_raw_data=True)
    dvalid = lgb.Dataset(Xva, label=yva, group=gva, categorical_feature=cat_idx,
                         reference=dtrain, free_raw_data=True)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5],
        'label_gain': [0, 1],
        'lambdarank_truncation_level': 5,
        'learning_rate': 0.045,
        'num_leaves': 63,
        'min_data_in_leaf': 120,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'lambda_l2': 2.0,
        'max_bin': 255,
        'num_threads': 4,
        'verbosity': -1,
        'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11,
        'bagging_seed': int(seed) + 23,
        'data_random_seed': int(seed) + 37,
        'force_col_wise': True,
    }
    booster = lgb.train(params, dtrain, num_boost_round=450, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(35, verbose=False)])
    return booster.predict(feats[target], num_iteration=booster.best_iteration)


# ----------------------------- blending ------------------------------------
def per_user_zscore(scores, user_ids):
    out = np.empty_like(scores, dtype=np.float64)
    by = defaultdict(list)
    for i, u in enumerate(user_ids):
        by[u].append(i)
    for idxs in by.values():
        idx = np.asarray(idxs, dtype=np.int64)
        s = scores[idx].astype(np.float64)
        sd = float(s.std())
        if sd < 1e-12:
            out[idx] = 0.0
        else:
            out[idx] = (s - float(s.mean())) / sd
    return out


def target_users_from_rows(rows):
    return np.asarray([r[1] for r in rows], dtype=object)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")

    os.makedirs('pred_cache', exist_ok=True)
    cache_tag = f"005_{a.split}_seed{a.seed}"
    fm_path = os.path.join('pred_cache', cache_tag + '_fm_bce.npy')
    lgb_path = os.path.join('pred_cache', cache_tag + '_lgb_rank_stats.npy')

    if os.path.isfile(fm_path):
        fm_pred = np.load(fm_path)
    else:
        t0 = time.time()
        fm_pred = train_fm_predict(splits, target, seed=a.seed, device=a.device)
        np.save(fm_path, fm_pred)
        print(f"trained FM in {time.time() - t0:.1f}s")

    if os.path.isfile(lgb_path):
        lgb_pred = np.load(lgb_path)
    else:
        t0 = time.time()
        lgb_pred = train_lgbm_rank_predict(splits, target, seed=a.seed)
        np.save(lgb_path, lgb_pred)
        print(f"trained LGBM ranker in {time.time() - t0:.1f}s")

    target_users = target_users_from_rows(splits[target])
    fm_z = per_user_zscore(fm_pred.astype(np.float64), target_users)
    lgb_z = per_user_zscore(lgb_pred.astype(np.float64), target_users)
    scores = 0.50 * fm_z + 0.50 * lgb_z

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print("standalone mode: predictions built; harness computes official metrics")
