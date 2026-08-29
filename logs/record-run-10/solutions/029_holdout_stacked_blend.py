"""Temporal holdout stacking on top of node 24's four members.

Train node-24 base models on an early slice of train, predict the last two train
calendar days, and fit a tiny LightGBM ranker that combines only per-user base
ranks/scores.  Then train the usual node-24 base models on all train and apply
that learned combiner to valid/test.  This tests learned combination rather than
another hand-set 4-way rank average.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, FIELDS
from evaluate import evaluate

EXT_FIELDS = FIELDS + ['dow', 'hour', 'tod4']


def yyyymmdd_to_dow(d):
    y = int(d) // 10000; m = (int(d) // 100) % 100; day = int(d) % 100
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    if m < 3: y -= 1
    return (y + y // 4 - y // 100 + y // 400 + t[m - 1] + day + 6) % 7


def dur_bucket(ms):
    try: ms = int(ms)
    except Exception: ms = 0
    if ms < 7000: return 0
    if ms < 15000: return 1
    if ms < 30000: return 2
    if ms < 60000: return 3
    return 4


def norm_int_str(x):
    s = str(x)
    try: return str(int(float(s)))
    except Exception: return s


def parse_hour(v):
    try: h = int(float(v))
    except Exception: return 0
    if h >= 100: h //= 100
    return max(0, min(23, h))


def raw_log_paths(data_dir):
    return [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
            os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]


def make_hour_lookup(splits, data_dir):
    q = defaultdict(deque); total = 0
    for path in raw_log_paths(data_dir):
        if not os.path.exists(path): continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f); cols = rdr.fieldnames or []
            hcol = 'hourmin' if 'hourmin' in cols else ('hour' if 'hour' in cols else None)
            dcol = 'date' if 'date' in cols else None
            if hcol is None or dcol is None: continue
            for r in rdr:
                key = (norm_int_str(r.get(dcol, '')), norm_int_str(r.get('user_id', '')),
                       norm_int_str(r.get('video_id', '')), norm_int_str(r.get('tab', '')))
                q[key].append(parse_hour(r.get(hcol, 0))); total += 1
    out = {}; miss = used = 0
    for name, rows in splits.items():
        arr = np.zeros(len(rows), dtype=np.int16)
        for i, r in enumerate(rows):
            key = (norm_int_str(r[0]), norm_int_str(r[1]), norm_int_str(r[2]), norm_int_str(r[4]))
            dq = q.get(key)
            if dq:
                arr[i] = dq.popleft(); used += 1
            else:
                miss += 1
        out[name] = arr
    print(f"loaded raw hourmin rows={total:,d}; matched={used:,d}; missing={miss:,d}")
    return out


def row_features(row, hour):
    return (row[1], row[2], row[3], row[4], dur_bucket(row[5]),
            yyyymmdd_to_dow(row[0]), int(hour), int(hour) // 4)


def encode_ext(splits, hour_lookup):
    vocabs = [{} for _ in EXT_FIELDS]
    for name, rows in splits.items():
        for r, h in zip(rows, hour_lookup[name]):
            for j, v in enumerate(row_features(r, h)):
                if v not in vocabs[j]: vocabs[j][v] = len(vocabs[j])
    offsets = np.cumsum([0] + [len(v) for v in vocabs[:-1]], dtype=np.int64)
    dim = int(offsets[-1] + len(vocabs[-1]))
    enc = {}; rawcat = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(EXT_FIELDS)), dtype=np.int64)
        C = np.empty((len(rows), len(EXT_FIELDS)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32); users = np.empty(len(rows), dtype=object)
        dur = np.empty(len(rows), dtype=np.float32)
        for i, (r, h) in enumerate(zip(rows, hour_lookup[name])):
            vals = row_features(r, h)
            for j, v in enumerate(vals):
                c = vocabs[j][v]; C[i, j] = c; X[i, j] = c + offsets[j]
            y[i] = float(r[6]); users[i] = r[1]
            try: dur[i] = float(r[5])
            except Exception: dur[i] = 0.0
        enc[name] = (X, y, users)
        rawcat[name] = np.column_stack([C, np.log1p(np.maximum(dur, 0.0)).astype(np.float32)]).astype(np.float32)
    return enc, rawcat, dim


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        return self.b + self.W[X].sum(1) + 0.5 * ((S * S).sum(1) - (E * E).sum((1, 2)))
    @torch.no_grad()
    def predict(self, X, bs=200000, device='cpu'):
        self.eval(); out = []
        for i in range(0, len(X), bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)


def train_bce(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xt = torch.from_numpy(Xtr.astype(np.int64)); yt = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed); best = -1; best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); order = rng.permutation(len(ytr)); model.train(); losses = []
        for i in range(0, len(order), bs):
            idx = torch.from_numpy(order[i:i+bs]); xb = Xt[idx].to(device); yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True); loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f"  BCE epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model


def build_user_pair_sources(y, users):
    buckets = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if u not in buckets: buckets[u] = [[], []]
        buckets[u][1 if yy > 0.5 else 0].append(i)
    pos_lists = []; neg_lists = []; weights = []
    for neg, pos in buckets.values():
        if pos and neg:
            pa = np.asarray(pos, dtype=np.int64); na = np.asarray(neg, dtype=np.int64)
            pos_lists.append(pa); neg_lists.append(na); weights.append(min(len(pa) * len(na), 2000))
    weights = np.asarray(weights, dtype=np.float64); weights /= weights.sum()
    return pos_lists, neg_lists, weights


def sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng):
    uids = rng.choice(len(pos_lists), size=n_pairs, replace=True, p=weights)
    pidx = np.empty(n_pairs, dtype=np.int64); nidx = np.empty(n_pairs, dtype=np.int64)
    for u in np.unique(uids):
        m = np.nonzero(uids == u)[0]; pos = pos_lists[u]; neg = neg_lists[u]
        pidx[m] = pos[rng.integers(0, len(pos), size=len(m))]
        nidx[m] = neg[rng.integers(0, len(neg), size=len(m))]
    return pidx, nidx


def train_bpr(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xt = torch.from_numpy(Xtr.astype(np.int64)); pos, neg, w = build_user_pair_sources(ytr, utr)
    n_pairs = max(1, int((ytr > 0.5).sum())); rng = np.random.default_rng(seed)
    best = -1; best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); pidx, nidx = sample_pairs(pos, neg, w, n_pairs, rng); order = rng.permutation(n_pairs)
        model.train(); losses = []
        for i in range(0, n_pairs, bs):
            sel = order[i:i+bs]
            xp = Xt[torch.from_numpy(pidx[sel])].to(device); xn = Xt[torch.from_numpy(nidx[sel])].to(device)
            opt.zero_grad(set_to_none=True); loss = F.softplus(-(model(xp) - model(xn))).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f"  BPR epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model


def user_sorted(X, y, users):
    order = np.argsort(users, kind='mergesort'); us = np.asarray(users, dtype=object)[order]
    group = []; s = 0
    while s < len(us):
        e = s + 1
        while e < len(us) and us[e] == us[s]: e += 1
        group.append(e - s); s = e
    return X[order], y[order], group, order


def train_lgb_ranker(rawcat, enc, objective='rank_xendcg', seed=0, verbose=True):
    Xtr, ytr, gtr, _ = user_sorted(rawcat['train'], enc['train'][1], enc['train'][2])
    Xva, yva, gva, _ = user_sorted(rawcat['valid'], enc['valid'][1], enc['valid'][2])
    dtrain = lgb.Dataset(Xtr, label=ytr.astype(np.int32), group=gtr, categorical_feature=list(range(len(EXT_FIELDS))), free_raw_data=False)
    dvalid = lgb.Dataset(Xva, label=yva.astype(np.int32), group=gva, categorical_feature=list(range(len(EXT_FIELDS))), reference=dtrain, free_raw_data=False)
    params = {
        'objective': objective, 'metric': 'ndcg', 'ndcg_eval_at': [5], 'label_gain': [0, 1],
        'learning_rate': 0.04, 'num_leaves': 63, 'min_data_in_leaf': 80,
        'feature_fraction': 0.85, 'bagging_fraction': 0.85, 'bagging_freq': 1,
        'lambda_l2': 1.0, 'max_cat_threshold': 64, 'verbosity': -1, 'num_threads': 4,
        'seed': int(seed), 'feature_fraction_seed': int(seed)+11, 'bagging_seed': int(seed)+17,
        'data_random_seed': int(seed)+23, 'force_col_wise': True,
    }
    callbacks = [lgb.early_stopping(30, verbose=verbose)]
    if verbose: callbacks.append(lgb.log_evaluation(20))
    return lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dvalid], valid_names=['valid'], callbacks=callbacks)


def per_user_rank_and_z(scores, users):
    scores = np.asarray(scores, dtype=np.float64); users = np.asarray(users)
    out_r = np.zeros(len(users), dtype=np.float32); out_z = np.zeros(len(users), dtype=np.float32)
    order = np.argsort(users, kind='mergesort'); s = 0
    while s < len(users):
        e = s + 1; u = users[order[s]]
        while e < len(users) and users[order[e]] == u: e += 1
        idx = order[s:e]; n = len(idx); vals = scores[idx]
        if n > 1:
            r = np.empty(n, dtype=np.float32); r[np.argsort(vals, kind='mergesort')] = np.linspace(0.0, 1.0, n, dtype=np.float32)
            out_r[idx] = r
            sd = float(vals.std())
            if sd > 1e-12: out_z[idx] = ((vals - vals.mean()) / sd).astype(np.float32)
        s = e
    return out_r, out_z


def stack_features(score_list, users):
    feats = []
    for sc in score_list:
        r, z = per_user_rank_and_z(sc, users)
        feats.append(r); feats.append(z)
    # include the original node-24 hand blend as an anchor feature
    anchor = 0.40 * feats[0] + 0.20 * feats[2] + 0.20 * feats[4] + 0.20 * feats[6]
    feats.append(anchor.astype(np.float32))
    return np.column_stack(feats).astype(np.float32)


def train_base(splits, data_dir, seed, k, lr, epochs, device, verbose):
    hour_lookup = make_hour_lookup(splits, data_dir)
    enc, rawcat, dim = encode_ext(splits, hour_lookup)
    bpr = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed, device=device, verbose=verbose)
    bce = train_bce(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed + 313, device=device, verbose=verbose)
    xendcg = train_lgb_ranker(rawcat, enc, objective='rank_xendcg', seed=seed + 1009, verbose=verbose)
    lambdarank = train_lgb_ranker(rawcat, enc, objective='lambdarank', seed=seed + 2003, verbose=verbose)
    return bpr, bce, xendcg, lambdarank, enc, rawcat


def predict_base(models, enc, rawcat, split, device):
    bpr, bce, xendcg, lambdarank = models
    X, y, users = enc[split]
    return [bpr.predict(X, device=device),
            xendcg.predict(rawcat[split], num_iteration=xendcg.best_iteration),
            lambdarank.predict(rawcat[split], num_iteration=lambdarank.best_iteration),
            bce.predict(X, device=device)], y, users


def train_combiner(Fh, yh, uh, seed=0):
    Xs, ys, gs, order = user_sorted(Fh, yh, uh)
    # Use early stopping on the same temporal holdout only to limit trees in this tiny combiner;
    # it sees no official valid/test labels.
    dtrain = lgb.Dataset(Xs, label=ys.astype(np.int32), group=gs, free_raw_data=False)
    params = {
        'objective': 'rank_xendcg', 'metric': 'ndcg', 'ndcg_eval_at': [5], 'label_gain': [0, 1],
        'learning_rate': 0.03, 'num_leaves': 15, 'min_data_in_leaf': 150,
        'feature_fraction': 1.0, 'bagging_fraction': 0.9, 'bagging_freq': 1,
        'lambda_l2': 5.0, 'verbosity': -1, 'num_threads': 4,
        'seed': int(seed), 'feature_fraction_seed': int(seed)+31, 'bagging_seed': int(seed)+37,
        'data_random_seed': int(seed)+41, 'force_col_wise': True,
    }
    return lgb.train(params, dtrain, num_boost_round=80)


def temporal_stack_split(full_splits):
    train_rows = list(full_splits['train'])
    dates = sorted({int(r[0]) for r in train_rows})
    cut_dates = set(dates[-2:]) if len(dates) >= 4 else {dates[-1]}
    early = [r for r in train_rows if int(r[0]) not in cut_dates]
    hold = [r for r in train_rows if int(r[0]) in cut_dates]
    out = {'train': early, 'valid': hold}
    print(f"stack split: early={len(early):,d}, holdout={len(hold):,d}, hold_dates={sorted(cut_dates)}")
    return out


def node24_rank_blend(score_list, users):
    Fm = stack_features(score_list, users)
    return Fm[:, -1].astype(np.float64)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args(); torch.manual_seed(a.seed)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={EXT_FIELDS}, temporal stacked node24 combiner")

    # Fit combiner from a train-only temporal holdout.
    stack_splits = temporal_stack_split(splits)
    ebpr, ebce, ex, el, eenc, eraw = train_base(stack_splits, a.data_dir, a.seed + 5000, a.k, a.lr, a.epochs, a.device, verbose=False)
    hscores, hy, hu = predict_base((ebpr, ebce, ex, el), eenc, eraw, 'valid', a.device)
    Fh = stack_features(hscores, hu)
    comb = train_combiner(Fh, hy, hu, seed=a.seed + 7000)

    # Train full node-24 base models and apply the learned combiner.
    bpr, bce, xendcg, lambdarank, enc, rawcat = train_base(splits, a.data_dir, a.seed, a.k, a.lr, a.epochs, a.device, verbose=a.out is None)
    scores, y, users = predict_base((bpr, bce, xendcg, lambdarank), enc, rawcat, target, a.device)
    Ft = stack_features(scores, users)
    pred = comb.predict(Ft, num_iteration=comb.best_iteration if comb.best_iteration else None)
    # Stable fallback: blend half learned combiner, half original node-24 rank anchor.
    final = 0.50 * per_user_rank_and_z(pred, users)[0].astype(np.float64) + 0.50 * Ft[:, -1].astype(np.float64)

    if a.out:
        np.save(a.out, final.astype(np.float64)); print(f"wrote {len(final):,d} predictions for split={a.split}")
    else:
        r = evaluate(users, y, final)
        print(f"GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
