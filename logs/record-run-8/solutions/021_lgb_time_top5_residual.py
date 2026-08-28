"""FM rank ensemble plus time-aware residual LightGBM focused on top-5.

Refines node 20 by keeping the same FM ensemble and time residual features, but
sets LightGBM's lambdarank_truncation_level to 5 so tree gradients focus on the
same cutoff as nDCG@5 instead of spending capacity deeper in each user list.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import lightgbm as lgb


def _add_starter_to_path():
    here = os.path.abspath(os.path.dirname(__file__))
    for p in [os.path.join(os.getcwd(), 'kuairand-starter-kit'),
              os.path.join(here, '..', 'kuairand-starter-kit'),
              os.path.join(here, '..', '..', 'kuairand-starter-kit'),
              os.path.join(here, '..', '..', '..', 'kuairand-starter-kit')]:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            sys.path.insert(0, p); return
    sys.path.insert(0, 'kuairand-starter-kit')


_add_starter_to_path()
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


def _to_int(x, default=0):
    try:
        if x is None or x == '':
            return default
        return int(float(x))
    except Exception:
        return default


def _find_raw_logs(data_dir):
    for base in [data_dir, os.path.join(data_dir, 'data'), os.path.dirname(data_dir)]:
        fs = [os.path.join(base, 'log_standard_4_08_to_4_21_pure.csv'),
              os.path.join(base, 'log_standard_4_22_to_5_08_pure.csv')]
        if all(os.path.exists(f) for f in fs):
            return fs
    return [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
            os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]


def _read_raw_date_hour(data_dir):
    recs, dc_seen, hc_seen = [], None, None
    for fp in _find_raw_logs(data_dir):
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            dc = next((c for c in ['date', 'request_date', 'upload_date'] if c in fields), None)
            hc = next((c for c in ['hourmin', 'hour_min', 'time', 'request_time'] if c in fields), None)
            dc_seen = dc_seen or dc; hc_seen = hc_seen or hc
            for r in reader:
                d = _to_int(r.get(dc, 0)) if dc else 0
                hm = _to_int(r.get(hc, 0)) if hc else 0
                if hm > 2359:
                    hm %= 10000
                recs.append((d, hm))
    print(f"raw time rows={len(recs):,d}; date_col={dc_seen}; hour_col={hc_seen}")
    return recs


def _date_ord(yyyymmdd):
    x = int(yyyymmdd); return (((x // 100) % 100) - 4) * 31 + (x % 100)


def build_time_feature_maps(data_dir, splits):
    raw = _read_raw_date_hour(data_dir)
    out = {}
    for sp, rows in splits.items():
        dates = set(_to_int(r[0]) for r in rows)
        sel = [(d, hm) for d, hm in raw if d in dates]
        if len(sel) < len(rows):
            print(f"time alignment {sp}: only {len(sel):,d}/{len(rows):,d}; falling back for missing rows")
            sel = sel + [(_to_int(r[0]), 0) for r in rows[len(sel):]]
        else:
            sel = sel[:len(rows)]
        out[sp] = sel
        print(f"time alignment {sp}: {min(len(sel), len(rows)):,d}/{len(rows):,d}")
    return out


def time_features_for_rows(rows, raw_time_rows):
    z = np.zeros((len(rows), 7), dtype=np.float32)
    for i, r in enumerate(rows):
        if i < len(raw_time_rows):
            d0, hm = raw_time_rows[i]; d = d0 if d0 else _to_int(r[0])
        else:
            d, hm = _to_int(r[0]), 0
        hour = max(0, min(23, hm // 100)); minute = max(0, min(59, hm % 100))
        tab = _to_int(r[4]); dord = _date_ord(d)
        z[i, 0] = hour; z[i, 1] = hour // 4; z[i, 2] = tab * 24 + hour; z[i, 3] = dord
        z[i, 4] = dord / 40.0; z[i, 5] = (hour * 60 + minute) / 1440.0; z[i, 6] = (dord % 7) / 6.0
    return z


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval(); out = []
        for i in range(0, len(X), bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)


def build_user_groups(y, users):
    pos, neg = defaultdict(list), defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        (pos if yy > 0.5 else neg)[uu].append(i)
    return [(np.asarray(p, dtype=np.int64), np.asarray(neg[u], dtype=np.int64)) for u, p in pos.items() if neg.get(u)]


def sample_pairs(groups, rng):
    ps, ns = [], []
    for p, n in groups:
        ps.append(p); ns.append(rng.choice(n, size=len(p), replace=True))
    pi, ni = np.concatenate(ps), np.concatenate(ns)
    o = rng.permutation(len(pi))
    return pi[o], ni[o]


def train_one_fm(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                 patience=4, seed=0, device='cpu', verbose=True,
                 bce_weight=0.15, tag=''):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    groups = build_user_groups(ytr, utr)
    if verbose:
        print(f"{tag} BPR eligible users={len(groups):,d}, sampled pairs/epoch={sum(len(p) for p, _ in groups):,d}")
    torch.manual_seed(seed)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); pos_idx, neg_idx = sample_pairs(groups, rng); model.train(); losses = []
        for i in range(0, len(pos_idx), bs):
            xp = Xtr_t[torch.from_numpy(pos_idx[i:i+bs])].to(device)
            xn = Xtr_t[torch.from_numpy(neg_idx[i:i+bs])].to(device)
            opt.zero_grad(set_to_none=True)
            sp, sn = model(xp), model(xn)
            bpr = -torch.nn.functional.logsigmoid(sp - sn).mean()
            bce = 0.5 * (torch.nn.functional.softplus(-sp).mean() + torch.nn.functional.softplus(sn).mean())
            loss = bpr + bce_weight * bce; loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0; best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  {tag} early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def user_percentile_scores(scores, users):
    out = np.zeros(len(scores), dtype=np.float64); g = defaultdict(list)
    for i, u in enumerate(users): g[u].append(i)
    for idx in g.values():
        idx = np.asarray(idx, dtype=np.int64); n = len(idx)
        if n == 1:
            out[idx[0]] = 0.0; continue
        order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64); ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
        out[idx] = ranks
    return out


def fm_rank_ensemble(models, X, users, device='cpu'):
    pred = np.zeros(len(X), dtype=np.float64)
    for m in models:
        pred += user_percentile_scores(m.predict(X, device=device), users)
    return pred / len(models)


def make_stat_maps(rows):
    gctr = sum(float(r[6]) for r in rows) / max(len(rows), 1)
    names = ['video', 'author', 'tab', 'dur', 'author_tab']
    sums = {k: defaultdict(float) for k in names}; cnts = {k: defaultdict(int) for k in names}
    for r in rows:
        y = float(r[6]); dur = int(min(20, max(0, int(r[5]) // 5000)))
        keys = {'video': r[2], 'author': r[3], 'tab': r[4], 'dur': dur, 'author_tab': (r[3], r[4])}
        for k, key in keys.items(): sums[k][key] += y; cnts[k][key] += 1
    return gctr, sums, cnts


def stats_for_rows(rows, gctr, sums, cnts):
    feats = np.zeros((len(rows), 11), dtype=np.float32); names = ['video', 'author', 'tab', 'dur', 'author_tab']
    for i, r in enumerate(rows):
        dur = int(min(20, max(0, int(r[5]) // 5000)))
        vals = []
        for name, key in zip(names, [r[2], r[3], r[4], dur, (r[3], r[4])]):
            c = cnts[name].get(key, 0); s = sums[name].get(key, 0.0)
            vals += [np.log1p(c), (s + 20.0 * gctr) / (c + 20.0)]
        feats[i, :10] = vals; feats[i, 10] = np.log1p(max(float(r[5]), 0.0))
    return feats


def lgb_matrix(enc_X, rows, gctr, sums, cnts, fm_rank, time_feat):
    return np.hstack([enc_X[:, 1:].astype(np.int32), time_feat.astype(np.float32),
                      stats_for_rows(rows, gctr, sums, cnts), fm_rank.reshape(-1, 1).astype(np.float32)]).astype(np.float32)


def sort_groups_by_user(X, y, users, init_score):
    ua = np.asarray(users); order = np.argsort(ua, kind='mergesort'); su = ua[order]
    group, last, cnt = [], None, 0
    for u in su:
        if last is None or u == last: cnt += 1
        else: group.append(cnt); cnt = 1
        last = u
    if cnt: group.append(cnt)
    return X[order], y[order], group, init_score[order]


def train_lgb_residual_ranker(enc, splits, fm_train_rank, fm_valid_rank, time_maps, seed=0, verbose=True):
    gctr, sums, cnts = make_stat_maps(splits['train'])
    Xtr_e, ytr, utr = enc['train']; Xva_e, yva, uva = enc['valid']
    Xtr = lgb_matrix(Xtr_e, splits['train'], gctr, sums, cnts, fm_train_rank,
                     time_features_for_rows(splits['train'], time_maps['train']))
    Xva = lgb_matrix(Xva_e, splits['valid'], gctr, sums, cnts, fm_valid_rank,
                     time_features_for_rows(splits['valid'], time_maps['valid']))
    Xtr_s, ytr_s, gtr, init_tr = sort_groups_by_user(Xtr, ytr.astype(np.float32), utr, fm_train_rank.astype(np.float64))
    Xva_s, yva_s, gva, init_va = sort_groups_by_user(Xva, yva.astype(np.float32), uva, fm_valid_rank.astype(np.float64))
    dtr = lgb.Dataset(Xtr_s, label=ytr_s, group=gtr, init_score=init_tr,
                      categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7], free_raw_data=False)
    dva = lgb.Dataset(Xva_s, label=yva_s, group=gva, init_score=init_va,
                      categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7], reference=dtr, free_raw_data=False)
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5], 'label_gain': [0, 1],
        'lambdarank_truncation_level': 5,
        'learning_rate': 0.03, 'num_leaves': 31, 'min_data_in_leaf': 120,
        'feature_fraction': 0.90, 'bagging_fraction': 0.90, 'bagging_freq': 1,
        'lambda_l2': 5.0, 'verbosity': -1, 'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11, 'bagging_seed': int(seed) + 17, 'num_threads': 4,
    }
    callbacks = [lgb.early_stopping(25, verbose=verbose)]
    if not verbose: callbacks.append(lgb.log_evaluation(period=0))
    model = lgb.train(params, dtr, num_boost_round=250, valid_sets=[dva], valid_names=['valid'], callbacks=callbacks)
    return model, (gctr, sums, cnts)


def train_all(splits, data_dir, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True, bce_weight=0.15, n_models=5):
    enc, dim = encode(splits); fms = []
    for m in range(n_models):
        s = int(seed + 997 * m)
        fms.append(train_one_fm(enc, dim, k=k, lr=lr, epochs=epochs, seed=s, device=device,
                                verbose=verbose, bce_weight=bce_weight, tag=f"m{m}/seed{s}"))
    Xtr, _, utr = enc['train']; Xva, _, uva = enc['valid']
    fm_tr = fm_rank_ensemble(fms, Xtr, utr, device=device); fm_va = fm_rank_ensemble(fms, Xva, uva, device=device)
    time_maps = build_time_feature_maps(data_dir, splits)
    lgb_model, stat_pack = train_lgb_residual_ranker(enc, splits, fm_tr, fm_va, time_maps, seed=seed + 4242, verbose=verbose)
    return fms, lgb_model, stat_pack, time_maps, enc


def predict_residual_blend(fms, lgb_model, stat_pack, time_maps, enc_X, rows, users, split_name, device='cpu', fm_weight=0.50):
    fm_rank = fm_rank_ensemble(fms, enc_X, users, device=device)
    gctr, sums, cnts = stat_pack
    Xl = lgb_matrix(enc_X, rows, gctr, sums, cnts, fm_rank, time_features_for_rows(rows, time_maps.get(split_name, [])))
    delta = lgb_model.predict(Xl, num_iteration=lgb_model.best_iteration)
    corr = user_percentile_scores(fm_rank + delta, users)
    return fm_weight * fm_rank + (1.0 - fm_weight) * corr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--bce_weight', type=float, default=0.15); ap.add_argument('--n_models', type=int, default=5)
    ap.add_argument('--fm_weight', type=float, default=0.50)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}, residual_lgb_time_top5, fm_models={a.n_models}, fm_weight={a.fm_weight}")
    fms, lgb_model, stat_pack, time_maps, enc = train_all(
        splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device,
        verbose=a.out is None, bce_weight=a.bce_weight, n_models=a.n_models)
    X, y, users = enc[target]
    scores = predict_residual_blend(fms, lgb_model, stat_pack, time_maps, X, splits[target], users, target,
                                    device=a.device, fm_weight=a.fm_weight)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== lgb_time_top5_residual (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, predict_residual_blend(fms, lgb_model, stat_pack, time_maps, Xs, splits[sp], us, sp,
                                                            device=a.device, fm_weight=a.fm_weight))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
