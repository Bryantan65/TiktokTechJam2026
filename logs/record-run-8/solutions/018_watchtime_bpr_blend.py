"""Add a watch-time pairwise FM member to the FM/LGB rank ensemble.

The main label is binary, so the successful BPR model only learns positive >
negative.  This draft reads train-only play_time_ms from the raw logs and trains
an extra same-user pairwise FM on label + capped watch-ratio, then blends its
within-user ranks at 30% with the current residual ensemble.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import lightgbm as lgb


def _add_starter_to_path():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(os.getcwd(), 'kuairand-starter-kit'),
        os.path.join(here, '..', 'kuairand-starter-kit'),
        os.path.join(here, '..', '..', 'kuairand-starter-kit'),
        os.path.join(here, '..', '..', '..', 'kuairand-starter-kit'),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            sys.path.insert(0, p)
            return
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


def _to_float(x, default=0.0):
    try:
        if x is None or x == '':
            return default
        return float(x)
    except Exception:
        return default


def _get(row, names, default=''):
    for n in names:
        if n in row and row[n] != '':
            return row[n]
    return default


def _row_key_tuple(r):
    return (_to_int(r[0]), str(r[1]), str(r[2]), str(r[3]), _to_int(r[4]), _to_int(r[5]))


def _row_key_csv(rec):
    return (
        _to_int(_get(rec, ['date'])),
        str(_get(rec, ['user_id', 'userId'])),
        str(_get(rec, ['video_id', 'photo_id', 'videoId'])),
        str(_get(rec, ['author_id', 'authorId'])),
        _to_int(_get(rec, ['tab', 'tab_id'])),
        _to_int(_get(rec, ['duration_ms', 'duration', 'video_duration'])),
    )


def read_train_watch_targets(data_dir, train_rows):
    """Return a train-row target using only raw log watch time.

    Missing rows fall back to the binary label, so a path/schema mismatch does
    not crash the experiment; the printed hit rate diagnoses whether CWM signal
    was actually available.
    """
    files = [
        os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv'),
    ]
    raw = defaultdict(deque)
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for rec in reader:
                key = _row_key_csv(rec)
                play = _to_float(_get(rec, ['play_time_ms', 'playtime_ms', 'play_ms', 'play_time', 'watch_time_ms']), 0.0)
                # Keep raw value only; the binary label comes from data.load so
                # we never use validation/test labels or post-action features at prediction time.
                raw[key].append(play)
    out = np.zeros(len(train_rows), dtype=np.float32)
    hits = 0
    for i, r in enumerate(train_rows):
        y = float(r[6])
        key = _row_key_tuple(r)
        play = None
        if raw.get(key):
            play = raw[key].popleft()
            hits += 1
        dur = max(float(r[5]), 1.0)
        ratio = 0.0 if play is None else min(max(float(play) / dur, 0.0), 2.0)
        out[i] = y + 0.25 * ratio
    print(f"watch-time raw alignment hits={hits:,d}/{len(train_rows):,d}")
    return out


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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


def build_user_groups(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        (pos if yy > 0.5 else neg)[uu].append(i)
    groups = []
    for uu, p in pos.items():
        n = neg.get(uu)
        if n:
            groups.append((np.asarray(p, dtype=np.int64), np.asarray(n, dtype=np.int64)))
    return groups


def build_watch_groups(wtarget, users):
    byu = defaultdict(list)
    for i, u in enumerate(users):
        byu[u].append(i)
    groups = []
    wt = np.asarray(wtarget)
    for idx in byu.values():
        if len(idx) < 2:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        vals = wt[idx]
        if float(vals.max() - vals.min()) < 1e-6:
            continue
        order = np.argsort(vals, kind='mergesort')
        m = max(1, len(idx) // 3)
        neg = idx[order[:m]]
        pos = idx[order[-m:]]
        # Avoid ambiguous equal-target pairs.
        if wt[pos].mean() > wt[neg].mean() + 1e-6:
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
    return groups


def sample_pairs(groups, rng):
    pos_parts, neg_parts = [], []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=len(p), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def train_pair_fm(enc, dim, groups, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                  patience=4, seed=0, device='cpu', verbose=True, bce_weight=0.15,
                  use_bce=True, tag=''):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    if verbose:
        print(f"{tag} pair users={len(groups):,d}, sampled pairs/epoch={sum(len(p) for p, _ in groups):,d}")
    torch.manual_seed(seed)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        model.train()
        losses = []
        for i in range(0, len(pos_idx), bs):
            psel = torch.from_numpy(pos_idx[i:i + bs])
            nsel = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            loss = -torch.nn.functional.logsigmoid(sp - sn).mean()
            if use_bce:
                bce = 0.5 * (torch.nn.functional.softplus(-sp).mean() +
                             torch.nn.functional.softplus(sn).mean())
                loss = loss + bce_weight * bce
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  {tag} early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def user_percentile_scores(scores, users):
    out = np.zeros(len(scores), dtype=np.float64)
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    for idx in groups.values():
        idx = np.asarray(idx, dtype=np.int64)
        n = len(idx)
        if n == 1:
            out[idx[0]] = 0.0
            continue
        order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
        out[idx] = ranks
    return out


def fm_rank_ensemble(models, X, users, device='cpu'):
    pred = np.zeros(len(X), dtype=np.float64)
    for model in models:
        pred += user_percentile_scores(model.predict(X, device=device), users)
    return pred / float(len(models))


def make_stat_maps(rows):
    gsum = sum(float(r[6]) for r in rows)
    gcnt = len(rows)
    gctr = gsum / max(gcnt, 1)
    names = ['video', 'author', 'tab', 'dur', 'author_tab']
    sums = {k: defaultdict(float) for k in names}
    cnts = {k: defaultdict(int) for k in names}
    for r in rows:
        y = float(r[6])
        dur_bucket = int(min(20, max(0, int(r[5]) // 5000)))
        keys = {'video': r[2], 'author': r[3], 'tab': r[4], 'dur': dur_bucket, 'author_tab': (r[3], r[4])}
        for k, key in keys.items():
            sums[k][key] += y
            cnts[k][key] += 1
    return gctr, sums, cnts


def stats_for_rows(rows, gctr, sums, cnts):
    feats = np.zeros((len(rows), 11), dtype=np.float32)
    names = ['video', 'author', 'tab', 'dur', 'author_tab']
    for i, r in enumerate(rows):
        dur_ms = float(r[5])
        dur_bucket = int(min(20, max(0, int(r[5]) // 5000)))
        keys = [r[2], r[3], r[4], dur_bucket, (r[3], r[4])]
        vals = []
        for name, key in zip(names, keys):
            c = cnts[name].get(key, 0)
            s = sums[name].get(key, 0.0)
            vals.append(np.log1p(c))
            vals.append((s + 20.0 * gctr) / (c + 20.0))
        feats[i, :10] = vals
        feats[i, 10] = np.log1p(max(dur_ms, 0.0))
    return feats


def lgb_matrix(enc_X, rows, gctr, sums, cnts, fm_rank):
    return np.hstack([enc_X[:, 1:].astype(np.int32), stats_for_rows(rows, gctr, sums, cnts),
                      fm_rank.reshape(-1, 1).astype(np.float32)]).astype(np.float32)


def sort_groups_by_user(X, y, users, init_score):
    users_arr = np.asarray(users)
    order = np.argsort(users_arr, kind='mergesort')
    su = users_arr[order]
    group = []
    last = None
    cnt = 0
    for u in su:
        if last is None or u == last:
            cnt += 1
        else:
            group.append(cnt)
            cnt = 1
        last = u
    if cnt:
        group.append(cnt)
    return X[order], y[order], group, init_score[order]


def train_lgb_residual_ranker(enc, splits, fm_train_rank, fm_valid_rank, seed=0, verbose=True):
    gctr, sums, cnts = make_stat_maps(splits['train'])
    Xtr_e, ytr, utr = enc['train']
    Xva_e, yva, uva = enc['valid']
    Xtr = lgb_matrix(Xtr_e, splits['train'], gctr, sums, cnts, fm_train_rank)
    Xva = lgb_matrix(Xva_e, splits['valid'], gctr, sums, cnts, fm_valid_rank)
    Xtr_s, ytr_s, gtr, init_tr = sort_groups_by_user(Xtr, ytr.astype(np.float32), utr, fm_train_rank.astype(np.float64))
    Xva_s, yva_s, gva, init_va = sort_groups_by_user(Xva, yva.astype(np.float32), uva, fm_valid_rank.astype(np.float64))
    dtr = lgb.Dataset(Xtr_s, label=ytr_s, group=gtr, init_score=init_tr,
                      categorical_feature=[0, 1, 2, 3], free_raw_data=False)
    dva = lgb.Dataset(Xva_s, label=yva_s, group=gva, init_score=init_va,
                      categorical_feature=[0, 1, 2, 3], reference=dtr, free_raw_data=False)
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5], 'label_gain': [0, 1],
        'learning_rate': 0.03, 'num_leaves': 31, 'min_data_in_leaf': 120,
        'feature_fraction': 0.90, 'bagging_fraction': 0.90, 'bagging_freq': 1,
        'lambda_l2': 5.0, 'verbosity': -1, 'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11, 'bagging_seed': int(seed) + 17, 'num_threads': 4,
    }
    callbacks = [lgb.early_stopping(25, verbose=verbose)]
    if not verbose:
        callbacks.append(lgb.log_evaluation(period=0))
    model = lgb.train(params, dtr, num_boost_round=250, valid_sets=[dva], valid_names=['valid'], callbacks=callbacks)
    return model, (gctr, sums, cnts)


def train_all(splits, data_dir, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True,
              bce_weight=0.15, n_models=5):
    enc, dim = encode(splits)
    label_groups = build_user_groups(enc['train'][1], enc['train'][2])
    fm_models = []
    for m in range(n_models):
        s = int(seed + 997 * m)
        fm_models.append(train_pair_fm(enc, dim, label_groups, k=k, lr=lr, epochs=epochs, seed=s,
                                       device=device, verbose=verbose, bce_weight=bce_weight,
                                       use_bce=True, tag=f"label{m}/seed{s}"))
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    fm_train_rank = fm_rank_ensemble(fm_models, Xtr, utr, device=device)
    fm_valid_rank = fm_rank_ensemble(fm_models, Xva, uva, device=device)
    lgb_model, stat_pack = train_lgb_residual_ranker(enc, splits, fm_train_rank, fm_valid_rank,
                                                     seed=seed + 4242, verbose=verbose)
    watch_target = read_train_watch_targets(data_dir, splits['train'])
    watch_groups = build_watch_groups(watch_target, utr)
    watch_model = train_pair_fm(enc, dim, watch_groups, k=k, lr=lr, epochs=epochs, seed=seed + 7777,
                                device=device, verbose=verbose, bce_weight=0.0, use_bce=False,
                                tag=f"watch/seed{seed+7777}")
    return fm_models, lgb_model, stat_pack, watch_model, enc


def predict_base(fm_models, lgb_model, stat_pack, enc_X, rows, users, device='cpu', fm_weight=0.50):
    fm_rank = fm_rank_ensemble(fm_models, enc_X, users, device=device)
    gctr, sums, cnts = stat_pack
    Xl = lgb_matrix(enc_X, rows, gctr, sums, cnts, fm_rank)
    delta = lgb_model.predict(Xl, num_iteration=lgb_model.best_iteration)
    corrected_rank = user_percentile_scores(fm_rank + delta, users)
    return fm_weight * fm_rank + (1.0 - fm_weight) * corrected_rank


def predict_all(fm_models, lgb_model, stat_pack, watch_model, enc_X, rows, users,
                device='cpu', watch_weight=0.30):
    base = predict_base(fm_models, lgb_model, stat_pack, enc_X, rows, users, device=device)
    watch_rank = user_percentile_scores(watch_model.predict(enc_X, device=device), users)
    return (1.0 - watch_weight) * base + watch_weight * watch_rank


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
    ap.add_argument('--bce_weight', type=float, default=0.15)
    ap.add_argument('--n_models', type=int, default=5)
    ap.add_argument('--watch_weight', type=float, default=0.30)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}, watch_weight={a.watch_weight}, fm_models={a.n_models}")
    fm_models, lgb_model, stat_pack, watch_model, enc = train_all(
        splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device,
        verbose=a.out is None, bce_weight=a.bce_weight, n_models=a.n_models)
    X, y, users = enc[target]
    scores = predict_all(fm_models, lgb_model, stat_pack, watch_model, X, splits[target], users,
                         device=a.device, watch_weight=a.watch_weight)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== watchtime_bpr_blend (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, predict_all(fm_models, lgb_model, stat_pack, watch_model,
                                                 Xs, splits[sp], us, device=a.device,
                                                 watch_weight=a.watch_weight))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
