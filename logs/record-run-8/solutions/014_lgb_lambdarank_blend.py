"""Blend the rank-ensemble BPR+BCE FM with a LightGBM LambdaRank model.

The FM part is node 13.  The extra member is deliberately given 30% of the
within-user rank vote so this is a readable test of a listwise tree ranker rather
than an almost-invisible low-weight ensemble addition.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__), '..'), 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


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


def sample_pairs(groups, rng):
    pos_parts, neg_parts = [], []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=len(p), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def train_one_fm(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                 patience=4, seed=0, device='cpu', verbose=True,
                 bce_weight=0.15, tag=''):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    groups = build_user_groups(ytr, utr)
    if verbose:
        print(f"{tag} BPR eligible users={len(groups):,d}, sampled pairs/epoch={sum(len(p) for p, _ in groups):,d}")

    torch.manual_seed(seed)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
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
            bpr = -torch.nn.functional.logsigmoid(sp - sn).mean()
            bce = 0.5 * (torch.nn.functional.softplus(-sp).mean() +
                         torch.nn.functional.softplus(sn).mean())
            loss = bpr + bce_weight * bce
            loss.backward()
            opt.step()
            losses.append(loss.item())

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
    specs = {
        'user': 1,
        'video': 2,
        'author': 3,
        'tab': 4,
        'dur': None,
        'user_tab': None,
        'author_tab': None,
    }
    sums = {k: defaultdict(float) for k in specs}
    cnts = {k: defaultdict(int) for k in specs}
    for r in rows:
        y = float(r[6])
        dur_bucket = int(min(20, max(0, int(r[5]) // 5000)))
        keys = {
            'user': r[1],
            'video': r[2],
            'author': r[3],
            'tab': r[4],
            'dur': dur_bucket,
            'user_tab': (r[1], r[4]),
            'author_tab': (r[3], r[4]),
        }
        for k, key in keys.items():
            sums[k][key] += y
            cnts[k][key] += 1
    return gctr, sums, cnts


def stats_for_rows(rows, gctr, sums, cnts):
    feats = np.zeros((len(rows), 15), dtype=np.float32)
    for i, r in enumerate(rows):
        dur_ms = float(r[5])
        dur_bucket = int(min(20, max(0, int(r[5]) // 5000)))
        keys = [r[1], r[2], r[3], r[4], dur_bucket, (r[1], r[4]), (r[3], r[4])]
        names = ['user', 'video', 'author', 'tab', 'dur', 'user_tab', 'author_tab']
        vals = []
        for name, key in zip(names, keys):
            c = cnts[name].get(key, 0)
            s = sums[name].get(key, 0.0)
            vals.append(np.log1p(c))
            vals.append((s + 20.0 * gctr) / (c + 20.0))
        feats[i, :14] = vals
        feats[i, 14] = np.log1p(max(dur_ms, 0.0))
    return feats


def lgb_matrix(enc_X, rows, gctr, sums, cnts):
    return np.hstack([enc_X.astype(np.int32), stats_for_rows(rows, gctr, sums, cnts)]).astype(np.float32)


def sort_groups_by_user(X, y, users):
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
    return X[order], y[order], group


def train_lgb_ranker(enc, splits, seed=0, verbose=True):
    gctr, sums, cnts = make_stat_maps(splits['train'])
    Xtr_e, ytr, utr = enc['train']
    Xva_e, yva, uva = enc['valid']
    Xtr = lgb_matrix(Xtr_e, splits['train'], gctr, sums, cnts)
    Xva = lgb_matrix(Xva_e, splits['valid'], gctr, sums, cnts)
    Xtr_s, ytr_s, gtr = sort_groups_by_user(Xtr, ytr.astype(np.float32), utr)
    Xva_s, yva_s, gva = sort_groups_by_user(Xva, yva.astype(np.float32), uva)
    dtr = lgb.Dataset(Xtr_s, label=ytr_s, group=gtr, categorical_feature=list(range(len(FIELDS))), free_raw_data=False)
    dva = lgb.Dataset(Xva_s, label=yva_s, group=gva, categorical_feature=list(range(len(FIELDS))), reference=dtr, free_raw_data=False)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'label_gain': [0, 1],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 80,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'lambda_l2': 2.0,
        'verbosity': -1,
        'seed': int(seed),
        'feature_fraction_seed': int(seed) + 11,
        'bagging_seed': int(seed) + 17,
        'num_threads': 4,
    }
    callbacks = [lgb.early_stopping(30, verbose=verbose)]
    if not verbose:
        callbacks.append(lgb.log_evaluation(period=0))
    model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva], valid_names=['valid'], callbacks=callbacks)
    return model, (gctr, sums, cnts)


def train_all(splits, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True,
              bce_weight=0.15, n_models=5):
    enc, dim = encode(splits)
    fm_models = []
    for m in range(n_models):
        s = int(seed + 997 * m)
        fm_models.append(train_one_fm(enc, dim, k=k, lr=lr, epochs=epochs, seed=s,
                                      device=device, verbose=verbose,
                                      bce_weight=bce_weight, tag=f"m{m}/seed{s}"))
    lgb_model, stat_pack = train_lgb_ranker(enc, splits, seed=seed + 4242, verbose=verbose)
    return fm_models, lgb_model, stat_pack, enc


def predict_blend(fm_models, lgb_model, stat_pack, enc_X, rows, users, device='cpu', fm_weight=0.70):
    fm_rank = fm_rank_ensemble(fm_models, enc_X, users, device=device)
    gctr, sums, cnts = stat_pack
    Xl = lgb_matrix(enc_X, rows, gctr, sums, cnts)
    lgb_rank = user_percentile_scores(lgb_model.predict(Xl, num_iteration=lgb_model.best_iteration), users)
    return fm_weight * fm_rank + (1.0 - fm_weight) * lgb_rank


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
    ap.add_argument('--fm_weight', type=float, default=0.70)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}, fm_models={a.n_models}, fm_weight={a.fm_weight}")

    fm_models, lgb_model, stat_pack, enc = train_all(
        splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device,
        verbose=a.out is None, bce_weight=a.bce_weight, n_models=a.n_models)
    X, y, users = enc[target]
    scores = predict_blend(fm_models, lgb_model, stat_pack, X, splits[target], users,
                           device=a.device, fm_weight=a.fm_weight)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== lgb_lambdarank_blend (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, predict_blend(fm_models, lgb_model, stat_pack, Xs, splits[sp], us,
                                                   device=a.device, fm_weight=a.fm_weight))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
