"""OOF-ish LambdaMART stacking over the cached node-012 ensemble.

Node 014 trained the LightGBM combiner on in-sample TRAIN predictions from the
same FM members, which can teach the stacker artefacts of memorisation.  This
variant trains the combiner on a chronological holdout from TRAIN: base FM
members are fit on earlier train rows and predict the last train dates, then the
learned combiner is applied to the full-train members used by node 012.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


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
        outs = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            outs.append(self(xb).cpu().numpy())
        return np.concatenate(outs)


def make_user_pairs(y, users):
    by_user_pos, by_user_neg = {}, {}
    for i, (yy, u) in enumerate(zip(y, users)):
        if yy > 0.5:
            by_user_pos.setdefault(u, []).append(i)
        else:
            by_user_neg.setdefault(u, []).append(i)
    pos_all, neg_pools = [], []
    for u, pos in by_user_pos.items():
        neg = by_user_neg.get(u)
        if neg:
            arr = np.asarray(neg, dtype=np.int64)
            for p in pos:
                pos_all.append(p)
                neg_pools.append(arr)
    return np.asarray(pos_all, dtype=np.int64), neg_pools


def sample_negatives(neg_pools, rng, n_neg):
    if n_neg == 1:
        neg = np.empty(len(neg_pools), dtype=np.int64)
        for i, pool in enumerate(neg_pools):
            neg[i] = pool[rng.integers(len(pool))]
        return neg
    neg = np.empty((len(neg_pools), n_neg), dtype=np.int64)
    for i, pool in enumerate(neg_pools):
        neg[i] = pool[rng.integers(len(pool), size=n_neg)]
    return neg


def train_bpr_member(enc, dim, target, seed=0, k=16, lr=0.001, l2=1e-6,
                     epochs=40, bs=8192, patience=4, device='cpu',
                     n_neg=1, soft_hard=False, tau=1.0, verbose=False):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    pos_idx, neg_pools = make_user_pairs(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('no same-user pos/neg pairs')
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    name = 'soft_hard' if soft_hard else 'bpr1'
    for ep in range(1, epochs + 1):
        neg_idx = sample_negatives(neg_pools, rng, n_neg)
        order = rng.permutation(len(pos_idx))
        model.train(); losses = []; t0 = time.time()
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            xp = Xtr_t[torch.from_numpy(pos_idx[sel])].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            if soft_hard:
                xn = Xtr_t[torch.from_numpy(neg_idx[sel].reshape(-1))].to(device)
                sn = model(xn).view(len(sel), n_neg)
                per_pair = torch.nn.functional.softplus(-(sp.view(-1, 1) - sn))
                w = torch.softmax((sn / tau).detach(), dim=1)
                loss = (per_pair * w).sum(dim=1).mean()
            else:
                xn = Xtr_t[torch.from_numpy(neg_idx[sel])].to(device)
                sn = model(xn)
                loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f'{name} seed {seed} ep {ep} loss {np.mean(losses):.4f} valid {va["primary"]:.4f} {time.time()-t0:.1f}s')
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    Xtar, _, _ = enc[target]
    return model.predict(Xtar, device=device).astype(np.float64)


def get_member_preds(member_name, enc, dim, target, cache_split_name, seed, device, verbose, prefix='010'):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache', f'{prefix}_{member_name}_seed{seed}_{cache_split_name}.npy')
    want_len = len(enc[target][0])
    if os.path.isfile(cache_path):
        try:
            p = np.load(cache_path)
            if len(p) == want_len:
                if verbose:
                    print('loaded', cache_path)
                return p.astype(np.float64)
        except Exception:
            pass
    if member_name == 'bpr1':
        p = train_bpr_member(enc, dim, target, seed=seed, n_neg=1,
                             soft_hard=False, device=device, verbose=verbose)
    elif member_name == 'soft5_tau1':
        p = train_bpr_member(enc, dim, target, seed=seed, n_neg=5,
                             soft_hard=True, tau=1.0, device=device, verbose=verbose)
    else:
        raise ValueError(member_name)
    np.save(cache_path, p)
    return p


def user_groups(users):
    groups = {}
    for i, u in enumerate(users):
        groups.setdefault(u, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def per_user_z(pred, groups):
    out = np.empty_like(pred, dtype=np.float64)
    for idx in groups:
        vals = pred[idx]; sd = vals.std()
        out[idx] = (vals - vals.mean()) / sd if sd > 1e-12 else 0.0
    return out


def per_user_rank_percentile(pred, groups):
    out = np.empty_like(pred, dtype=np.float64)
    for idx in groups:
        vals = pred[idx]; n = len(idx)
        if n <= 1:
            out[idx] = 0.0; continue
        order = np.argsort(vals, kind='mergesort')
        ranks = np.empty(n, dtype=np.float64); ranks[order] = np.arange(n, dtype=np.float64)
        out[idx] = ranks / (n - 1.0)
    return out


def build_features(preds_bpr, preds_soft, groups):
    z_bpr = [per_user_z(p, groups) for p in preds_bpr]
    z_soft = [per_user_z(p, groups) for p in preds_soft]
    r_bpr = [per_user_rank_percentile(p, groups) for p in preds_bpr]
    r_soft = [per_user_rank_percentile(p, groups) for p in preds_soft]
    mean_z_bpr = np.mean(z_bpr, axis=0)
    mean_z_soft = np.mean(z_soft, axis=0)
    mean_r_bpr = np.mean(r_bpr, axis=0)
    mean_r_soft = np.mean(r_soft, axis=0)
    score_z = 0.60 * mean_z_bpr + 0.40 * mean_z_soft
    score_rank_raw = 0.60 * mean_r_bpr + 0.40 * mean_r_soft
    score_rank = per_user_z(score_rank_raw, groups)
    parent_score = 0.70 * score_z + 0.30 * score_rank
    disagreement = np.abs(mean_z_bpr - mean_z_soft)
    rank_disagreement = np.abs(mean_r_bpr - mean_r_soft)
    cols = []
    cols.extend(z_bpr); cols.extend(z_soft); cols.extend(r_bpr); cols.extend(r_soft)
    cols.extend([mean_z_bpr, mean_z_soft, mean_r_bpr, mean_r_soft,
                 score_z, score_rank, parent_score, disagreement, rank_disagreement])
    return np.vstack(cols).T.astype(np.float32), parent_score


def sorted_group_info(users):
    users_arr = np.asarray(users).astype(str)
    order = np.argsort(users_arr, kind='mergesort')
    su = users_arr[order]
    if len(su) == 0:
        return order, []
    cuts = np.flatnonzero(su[1:] != su[:-1]) + 1
    starts = np.r_[0, cuts]; ends = np.r_[cuts, len(su)]
    return order, (ends - starts).astype(int).tolist()


def chronological_train_holdout(train_rows, holdout_days=3):
    dates = sorted(set(int(r[0]) for r in train_rows))
    if len(dates) <= holdout_days:
        cut_dates = set(dates[-1:])
    else:
        cut_dates = set(dates[-holdout_days:])
    early = [r for r in train_rows if int(r[0]) not in cut_dates]
    hold = [r for r in train_rows if int(r[0]) in cut_dates]
    return early, hold


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'; target_cache = 'dev'
    else:
        splits = load(a.data_dir)
        target = a.split; target_cache = a.split
    if a.out is None:
        print({k: len(v) for k, v in splits.items()}, f'fields={FIELDS}')

    # Full-train member predictions for the requested target: identical members to node 012.
    enc_full, dim_full = encode(splits)
    _, _, utar = enc_full[target]
    groups_tar = user_groups(utar)
    bag_seeds = [0, 1, 2]
    tar_bpr, tar_soft = [], []
    for s in bag_seeds:
        tar_bpr.append(get_member_preds('bpr1', enc_full, dim_full, target, target_cache, s,
                                        a.device, verbose=(a.out is None), prefix='010'))
        tar_soft.append(get_member_preds('soft5_tau1', enc_full, dim_full, target, target_cache, s,
                                         a.device, verbose=(a.out is None), prefix='010'))
    X_meta_tar, parent_tar = build_features(tar_bpr, tar_soft, groups_tar)

    # Chronological OOF-ish meta training from TRAIN only.
    early_rows, hold_rows = chronological_train_holdout(splits['train'], holdout_days=3)
    meta_splits = {'train': early_rows, 'valid': hold_rows}
    enc_meta, dim_meta = encode(meta_splits)
    _, y_hold, u_hold = enc_meta['valid']
    groups_hold = user_groups(u_hold)
    hold_bpr, hold_soft = [], []
    # Include split sizes in cache key to avoid stale reuse across dev/valid loaders.
    meta_tag = f'oof{len(early_rows)}_{len(hold_rows)}'
    for s in bag_seeds:
        hold_bpr.append(get_member_preds('bpr1', enc_meta, dim_meta, 'valid', meta_tag, s,
                                         a.device, verbose=(a.out is None), prefix='016'))
        hold_soft.append(get_member_preds('soft5_tau1', enc_meta, dim_meta, 'valid', meta_tag, s,
                                          a.device, verbose=(a.out is None), prefix='016'))
    X_meta_hold, _ = build_features(hold_bpr, hold_soft, groups_hold)

    order, group = sorted_group_info(u_hold)
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=60,
        learning_rate=0.04,
        num_leaves=7,
        max_depth=3,
        min_child_samples=40,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=2028,
        n_jobs=2,
        verbosity=-1,
    )
    ranker.fit(X_meta_hold[order], y_hold[order].astype(int), group=group)
    meta_tar = ranker.predict(X_meta_tar).astype(np.float64)
    meta_tar = per_user_z(meta_tar, groups_tar)

    # Still a readable stacker test (>=20%), but less than node 014 because the
    # OOF holdout is smaller and noisier than the full TRAIN set.
    scores = 0.75 * parent_tar + 0.25 * meta_tar

    if a.out:
        np.save(a.out, scores.astype(np.float64))
    else:
        print('wrote predictions only when --out is supplied')
