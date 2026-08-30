"""Cached rank/score FM ensemble plus train-only historical residual priors.

Builds on 019: same six cached BPR/soft-hard FM members and 50/50
score-vs-rank fusion.  Keeps the user-context residual prior and adds a
small TRAIN-only global item/author popularity residual, then gives the
combined prior a readable 15% blend weight.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

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
            neg_arr = np.asarray(neg, dtype=np.int64)
            for p in pos:
                pos_all.append(p)
                neg_pools.append(neg_arr)
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
        raise RuntimeError('no within-user positive/negative pairs in train split')

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
        t0 = time.time()
        model.train()
        losses = []
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
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {name} seed {seed} epoch {ep:2d} | loss {np.mean(losses):.4f} | "
                  f"valid primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
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


def get_member_preds(member_name, enc, dim, target, split_name, seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache', f'010_{member_name}_seed{seed}_{split_name}.npy')
    want_len = len(enc[target][0])
    if os.path.isfile(cache_path):
        try:
            p = np.load(cache_path)
            if len(p) == want_len:
                if verbose:
                    print(f'loaded {member_name} seed {seed} from {cache_path}')
                return p.astype(np.float64)
        except Exception:
            pass
    if member_name == 'bpr1':
        p = train_bpr_member(enc, dim, target, seed=seed, n_neg=1,
                             soft_hard=False, device=device, verbose=verbose)
    elif member_name == 'soft5_tau1':
        p = train_bpr_member(enc, dim, target, seed=seed, n_neg=5,
                             soft_hard=True, tau=1.0, device=device,
                             verbose=verbose)
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
        vals = pred[idx]
        sd = vals.std()
        if sd > 1e-12:
            out[idx] = (vals - vals.mean()) / sd
        else:
            out[idx] = 0.0
    return out


def per_user_rank_percentile(pred, groups):
    out = np.empty_like(pred, dtype=np.float64)
    for idx in groups:
        vals = pred[idx]
        n = len(idx)
        if n <= 1:
            out[idx] = 0.0
            continue
        order = np.argsort(vals, kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[idx] = ranks / (n - 1.0)
    return out


def add_stat(dct, key, y):
    s, c = dct.get(key, (0.0, 0))
    dct[key] = (s + float(y), c + 1)


def smoothed_dev(dct, key, base, alpha):
    s, c = dct.get(key, (0.0, 0))
    if c <= 0:
        return 0.0
    rate = (s + alpha * base) / (c + alpha)
    return rate - base


def build_history_signal(splits, target):
    """TRAIN-only residual target encoding for repeated user and global context."""
    user_stat = {}
    uv_stat, ua_stat, ut_stat = {}, {}, {}
    v_stat, a_stat, t_stat = {}, {}, {}
    global_sum, global_cnt = 0.0, 0
    for row in splits['train']:
        u, v, au, tab, y = row[1], row[2], row[3], row[4], row[6]
        yy = float(y)
        add_stat(user_stat, u, yy)
        add_stat(uv_stat, (u, v), yy)
        add_stat(ua_stat, (u, au), yy)
        add_stat(ut_stat, (u, tab), yy)
        add_stat(v_stat, v, yy)
        add_stat(a_stat, au, yy)
        add_stat(t_stat, tab, yy)
        global_sum += yy
        global_cnt += 1
    global_mean = global_sum / max(1, global_cnt)

    out = np.zeros(len(splits[target]), dtype=np.float64)
    for i, row in enumerate(splits[target]):
        u, v, au, tab = row[1], row[2], row[3], row[4]
        us, uc = user_stat.get(u, (global_mean, 0))
        base = us / uc if uc > 0 else global_mean
        user_part = (1.00 * smoothed_dev(uv_stat, (u, v), base, alpha=1.0) +
                     0.45 * smoothed_dev(ua_stat, (u, au), base, alpha=5.0) +
                     0.20 * smoothed_dev(ut_stat, (u, tab), base, alpha=10.0))
        # Global priors use stronger smoothing; they can separate items/authors
        # even when a user has no exact historical match, but should not dominate.
        global_part = (0.35 * smoothed_dev(v_stat, v, global_mean, alpha=20.0) +
                       0.20 * smoothed_dev(a_stat, au, global_mean, alpha=50.0) +
                       0.05 * smoothed_dev(t_stat, tab, global_mean, alpha=200.0))
        out[i] = user_part + global_part
    return out


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
        target = 'valid'
        split_name = 'dev'
    else:
        splits = load(a.data_dir)
        target = a.split
        split_name = a.split
    if a.out is None:
        print({k: len(v) for k, v in splits.items()}, f'fields={FIELDS}')

    enc, dim = encode(splits)
    _, _, users = enc[target]
    groups = user_groups(users)

    bag_seeds = [0, 1, 2]
    z_bpr, z_soft, r_bpr, r_soft = [], [], [], []
    for s in bag_seeds:
        pb = get_member_preds('bpr1', enc, dim, target, split_name, s, a.device,
                              verbose=(a.out is None))
        ps = get_member_preds('soft5_tau1', enc, dim, target, split_name, s, a.device,
                              verbose=(a.out is None))
        z_bpr.append(per_user_z(pb, groups))
        z_soft.append(per_user_z(ps, groups))
        r_bpr.append(per_user_rank_percentile(pb, groups))
        r_soft.append(per_user_rank_percentile(ps, groups))

    score_z = 0.60 * np.mean(z_bpr, axis=0) + 0.40 * np.mean(z_soft, axis=0)
    score_rank = 0.60 * np.mean(r_bpr, axis=0) + 0.40 * np.mean(r_soft, axis=0)
    score_rank = per_user_z(score_rank, groups)
    ensemble = 0.50 * score_z + 0.50 * score_rank

    hist = per_user_z(build_history_signal(splits, target), groups)
    scores = 0.85 * ensemble + 0.15 * hist

    if a.out:
        np.save(a.out, scores.astype(np.float64))
    else:
        print('wrote predictions only when --out is supplied')
