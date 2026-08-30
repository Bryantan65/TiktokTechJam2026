"""Debug node 029: cheap tuple-only calendar residual on top of node 024.

Node 029 timed out while joining raw CSV hourmin.  This keeps the same time-feature
hypothesis but removes the expensive/raw join: use date already present in the
loader tuples (weekday and train-window date trend) to build a small TRAIN-only
calendar residual, then blend it lightly with the cached node-024 ensemble.
"""
import argparse
import datetime as _dt
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
        out[idx] = (vals - vals.mean()) / sd if sd > 1e-12 else 0.0
    return out


def per_user_rank_percentile(pred, groups, power=1.0):
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
        pct = ranks / (n - 1.0)
        out[idx] = pct ** power if power != 1.0 else pct
    return out


def add_stat(dct, key, y):
    s, c = dct.get(key, (0.0, 0))
    dct[key] = (s + float(y), c + 1)


def smoothed_dev(dct, key, base, alpha):
    s, c = dct.get(key, (0.0, 0))
    if c <= 0:
        return 0.0
    return (s + alpha * base) / (c + alpha) - base


def build_history_signal(splits, target):
    user_stat = {}
    uv_stat, ua_stat, ut_stat = {}, {}, {}
    global_sum, global_cnt = 0.0, 0
    for row in splits['train']:
        u, v, au, tab, y = row[1], row[2], row[3], row[4], row[6]
        yy = float(y)
        add_stat(user_stat, u, yy)
        add_stat(uv_stat, (u, v), yy)
        add_stat(ua_stat, (u, au), yy)
        add_stat(ut_stat, (u, tab), yy)
        global_sum += yy
        global_cnt += 1
    global_mean = global_sum / max(1, global_cnt)

    out = np.zeros(len(splits[target]), dtype=np.float64)
    for i, row in enumerate(splits[target]):
        u, v, au, tab = row[1], row[2], row[3], row[4]
        us, uc = user_stat.get(u, (global_mean, 0))
        base = us / uc if uc > 0 else global_mean
        out[i] = (1.00 * smoothed_dev(uv_stat, (u, v), base, alpha=1.0) +
                  0.45 * smoothed_dev(ua_stat, (u, au), base, alpha=5.0) +
                  0.20 * smoothed_dev(ut_stat, (u, tab), base, alpha=10.0))
    return out


def weekday(date_val):
    try:
        s = str(int(date_val))
        return _dt.datetime.strptime(s, '%Y%m%d').weekday()
    except Exception:
        return -1


def date_ord(date_val):
    try:
        s = str(int(date_val))
        return _dt.datetime.strptime(s, '%Y%m%d').toordinal()
    except Exception:
        return 0


def build_calendar_signal(splits, target):
    user_stat, tab_stat = {}, {}
    uw_stat, utw_stat, tw_stat, gw_stat = {}, {}, {}, {}
    # simple smoothed linear label trend by tab and globally from TRAIN dates
    rows_for_slope = []
    by_tab_slope = {}
    tab_rows = {}
    gsum, gcnt = 0.0, 0
    for row in splits['train']:
        d, u, tab, y = row[0], row[1], row[4], float(row[6])
        w = weekday(d)
        od = date_ord(d)
        add_stat(user_stat, u, y)
        add_stat(tab_stat, tab, y)
        if w >= 0:
            add_stat(uw_stat, (u, w), y)
            add_stat(utw_stat, (u, tab, w), y)
            add_stat(tw_stat, (tab, w), y)
            add_stat(gw_stat, w, y)
        if od > 0:
            rows_for_slope.append((od, y))
            tab_rows.setdefault(tab, []).append((od, y))
        gsum += y
        gcnt += 1
    gm = gsum / max(1, gcnt)

    def slope(pairs):
        if len(pairs) < 20:
            return 0.0
        x = np.asarray([p[0] for p in pairs], dtype=np.float64)
        y = np.asarray([p[1] for p in pairs], dtype=np.float64)
        x = x - x.mean()
        den = float((x * x).sum())
        return float((x * (y - y.mean())).sum() / den) if den > 1e-12 else 0.0

    gslope = slope(rows_for_slope)
    for tab, pairs in tab_rows.items():
        by_tab_slope[tab] = slope(pairs)
    train_ords = [p[0] for p in rows_for_slope]
    ref_ord = max(train_ords) if train_ords else 0

    out = np.zeros(len(splits[target]), dtype=np.float64)
    for i, row in enumerate(splits[target]):
        d, u, tab = row[0], row[1], row[4]
        w = weekday(d)
        od = date_ord(d)
        us, uc = user_stat.get(u, (gm, 0))
        ub = us / uc if uc > 0 else gm
        ts, tc = tab_stat.get(tab, (gm, 0))
        tb = ts / tc if tc > 0 else gm
        val = 0.0
        if w >= 0:
            val += 0.55 * smoothed_dev(uw_stat, (u, w), ub, alpha=8.0)
            val += 0.25 * smoothed_dev(utw_stat, (u, tab, w), ub, alpha=12.0)
            val += 0.18 * smoothed_dev(tw_stat, (tab, w), tb, alpha=80.0)
            val += 0.10 * smoothed_dev(gw_stat, w, gm, alpha=200.0)
        if od > 0 and ref_ord > 0:
            days = max(-14.0, min(21.0, float(od - ref_ord)))
            val += 0.20 * days * gslope + 0.18 * days * by_tab_slope.get(tab, 0.0)
        out[i] = val
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
        r_bpr.append(per_user_rank_percentile(pb, groups, power=2.0))
        r_soft.append(per_user_rank_percentile(ps, groups, power=2.0))

    score_z = 0.60 * np.mean(z_bpr, axis=0) + 0.40 * np.mean(z_soft, axis=0)
    score_rank = 0.60 * np.mean(r_bpr, axis=0) + 0.40 * np.mean(r_soft, axis=0)
    score_rank = per_user_z(score_rank, groups)
    ensemble = 0.40 * score_z + 0.60 * score_rank

    hist = per_user_z(build_history_signal(splits, target), groups)
    base = 0.90 * ensemble + 0.10 * hist

    cal = per_user_z(build_calendar_signal(splits, target), groups)
    scores = 0.97 * base + 0.03 * cal

    if a.out:
        np.save(a.out, scores.astype(np.float64))
    else:
        print('wrote predictions only when --out is supplied')
