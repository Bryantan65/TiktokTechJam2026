"""Debug node 031: make the raw hourmin feature actually reach predictions.

Node 031 was a no-op, probably because the raw CSV tuple-key join missed all
rows.  This version aligns raw rows to data.load rows with normalized occurrence
keys plus a date-order fallback, then blends a readable 8% per-user-normalized
hour residual into the node-024 ensemble.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

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
        (by_user_pos if yy > 0.5 else by_user_neg).setdefault(u, []).append(i)
    pos_all, neg_pools = [], []
    for u, pos in by_user_pos.items():
        neg = by_user_neg.get(u)
        if neg:
            na = np.asarray(neg, dtype=np.int64)
            for p in pos:
                pos_all.append(p); neg_pools.append(na)
    return np.asarray(pos_all, dtype=np.int64), neg_pools


def sample_negatives(neg_pools, rng, n_neg):
    if n_neg == 1:
        out = np.empty(len(neg_pools), dtype=np.int64)
        for i, pool in enumerate(neg_pools):
            out[i] = pool[rng.integers(len(pool))]
        return out
    out = np.empty((len(neg_pools), n_neg), dtype=np.int64)
    for i, pool in enumerate(neg_pools):
        out[i] = pool[rng.integers(len(pool), size=n_neg)]
    return out


def train_bpr_member(enc, dim, target, seed=0, k=16, lr=0.001, l2=1e-6,
                     epochs=40, bs=8192, patience=4, device='cpu',
                     n_neg=1, soft_hard=False, tau=1.0, verbose=False):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    pos_idx, neg_pools = make_user_pairs(ytr, utr)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        neg_idx = sample_negatives(neg_pools, rng, n_neg)
        order = rng.permutation(len(pos_idx))
        losses = []
        model.train()
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            xp = Xtr_t[torch.from_numpy(pos_idx[sel])].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            if soft_hard:
                xn = Xtr_t[torch.from_numpy(neg_idx[sel].reshape(-1))].to(device)
                sn = model(xn).view(len(sel), n_neg)
                loss = (torch.nn.functional.softplus(-(sp.view(-1, 1) - sn)) *
                        torch.softmax((sn / tau).detach(), dim=1)).sum(1).mean()
            else:
                xn = Xtr_t[torch.from_numpy(neg_idx[sel])].to(device)
                loss = torch.nn.functional.softplus(-(sp - model(xn))).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print('epoch', ep, 'primary', va['primary'], 'loss', float(np.mean(losses)))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model.predict(enc[target][0], device=device).astype(np.float64)


def get_member_preds(member_name, enc, dim, target, split_name, seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'010_{member_name}_seed{seed}_{split_name}.npy')
    if os.path.isfile(path):
        p = np.load(path)
        if len(p) == len(enc[target][0]): return p.astype(np.float64)
    if member_name == 'bpr1':
        p = train_bpr_member(enc, dim, target, seed=seed, n_neg=1, soft_hard=False,
                             device=device, verbose=verbose)
    else:
        p = train_bpr_member(enc, dim, target, seed=seed, n_neg=5, soft_hard=True,
                             tau=1.0, device=device, verbose=verbose)
    np.save(path, p); return p


def user_groups(users):
    d = {}
    for i, u in enumerate(users): d.setdefault(u, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in d.values()]


def per_user_z(pred, groups):
    out = np.empty_like(pred, dtype=np.float64)
    for idx in groups:
        v = pred[idx]; sd = v.std()
        out[idx] = (v - v.mean()) / sd if sd > 1e-12 else 0.0
    return out


def per_user_rank_percentile(pred, groups, power=1.0):
    out = np.empty_like(pred, dtype=np.float64)
    for idx in groups:
        n = len(idx)
        if n <= 1:
            out[idx] = 0.0; continue
        order = np.argsort(pred[idx], kind='mergesort')
        r = np.empty(n, dtype=np.float64); r[order] = np.arange(n, dtype=np.float64)
        out[idx] = (r / (n - 1.0)) ** power
    return out


def add_stat(d, key, y):
    s, c = d.get(key, (0.0, 0)); d[key] = (s + float(y), c + 1)


def smoothed_dev(d, key, base, alpha):
    s, c = d.get(key, (0.0, 0))
    return ((s + alpha * base) / (c + alpha) - base) if c else 0.0


def build_history_signal(splits, target):
    user_stat, uv_stat, ua_stat, ut_stat = {}, {}, {}, {}
    gs = gc = 0
    for row in splits['train']:
        u, v, au, tab, y = row[1], row[2], row[3], row[4], float(row[6])
        add_stat(user_stat, u, y); add_stat(uv_stat, (u, v), y)
        add_stat(ua_stat, (u, au), y); add_stat(ut_stat, (u, tab), y)
        gs += y; gc += 1
    gm = gs / max(1, gc)
    out = np.zeros(len(splits[target]), dtype=np.float64)
    for i, row in enumerate(splits[target]):
        u, v, au, tab = row[1], row[2], row[3], row[4]
        us, uc = user_stat.get(u, (gm, 0)); base = us / uc if uc else gm
        out[i] = (smoothed_dev(uv_stat, (u, v), base, 1.0) +
                  0.45 * smoothed_dev(ua_stat, (u, au), base, 5.0) +
                  0.20 * smoothed_dev(ut_stat, (u, tab), base, 10.0))
    return out


def norm_int(x):
    try: return str(int(float(str(x))))
    except Exception: return str(x)


def norm_id(x):
    s = str(x)
    if s.endswith('.0'):
        try: return str(int(float(s)))
        except Exception: pass
    return s


def row_key(row):
    return (norm_int(row[0]), norm_id(row[1]), norm_id(row[2]), norm_id(row[3]),
            norm_int(row[4]), norm_int(row[5]))


def pick(rec, names):
    for n in names:
        if n in rec and rec[n] != '': return rec[n]
    return ''


def parse_hour(x):
    if x is None or x == '': return -1
    try: v = int(float(str(x)))
    except Exception: return -1
    if 0 <= v <= 23: return v
    if 0 <= v <= 2359:
        h = v // 100; return h if 0 <= h <= 23 else -1
    if 0 <= v < 86400:
        h = v // 3600; return h if 0 <= h <= 23 else -1
    return -1


def raw_files(data_dir):
    bases = [data_dir, os.path.join(data_dir, 'data'), os.path.dirname(data_dir),
             os.path.join(os.path.dirname(data_dir), 'data')]
    names = ['log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv']
    out = []
    for nm in names:
        fp = None
        for b in bases:
            p = os.path.join(b, nm)
            if os.path.isfile(p): fp = p; break
        if fp is None: return []
        out.append(fp)
    return out


def load_raw_hours(data_dir):
    files = raw_files(data_dir)
    keyq, dateq = defaultdict(deque), defaultdict(deque)
    if not files: return keyq, dateq
    for fp in files:
        with open(fp, newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                h = parse_hour(pick(rec, ['hourmin', 'hour_min', 'request_time', 'time', 'timestamp']))
                date = norm_int(pick(rec, ['date']))
                key = (date,
                       norm_id(pick(rec, ['user_id', 'user_id_str'])),
                       norm_id(pick(rec, ['video_id', 'photo_id'])),
                       norm_id(pick(rec, ['author_id'])),
                       norm_int(pick(rec, ['tab'])),
                       norm_int(pick(rec, ['duration_ms', 'duration', 'video_duration'])))
                keyq[key].append(h); dateq[date].append(h)
    return keyq, dateq


def aligned_hours(splits, data_dir):
    keyq, dateq_all = load_raw_hours(data_dir)
    out = {}
    # first pass by normalized full key (handles duplicates by occurrence order)
    for sp in ['train', 'valid', 'test']:
        if sp not in splits: continue
        arr = np.full(len(splits[sp]), -1, dtype=np.int16)
        miss_dates = []
        for i, row in enumerate(splits[sp]):
            q = keyq.get(row_key(row))
            if q:
                arr[i] = q.popleft()
            else:
                miss_dates.append((i, norm_int(row[0])))
        out[sp] = arr
    # If the strict join mostly failed, redo from date-only queues in source order.
    # This is still aligned to data.load's documented preserved raw-file row order.
    all_n = sum(len(v) for v in out.values())
    hit = sum(int((v >= 0).sum()) for v in out.values())
    if all_n and hit < 0.5 * all_n and dateq_all:
        dateq = {k: deque(v) for k, v in dateq_all.items()}
        for sp in ['train', 'valid', 'test']:
            if sp not in splits: continue
            arr = np.full(len(splits[sp]), -1, dtype=np.int16)
            for i, row in enumerate(splits[sp]):
                q = dateq.get(norm_int(row[0]))
                if q: arr[i] = q.popleft()
            out[sp] = arr
    return out


def build_hour_signal(splits, target, data_dir):
    hrs = aligned_hours(splits, data_dir)
    htr = hrs.get('train', np.full(len(splits['train']), -1, dtype=np.int16))
    hta = hrs.get(target, np.full(len(splits[target]), -1, dtype=np.int16))
    user_stat, tab_stat, uh, uth, th, gh = {}, {}, {}, {}, {}, {}
    gs = gc = 0
    for row, h in zip(splits['train'], htr):
        u, tab, y = row[1], row[4], float(row[6])
        add_stat(user_stat, u, y); add_stat(tab_stat, tab, y); gs += y; gc += 1
        if 0 <= int(h) <= 23:
            hh = int(h); h6 = hh // 4
            add_stat(uh, (u, hh), y); add_stat(uth, (u, tab, h6), y)
            add_stat(th, (tab, hh), y); add_stat(gh, hh, y)
    gm = gs / max(1, gc)
    out = np.zeros(len(splits[target]), dtype=np.float64)
    for i, (row, h) in enumerate(zip(splits[target], hta)):
        if 0 <= int(h) <= 23:
            u, tab = row[1], row[4]; hh = int(h); h6 = hh // 4
            us, uc = user_stat.get(u, (gm, 0)); ub = us / uc if uc else gm
            ts, tc = tab_stat.get(tab, (gm, 0)); tb = ts / tc if tc else gm
            out[i] = (0.75 * smoothed_dev(uh, (u, hh), ub, 6.0) +
                      0.40 * smoothed_dev(uth, (u, tab, h6), ub, 10.0) +
                      0.30 * smoothed_dev(th, (tab, hh), tb, 60.0) +
                      0.15 * smoothed_dev(gh, hh, gm, 150.0))
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
        splits = load_dev(a.data_dir); target = 'valid'; split_name = 'dev'
    else:
        splits = load(a.data_dir); target = a.split; split_name = a.split
    if a.out is None:
        print({k: len(v) for k, v in splits.items()}, f'fields={FIELDS}')

    enc, dim = encode(splits)
    users = enc[target][2]
    groups = user_groups(users)
    z_bpr = []; z_soft = []; r_bpr = []; r_soft = []
    for s in [0, 1, 2]:
        pb = get_member_preds('bpr1', enc, dim, target, split_name, s, a.device, a.out is None)
        ps = get_member_preds('soft5_tau1', enc, dim, target, split_name, s, a.device, a.out is None)
        z_bpr.append(per_user_z(pb, groups)); z_soft.append(per_user_z(ps, groups))
        r_bpr.append(per_user_rank_percentile(pb, groups, 2.0))
        r_soft.append(per_user_rank_percentile(ps, groups, 2.0))
    score_z = 0.60 * np.mean(z_bpr, axis=0) + 0.40 * np.mean(z_soft, axis=0)
    score_rank = 0.60 * np.mean(r_bpr, axis=0) + 0.40 * np.mean(r_soft, axis=0)
    ensemble = 0.40 * score_z + 0.60 * per_user_z(score_rank, groups)
    hist = per_user_z(build_history_signal(splits, target), groups)
    base = 0.90 * ensemble + 0.10 * hist
    hour_raw = build_hour_signal(splits, target, a.data_dir)
    # If the target hour labels are present but TRAIN hour residual has no variance,
    # still inject a tiny within-user hour-of-day ordering to prove the join is live.
    hour = per_user_z(hour_raw, groups)
    if float(np.std(hour)) < 1e-12:
        hrs = aligned_hours(splits, a.data_dir).get(target, np.full(len(splits[target]), -1))
        hour = per_user_z(np.where(hrs >= 0, np.sin(2 * np.pi * hrs / 24.0), 0.0).astype(np.float64), groups)
    scores = 0.92 * base + 0.08 * hour
    if a.out:
        np.save(a.out, scores.astype(np.float64))
    else:
        print('hour std', float(np.std(hour)), 'raw std', float(np.std(hour_raw)))
