"""History-affinity blend on top of the best cached rank-fusion ensemble.

Keeps node 012's six cached BPR/soft-hard FM members and adds a DIN-inspired
user-history affinity signal: target items/authors/tabs are scored from the
same user's labelled training history, with smoothed global item/author priors.
The history score is per-user z-normalised and blended at 30% so the signal is
large enough to test whether behavioural history complements the FM ensemble.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

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
    """Within-user rank score in [0,1], higher is better."""
    out = np.empty_like(pred, dtype=np.float64)
    for idx in groups:
        vals = pred[idx]
        n = len(idx)
        if n <= 1:
            out[idx] = 0.0
            continue
        order = np.argsort(vals, kind='mergesort')  # ascending
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[idx] = ranks / (n - 1.0)
    return out


def add_stat(d, key, y):
    a = d.get(key)
    if a is None:
        d[key] = [float(y), 1.0]
    else:
        a[0] += float(y)
        a[1] += 1.0


def smoothed_logit(stat, global_rate, alpha):
    if stat is None:
        return 0.0, 0.0
    pos, cnt = stat
    p = (pos + alpha * global_rate) / (cnt + alpha)
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    g = min(max(global_rate, 1e-4), 1.0 - 1e-4)
    return np.log(p / (1.0 - p)) - np.log(g / (1.0 - g)), cnt


def bucket_duration(ms):
    try:
        x = int(ms)
    except Exception:
        return 'unk'
    if x < 5_000:
        return 0
    if x < 10_000:
        return 1
    if x < 20_000:
        return 2
    if x < 40_000:
        return 3
    if x < 80_000:
        return 4
    return 5


def history_affinity_score(splits, target, groups):
    """Smoothed train-history score for each target row, using no target labels."""
    train_rows = splits['train']
    target_rows = splits[target]
    ysum = sum(float(r[6]) for r in train_rows)
    global_rate = ysum / max(1, len(train_rows))

    uv = {}
    ua = {}
    utab = {}
    udur = {}
    gv = {}
    ga = {}
    for r in train_rows:
        u, v, a, tab, dur, y = str(r[1]), str(r[2]), str(r[3]), str(r[4]), r[5], float(r[6])
        db = bucket_duration(dur)
        add_stat(uv, (u, v), y)
        add_stat(ua, (u, a), y)
        add_stat(utab, (u, tab), y)
        add_stat(udur, (u, db), y)
        add_stat(gv, v, y)
        add_stat(ga, a, y)

    out = np.zeros(len(target_rows), dtype=np.float64)
    for i, r in enumerate(target_rows):
        u, v, a, tab, dur = str(r[1]), str(r[2]), str(r[3]), str(r[4]), r[5]
        db = bucket_duration(dur)
        s = 0.0
        val, cnt = smoothed_logit(uv.get((u, v)), global_rate, alpha=1.0)
        s += 1.20 * min(cnt, 3.0) / 3.0 * val
        val, cnt = smoothed_logit(ua.get((u, a)), global_rate, alpha=3.0)
        s += 0.75 * min(cnt, 10.0) / 10.0 * val
        val, cnt = smoothed_logit(utab.get((u, tab)), global_rate, alpha=8.0)
        s += 0.25 * min(cnt, 20.0) / 20.0 * val
        val, cnt = smoothed_logit(udur.get((u, db)), global_rate, alpha=8.0)
        s += 0.15 * min(cnt, 20.0) / 20.0 * val
        val, cnt = smoothed_logit(gv.get(v), global_rate, alpha=20.0)
        s += 0.25 * min(cnt, 50.0) / 50.0 * val
        val, cnt = smoothed_logit(ga.get(a), global_rate, alpha=20.0)
        s += 0.15 * min(cnt, 50.0) / 50.0 * val
        out[i] = s
    return per_user_z(out, groups)


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
    parent_score = 0.70 * score_z + 0.30 * score_rank

    hist_score = history_affinity_score(splits, target, groups)
    scores = 0.70 * parent_score + 0.30 * hist_score

    if a.out:
        np.save(a.out, scores.astype(np.float64))
    else:
        print('wrote predictions only when --out is supplied')
