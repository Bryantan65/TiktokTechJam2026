"""Time-aware five-seed node-8 BCE/BPR/rank FM ensemble.

Adds temporal categorical fields to the same weighted five-seed BCE+BPR FM
ensemble as node 13: date, day-of-week, and a 4-hour bucket read from the raw
KuaiRand log CSV hourmin column when available.  Member cache keys are new
because the FM feature space changed.
"""
import argparse
import csv
import datetime as _dt
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402  early stopping only


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


def fit_bce(splits, enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train(); losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BCE(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def make_user_pair_sources(y, users):
    pos = defaultdict(list); neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)
    pos_idx = []; neg_pools = []
    for uu, ps in pos.items():
        ns = neg.get(uu)
        if ns:
            arr = np.asarray(ns, dtype=np.int64)
            for p in ps:
                pos_idx.append(p); neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def sample_uniform_pairs(pos_idx, neg_pools, rng, neg_per_pos=3):
    total = len(pos_idx) * neg_per_pos
    p_out = np.empty(total, dtype=np.int64)
    n_out = np.empty(total, dtype=np.int64)
    k = 0
    for p, pool in zip(pos_idx, neg_pools):
        m = len(pool)
        for _ in range(neg_per_pos):
            p_out[k] = p
            n_out[k] = pool[rng.integers(0, m)]
            k += 1
    order = rng.permutation(total)
    return p_out[order], n_out[order]


def fit_bpr(splits, enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, neg_per_pos=3, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, neg_pools = make_user_pair_sources(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        p_idx, n_idx = sample_uniform_pairs(pos_idx, neg_pools, rng, neg_per_pos=neg_per_pos)
        model.train(); losses = []
        for i in range(0, len(p_idx), bs):
            ps = torch.from_numpy(p_idx[i:i + bs])
            ns = torch.from_numpy(n_idx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp) - model(xn))).mean()
            loss.backward(); opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BPR(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | pairs {len(p_idx):,d} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def user_groups(users):
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def per_user_zscore(scores, groups):
    scores = scores.astype(np.float64, copy=False)
    out = np.empty_like(scores, dtype=np.float64)
    for idx in groups:
        s = scores[idx]
        sd = s.std()
        if sd > 1e-12:
            out[idx] = (s - s.mean()) / sd
        else:
            out[idx] = s - s.mean()
    return out


def per_user_rank01(scores, groups):
    scores = scores.astype(np.float64, copy=False)
    out = np.empty_like(scores, dtype=np.float64)
    for idx in groups:
        n = len(idx)
        if n <= 1:
            out[idx] = 0.0
            continue
        order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64) / (n - 1.0)
        out[idx] = ranks
    return out


def _norm(x):
    s = '' if x is None else str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def _first(rec, names):
    for n in names:
        if n in rec and rec[n] != '':
            return rec[n]
    return None


def _hour_bucket(hourmin):
    if hourmin is None:
        return 'hUNK'
    s = str(hourmin).strip()
    if not s:
        return 'hUNK'
    try:
        if ':' in s:
            h = int(s.split(':', 1)[0])
        else:
            v = int(float(s))
            h = v // 100 if v >= 100 else v
        if 0 <= h <= 23:
            return 'h%02d' % (h // 4)
    except Exception:
        pass
    return 'hUNK'


def build_hour_maps(data_dir):
    """Raw log order is preserved; keep occurrence lists for duplicate keys."""
    full = defaultdict(list)
    short = defaultdict(list)
    files = ['log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv']
    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                rd = csv.DictReader(f)
                for rec in rd:
                    date = _norm(_first(rec, ['date']))
                    uid = _norm(_first(rec, ['user_id', 'userId', 'user']))
                    vid = _norm(_first(rec, ['video_id', 'videoId', 'item_id', 'itemId']))
                    aid = _norm(_first(rec, ['author_id', 'authorId']))
                    tab = _norm(_first(rec, ['tab']))
                    dur = _norm(_first(rec, ['duration_ms', 'duration', 'video_duration']))
                    hb = _hour_bucket(_first(rec, ['hourmin', 'hour_min', 'time', 'timestamp']))
                    if date and uid and vid and tab and dur:
                        short[(date, uid, vid, tab, dur)].append(hb)
                        if aid:
                            full[(date, uid, vid, aid, tab, dur)].append(hb)
        except Exception as e:
            print(f"warning: failed to read {path}: {e}")
    return full, short


def _date_feats(d):
    ds = _norm(d)
    try:
        dt = _dt.datetime.strptime(ds, '%Y%m%d').date()
        return 'd' + ds, 'dow%d' % dt.weekday()
    except Exception:
        return 'd' + ds, 'dowUNK'


def augment_with_time(splits, enc_base, dim_base, data_dir):
    full, short = build_hour_maps(data_dir)
    full_pos = defaultdict(int); short_pos = defaultdict(int)
    extras = {}
    hit = 0; total = 0
    for name, rows in splits.items():
        cur = []
        for r in rows:
            date, uid, vid, aid, tab, dur = r[0], r[1], r[2], r[3], r[4], r[5]
            dcat, dow = _date_feats(date)
            fkey = (_norm(date), _norm(uid), _norm(vid), _norm(aid), _norm(tab), _norm(dur))
            skey = (_norm(date), _norm(uid), _norm(vid), _norm(tab), _norm(dur))
            hb = None
            p = full_pos[fkey]
            if p < len(full.get(fkey, [])):
                hb = full[fkey][p]
                full_pos[fkey] += 1
            else:
                p = short_pos[skey]
                if p < len(short.get(skey, [])):
                    hb = short[skey][p]
                    short_pos[skey] += 1
            if hb is None:
                hb = 'hUNK'
            else:
                hit += 1
            total += 1
            cur.append((dcat, dow, hb))
        extras[name] = cur
    print(f"time feature hour lookup hits: {hit}/{total}")

    maps = []
    for j in range(3):
        mp = {}
        for name in splits.keys():
            for tup in extras[name]:
                v = tup[j]
                if v not in mp:
                    mp[v] = len(mp)
        maps.append(mp)

    offsets = []
    off = dim_base
    for mp in maps:
        offsets.append(off)
        off += len(mp)

    enc = {}
    for name, (X, y, u) in enc_base.items():
        extra_arr = np.empty((len(X), 3), dtype=np.int64)
        for i, tup in enumerate(extras[name]):
            for j in range(3):
                extra_arr[i, j] = offsets[j] + maps[j][tup[j]]
        enc[name] = (np.hstack([X.astype(np.int64), extra_arr]), y, u)
    print(f"augmented fields={FIELDS + ['date', 'dow', 'hour4']} dim {dim_base}->{off}")
    return enc, off


def cached_predict(member_name, train_fn, enc, target, member_seed, device, use_cache=True):
    os.makedirs('pred_cache', exist_ok=True)
    X, _, _ = enc[target]
    cache_path = os.path.join('pred_cache', f'014_timev1_{member_name}_{target}_seed{member_seed}.npy')
    if use_cache and os.path.isfile(cache_path):
        preds = np.load(cache_path)
        if len(preds) == len(X):
            return preds.astype(np.float64, copy=False)
    model = train_fn()
    preds = model.predict(X, device=device).astype(np.float64)
    if use_cache:
        np.save(cache_path, preds)
    return preds


def node8_seed_score(bce_preds, bpr_preds, groups):
    zblend = 0.35 * per_user_zscore(bce_preds, groups) + 0.65 * per_user_zscore(bpr_preds, groups)
    rblend = 0.35 * per_user_rank01(bce_preds, groups) + 0.65 * per_user_rank01(bpr_preds, groups)
    return 0.70 * zblend + 0.30 * rblend


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--neg_per_pos', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
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
    print({kk: len(vv) for kk, vv in splits.items()}, f"fields={FIELDS}")

    enc_base, dim_base = encode(splits)
    enc, dim = augment_with_time(splits, enc_base, dim_base, a.data_dir)
    verbose = a.out is None
    use_cache = a.out is not None and a.split != 'dev'
    X, y, users = enc[target]
    groups = user_groups(users)

    member_seeds = [0, 1, 2, 3, 4]
    weights = np.asarray([0.12, 0.12, 0.12, 0.32, 0.32], dtype=np.float64)
    seed_scores = []
    for ms in member_seeds:
        bce_preds = cached_predict(
            'bce',
            lambda ms=ms: fit_bce(splits, enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                  seed=ms, device=a.device, verbose=verbose),
            enc, target, ms, a.device, use_cache=use_cache)
        bpr_preds = cached_predict(
            'bpr_uniform_np3',
            lambda ms=ms: fit_bpr(splits, enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                  neg_per_pos=a.neg_per_pos, seed=ms,
                                  device=a.device, verbose=verbose),
            enc, target, ms, a.device, use_cache=use_cache)
        s = node8_seed_score(bce_preds, bpr_preds, groups)
        seed_scores.append(per_user_zscore(s, groups))

    score_mat = np.vstack(seed_scores)
    bag = weights @ score_mat
    cur_idx = member_seeds.index(a.seed) if a.seed in member_seeds else (a.seed % len(member_seeds))
    scores = 0.995 * bag + 0.005 * seed_scores[cur_idx]

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== time_features_seedbag5 (seed={a.seed}, device={a.device}) ===")
        r = evaluate(users, y, scores)
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
