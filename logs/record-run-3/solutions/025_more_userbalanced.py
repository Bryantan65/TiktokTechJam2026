"""Time/watch FM BPR ensemble with three weak user-balanced BPR members.

Improve node 24: the first user-balanced same-user BPR member moved both GAUC
and nDCG upward.  Keep the node-24 recipe but reduce per-member weight and use
three independently initialized user-balanced members (total weight 1.5) to make
that complementary signal less seed-specific without letting it dominate the
GAUC-weighted label ensemble.
"""
import argparse
import csv
import datetime as _dt
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402
from evaluate import evaluate  # noqa: E402

RAW_LOGS = ['log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv']


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
    def predict(self, X, bs=200000, device='cpu'):
        self.eval(); out = []
        for i in range(0, len(X), bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)


def _pick(rec, names, default=None):
    for n in names:
        if n in rec and rec[n] != '':
            return rec[n]
    return default


def _to_int(x, default=0):
    try: return int(float(x))
    except Exception: return default


def _to_float(x, default=0.0):
    try: return float(x)
    except Exception: return default


def _weekday(date_int):
    try:
        s = str(int(date_int)); return _dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).weekday()
    except Exception:
        return 0


def _time_feats(date_int, hourmin, tab):
    hm = _to_int(hourmin, 0); hour = max(0, min(23, hm // 100))
    dow = _weekday(date_int); tab = _to_int(tab, 0)
    return (hour, hour // 4, dow, tab * 24 + hour, tab * 7 + dow)


def _raw_log_path(data_dir, filename):
    for p in (os.path.join(data_dir, filename), os.path.join(data_dir, 'data', filename), os.path.join(os.path.dirname(data_dir), 'data', filename)):
        if os.path.exists(p): return p
    return os.path.join(data_dir, filename)


def read_raw_lookup(data_dir):
    lookup = defaultdict(deque); n = 0; headers = None
    for fn in RAW_LOGS:
        with open(_raw_log_path(data_dir, fn), 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f); headers = reader.fieldnames if headers is None else headers
            for rec in reader:
                date = _to_int(_pick(rec, ['date']))
                user = _to_int(_pick(rec, ['user_id', 'userid', 'user']))
                video = _to_int(_pick(rec, ['video_id', 'videoid', 'item_id', 'itemid']))
                tab = _to_int(_pick(rec, ['tab', 'tab_id']))
                dur = _to_int(_pick(rec, ['duration_ms', 'video_duration_ms', 'duration', 'video_duration']))
                y = _to_int(_pick(rec, ['long_view', 'label'], 0))
                hourmin = _pick(rec, ['hourmin', 'hour_min', 'time', 'request_time'], 0)
                play = _to_float(_pick(rec, ['play_time_ms'], 0.0), 0.0)
                play_ratio = min(max(play / max(float(dur), 1000.0), 0.0), 1.0)
                lookup[(date, user, video, tab, dur, y)].append((_time_feats(date, hourmin, tab), play_ratio))
                n += 1
    return lookup, n, headers


def aligned_raw_features(data_dir, splits):
    lookup, nraw, headers = read_raw_lookup(data_dir)
    time_meta, play_meta = {}, {}; exact = miss = 0
    for sp, rows in splits.items():
        tarr = np.zeros((len(rows), 5), dtype=np.int64); parr = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            date, user, video, tab, dur, y = _to_int(row[0]), _to_int(row[1]), _to_int(row[2]), _to_int(row[4]), _to_int(row[5]), _to_int(row[6])
            q = lookup.get((date, user, video, tab, dur, y))
            if q:
                tf, pr = q.popleft(); tarr[i] = tf; parr[i] = pr; exact += 1
            else:
                tarr[i] = _time_feats(date, 0, tab); miss += 1
        time_meta[sp] = tarr; play_meta[sp] = parr
    print(f"raw alignment: exact={exact} missing={miss} raw={nraw} +play_ratio headers={headers}")
    return time_meta, play_meta


def augment_encoded(enc, time_meta):
    offset = max(int(v[0].max()) for v in enc.values()) + 1
    maps = []
    for j in range(next(iter(time_meta.values())).shape[1]):
        vals = np.concatenate([time_meta[sp][:, j] for sp in ('train', 'valid', 'test')])
        uniq = sorted(set(int(x) for x in vals)); mp = {v: offset+i for i, v in enumerate(uniq)}
        maps.append(mp); offset += len(uniq)
    out = {}
    for sp, (X, y, users) in enc.items():
        extra = np.zeros((len(X), len(maps)), dtype=np.int64)
        for j, mp in enumerate(maps): extra[:, j] = [mp[int(v)] for v in time_meta[sp][:, j]]
        out[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, users)
    return out, offset


def encode_with_raw(data_dir, splits):
    enc0, _ = encode(splits)
    time_meta, play_meta = aligned_raw_features(data_dir, splits)
    enc, dim = augment_encoded(enc0, time_meta)
    return enc, dim, play_meta


def make_user_pairs(y, users, user_balanced=False):
    y = np.asarray(y); users = np.asarray(users)
    order = np.argsort(users, kind='mergesort'); su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]; ends = np.r_[starts[1:], len(order)]
    pos_lists, neg_lists, w_lists, n_pos = [], [], [], 0
    for s, e in zip(starts, ends):
        idx = order[s:e]; pos = idx[y[idx] > 0.5]; neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_lists.append(pos.astype(np.int64)); neg_lists.append(neg.astype(np.int64)); n_pos += len(pos)
            w = np.full(len(pos), 1.0 / len(pos) if user_balanced else 1.0, dtype=np.float32)
            w_lists.append(w)
    if user_balanced and w_lists:
        scale = n_pos / sum(float(w.sum()) for w in w_lists)
        w_lists = [w * scale for w in w_lists]
    return pos_lists, neg_lists, w_lists, n_pos


def sample_epoch_pairs(pos_lists, neg_lists, w_lists, n_pos, rng):
    pos_all = np.empty(n_pos, dtype=np.int64); neg_all = np.empty(n_pos, dtype=np.int64); w_all = np.empty(n_pos, dtype=np.float32)
    off = 0
    for pos, neg, w in zip(pos_lists, neg_lists, w_lists):
        m = len(pos); pos_all[off:off+m] = pos; neg_all[off:off+m] = rng.choice(neg, size=m, replace=True); w_all[off:off+m] = w; off += m
    perm = rng.permutation(n_pos)
    return pos_all[perm], neg_all[perm], w_all[perm]


def train_one(enc, dim, play_train=None, watch_conf=False, user_balanced=False, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, device='cpu', verbose=False, tag=''):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); play_t = None if play_train is None else torch.from_numpy(play_train.astype(np.float32))
    pos_lists, neg_lists, w_lists, n_pos = make_user_pairs(ytr, utr, user_balanced=user_balanced)
    rng = np.random.default_rng(seed); best = -1.0; best_state = None; bad = 0
    if verbose: print(f"{tag} BPR users={len(pos_lists):,d} positives/epoch={n_pos:,d} watch_conf={watch_conf} user_bal={user_balanced}")
    for ep in range(1, epochs + 1):
        pos_idx, neg_idx, pair_w = sample_epoch_pairs(pos_lists, neg_lists, w_lists, n_pos, rng)
        t0 = time.time(); model.train(); losses = []
        for i in range(0, n_pos, bs):
            bp = pos_idx[i:i+bs]; bn = neg_idx[i:i+bs]
            xp = Xtr_t[torch.from_numpy(bp)].to(device); xn = Xtr_t[torch.from_numpy(bn)].to(device)
            opt.zero_grad(set_to_none=True)
            per = torch.nn.functional.softplus(-(model(xp) - model(xn)))
            w = torch.from_numpy(pair_w[i:i+bs]).to(device)
            if watch_conf and play_t is not None:
                wc = 0.35 + torch.clamp(play_t[torch.from_numpy(bp)] - play_t[torch.from_numpy(bn)], min=0.0, max=1.0)
                wc = wc / torch.clamp(wc.mean(), min=1e-6); w = w * wc.to(device)
            loss = (per * w).mean(); loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f"  {tag} epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  {tag} early stop at epoch {ep}")
                break
    model.load_state_dict(best_state); return model


def train_ensemble(splits, data_dir, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True):
    enc, dim, play_meta = encode_with_raw(data_dir, splits)
    seeds = [seed, seed+1009, seed+2027, seed+3037, seed+4051, seed+5059, seed+6073]
    models, weights = [], []
    for j, s in enumerate(seeds):
        torch.manual_seed(s); models.append(train_one(enc, dim, k=k, lr=lr, epochs=epochs, seed=s, device=device, verbose=verbose, tag=f"label{j+1}/7")); weights.append(1.0)
    s = seed + 7079; torch.manual_seed(s)
    models.append(train_one(enc, dim, play_train=play_meta['train'], watch_conf=True, k=k, lr=lr, epochs=epochs, seed=s, device=device, verbose=verbose, tag="watchconf1/1")); weights.append(0.75)
    for j, s in enumerate([seed+8089, seed+9091, seed+10093]):
        torch.manual_seed(s)
        models.append(train_one(enc, dim, user_balanced=True, k=k, lr=lr, epochs=epochs, seed=s, device=device, verbose=verbose, tag=f"userbal{j+1}/3")); weights.append(0.50)
    return models, np.asarray(weights, dtype=np.float64), enc


def within_user_rank_scores(scores, users):
    scores = np.asarray(scores); users = np.asarray(users); out = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='mergesort'); su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]; ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]; m = e - s
        if m <= 1: out[idx] = 0.0; continue
        vals = scores[idx]; ord_local = np.argsort(vals, kind='mergesort'); ranks = np.empty(m, dtype=np.float64)
        ranks[ord_local] = np.arange(m, dtype=np.float64) / (m - 1.0); out[idx] = ranks
    return out


@torch.no_grad()
def ensemble_predict(models, weights, X, users, device='cpu'):
    pred = np.zeros(len(X), dtype=np.float64); weights = np.asarray(weights, dtype=np.float64)
    for m, w in zip(models, weights): pred += w * within_user_rank_scores(m.predict(X, device=device), users)
    return pred / weights.sum()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args(); torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+time; ensemble=7label+0.75watchconf+3x0.50userbal")
    models, weights, enc = train_ensemble(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[a.split]; scores = ensemble_predict(models, weights, X, users, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== more_userbalanced (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]; r = evaluate(us, ys, ensemble_predict(models, weights, Xs, us, device=a.device))
            print(f"  {sp:5s} GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
