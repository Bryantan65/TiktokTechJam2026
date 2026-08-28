"""Seven-member same-user BPR FM ensemble with temporal context fields.

Drafts direction 6 on top of the best BPR rank ensemble: read hourmin from the
raw KuaiRand logs, align it back to data.load() rows, and append reusable time
context fields (hour, 4-hour block, weekday, tab-hour, tab-weekday) to the FM.
The training loss and within-user rank/Borda ensemble are unchanged from 010.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


RAW_LOGS = [
    'log_standard_4_08_to_4_21_pure.csv',
    'log_standard_4_22_to_5_08_pure.csv',
]


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
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


def _pick(rec, names, default=None):
    for n in names:
        if n in rec and rec[n] != '':
            return rec[n]
    return default


def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def _weekday(date_int):
    try:
        s = str(int(date_int))
        return _dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).weekday()
    except Exception:
        return 0


def _time_feats(date_int, hourmin, tab):
    hm = _to_int(hourmin, 0)
    hour = max(0, min(23, hm // 100))
    hblock = hour // 4
    dow = _weekday(date_int)
    tab = _to_int(tab, 0)
    return (hour, hblock, dow, tab * 24 + hour, tab * 7 + dow)


def _raw_log_path(data_dir, filename):
    p = os.path.join(data_dir, filename)
    if os.path.exists(p):
        return p
    # Be tolerant if the caller passes the dataset root instead of the data dir.
    p2 = os.path.join(data_dir, 'data', filename)
    if os.path.exists(p2):
        return p2
    return p


def read_raw_time_lookup(data_dir):
    """Return key -> queue of temporal features in original raw-log order."""
    lookup = defaultdict(deque)
    n = 0
    for fn in RAW_LOGS:
        path = _raw_log_path(data_dir, fn)
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for rec in reader:
                date = _to_int(_pick(rec, ['date']))
                user = _to_int(_pick(rec, ['user_id', 'userid', 'user']))
                video = _to_int(_pick(rec, ['video_id', 'videoid', 'item_id', 'itemid']))
                author = _to_int(_pick(rec, ['author_id', 'authorid']))
                tab = _to_int(_pick(rec, ['tab']))
                dur = _to_int(_pick(rec, ['duration_ms', 'video_duration_ms',
                                          'duration', 'video_duration']))
                hourmin = _pick(rec, ['hourmin', 'hour_min', 'time_ms', 'time'], 0)
                key = (date, user, video, author, tab, dur)
                lookup[key].append(_time_feats(date, hourmin, tab))
                n += 1
    return lookup, n


def aligned_time_features(data_dir, splits):
    lookup, _nraw = read_raw_time_lookup(data_dir)
    metas = {}
    missing = 0
    for sp, rows in splits.items():
        arr = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            key = tuple(_to_int(row[j]) for j in range(6))
            q = lookup.get(key)
            if q:
                arr[i] = q.popleft()
            else:
                # Fallback uses date/tab from the tuple and an unknown/zero hour.
                missing += 1
                date, tab = _to_int(row[0]), _to_int(row[4])
                arr[i] = _time_feats(date, 0, tab)
        metas[sp] = arr
    if missing:
        print(f"warning: {missing} rows could not be aligned to raw hourmin; used hour=0 fallback")
    return metas


def augment_encoded(enc, time_meta):
    """Append categorical temporal fields to the starter-kit encoding."""
    n_extra = next(iter(time_meta.values())).shape[1]
    maps = []
    offset = max(int(v[0].max()) for v in enc.values()) + 1
    for j in range(n_extra):
        vals = np.concatenate([time_meta[sp][:, j] for sp in ('train', 'valid', 'test')])
        uniq = sorted(set(int(x) for x in vals))
        mp = {v: offset + i for i, v in enumerate(uniq)}
        maps.append(mp)
        offset += len(uniq)

    out = {}
    for sp, (X, y, users) in enc.items():
        extra = np.zeros((len(X), n_extra), dtype=np.int64)
        for j, mp in enumerate(maps):
            # Values are known because maps are built over all splits.
            extra[:, j] = [mp[int(v)] for v in time_meta[sp][:, j]]
        out[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, users)
    return out, offset


def encode_with_time(data_dir, splits):
    enc0, _dim0 = encode(splits)
    meta = aligned_time_features(data_dir, splits)
    return augment_encoded(enc0, meta)


def make_user_pairs(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]

    pos_lists, neg_lists = [], []
    n_pos = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_lists.append(pos.astype(np.int64))
            neg_lists.append(neg.astype(np.int64))
            n_pos += len(pos)
    return pos_lists, neg_lists, n_pos


def sample_epoch_pairs(pos_lists, neg_lists, n_pos, rng):
    pos_all = np.empty(n_pos, dtype=np.int64)
    neg_all = np.empty(n_pos, dtype=np.int64)
    off = 0
    for pos, neg in zip(pos_lists, neg_lists):
        m = len(pos)
        pos_all[off:off + m] = pos
        neg_all[off:off + m] = rng.choice(neg, size=m, replace=True)
        off += m
    perm = rng.permutation(n_pos)
    return pos_all[perm], neg_all[perm]


def train_one(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=False, tag=''):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_lists, neg_lists, n_pos = make_user_pairs(ytr, enc['train'][2])
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    if verbose:
        print(f"{tag} BPR users={len(pos_lists):,d} positives paired/epoch={n_pos:,d}")

    for ep in range(1, epochs + 1):
        pos_idx, neg_idx = sample_epoch_pairs(pos_lists, neg_lists, n_pos, rng)
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, n_pos, bs):
            psel = torch.from_numpy(pos_idx[i:i + bs])
            nsel = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

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


def train_ensemble(splits, data_dir, k=16, lr=0.001, epochs=40, seed=0,
                   device='cpu', verbose=True):
    enc, dim = encode_with_time(data_dir, splits)
    seeds = [seed, seed + 1009, seed + 2027, seed + 3037, seed + 4051,
             seed + 5059, seed + 6073]
    models = []
    for j, s in enumerate(seeds):
        torch.manual_seed(s)
        m = train_one(enc, dim, k=k, lr=lr, epochs=epochs, seed=s,
                      device=device, verbose=verbose, tag=f"m{j+1}/7")
        models.append(m)
    return models, enc


def within_user_rank_scores(scores, users):
    """Map scores to [0,1] ranks within each user; higher score -> higher rank."""
    scores = np.asarray(scores)
    users = np.asarray(users)
    out = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        m = e - s
        if m <= 1:
            out[idx] = 0.0
            continue
        vals = scores[idx]
        ord_local = np.argsort(vals, kind='mergesort')
        ranks = np.empty(m, dtype=np.float64)
        ranks[ord_local] = np.arange(m, dtype=np.float64) / (m - 1.0)
        out[idx] = ranks
    return out


@torch.no_grad()
def ensemble_predict(models, X, users, device='cpu'):
    pred = np.zeros(len(X), dtype=np.float64)
    for m in models:
        raw = m.predict(X, device=device)
        pred += within_user_rank_scores(raw, users)
    pred /= len(models)
    return pred


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}+['hour','hblock','weekday','tab_hour','tab_weekday']")

    models, enc = train_ensemble(splits, a.data_dir, k=a.k, lr=a.lr,
                                 epochs=a.epochs, seed=a.seed,
                                 device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = ensemble_predict(models, X, users, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr7_time_context (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, ensemble_predict(models, Xs, us, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
