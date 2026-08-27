"""Sampled-softmax FM with coarse time-context categorical features.

Builds on the current best loss/model, but changes direction by adding robustly
computed time buckets (hour/date/hour-of-week when raw timestamp columns exist).
The script is standalone and falls back to the original fields if no usable time
column is present.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, FIELDS          # noqa: E402
from evaluate import evaluate          # noqa: E402


BASE_FIELDS = list(FIELDS)


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


def _find_label_col(df):
    for c in ('long_view', 'is_long_view', 'label', 'is_click'):
        if c in df.columns:
            return c
    # starter-kit FIELDS excludes labels; choose a binary target-like fallback
    for c in df.columns:
        if c not in BASE_FIELDS and df[c].dropna().isin([0, 1]).all():
            return c
    raise RuntimeError('Could not identify label column')


def _find_user_col(df):
    for c in ('user_id', 'userId', 'user'):
        if c in df.columns:
            return c
    return BASE_FIELDS[0]


def _find_time_col(all_dfs):
    names = []
    for df in all_dfs:
        names.extend(list(df.columns))
    cols = list(dict.fromkeys(names))
    preferred = ['time_ms', 'timestamp', 'ts', 'request_time', 'date_time', 'datetime',
                 'time', 'hourmin', 'date']
    for c in preferred:
        if c in cols:
            return c
    # Last resort: any column name containing time/date, excluding play_time/duration.
    for c in cols:
        lc = c.lower()
        if ('time' in lc or 'date' in lc or 'hour' in lc) and 'play' not in lc and 'duration' not in lc:
            return c
    return None


def _as_datetime_series(s):
    # Numeric Unix timestamps are common. Try ms, seconds, then pandas generic.
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors='coerce')
        med = vals.dropna().median() if vals.notna().any() else np.nan
        if np.isfinite(med):
            if med > 1e12:
                return pd.to_datetime(vals, unit='ms', errors='coerce')
            if med > 1e9:
                return pd.to_datetime(vals, unit='s', errors='coerce')
        return vals
    return pd.to_datetime(s, errors='coerce')


def add_time_features(splits, verbose=True):
    out = {k: v.copy() for k, v in splits.items()}
    tcol = _find_time_col(list(out.values()))
    new_fields = []
    if tcol is None:
        if verbose:
            print('no time column found; using base fields only')
        return out, BASE_FIELDS

    # If hourmin is already a compact minute/hour code, use it directly and also coarsen.
    for name, df in out.items():
        raw = df[tcol]
        parsed = _as_datetime_series(raw)
        if np.issubdtype(getattr(parsed, 'dtype', object), np.datetime64):
            dt = parsed
            df['time_hour'] = dt.dt.hour.fillna(-1).astype(np.int16)
            df['time_dow'] = dt.dt.dayofweek.fillna(-1).astype(np.int16)
            df['time_how'] = (df['time_dow'].astype(np.int32) * 24 + df['time_hour'].astype(np.int32)).astype(np.int16)
            # Relative day bucket keeps drift/order info without relying on absolute string categories.
            day_num = (dt.astype('int64') // (24 * 3600 * 10**9)).where(dt.notna(), -1)
            df['time_day'] = pd.Series(day_num, index=df.index).astype(np.int64)
        else:
            vals = pd.to_numeric(parsed, errors='coerce').fillna(-1).astype(np.int64)
            # Common hourmin format can be HHMM or minute-of-day. Otherwise quantile-like modulo buckets.
            if tcol.lower() == 'hourmin' or vals.between(0, 2359).mean() > 0.9:
                hour = (vals // 100).clip(-1, 23)
                minute = (vals % 100).clip(0, 59)
                df['time_hour'] = hour.astype(np.int16)
                df['time_how'] = ((hour * 2) + (minute >= 30).astype(np.int64)).astype(np.int16)
            else:
                df['time_hour'] = (vals % 24).astype(np.int16)
                df['time_how'] = (vals % 168).astype(np.int16)
            df['time_day'] = (vals // 24).astype(np.int64)

    # Make day relative to global minimum nonnegative day to reduce category spread.
    days = pd.concat([out[k]['time_day'] for k in out], axis=0)
    valid_days = days[days >= 0]
    if len(valid_days):
        min_day = int(valid_days.min())
        for df in out.values():
            df['time_day'] = np.where(df['time_day'].values >= 0,
                                      df['time_day'].values - min_day, -1).astype(np.int16)

    # Keep only compact time fields; raw timestamp itself can overfit badly.
    for c in ('time_hour', 'time_dow', 'time_how', 'time_day'):
        if all(c in df.columns for df in out.values()):
            new_fields.append(c)
    if verbose:
        print(f'time column={tcol}; added fields={new_fields}')
    return out, BASE_FIELDS + new_fields


def encode_with_fields(splits, fields):
    maps = {}
    offsets = {}
    dim = 0
    # Fit on all splits just like starter encode, so validation/test unseen IDs get valid indices.
    for f in fields:
        vals = pd.concat([splits[k][f] for k in splits], axis=0).astype('category')
        cats = vals.cat.categories
        maps[f] = {v: i for i, v in enumerate(cats)}
        offsets[f] = dim
        dim += len(cats)

    enc = {}
    label_col = _find_label_col(next(iter(splits.values())))
    user_col = _find_user_col(next(iter(splits.values())))
    for name, df in splits.items():
        X = np.empty((len(df), len(fields)), dtype=np.int64)
        for j, f in enumerate(fields):
            mp = maps[f]
            X[:, j] = df[f].map(mp).fillna(0).astype(np.int64).values + offsets[f]
        y = df[label_col].astype(np.float32).values
        u = df[user_col].values
        enc[name] = (X, y, u)
    return enc, dim


def build_user_pair_pools(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    cuts = np.flatnonzero(su[1:] != su[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(order)]

    pos_chunks, neg_chunks = [], []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        yy = y[idx] > 0.5
        if yy.any() and (~yy).any():
            pos_chunks.append(idx[yy].astype(np.int64, copy=False))
            neg_chunks.append(idx[~yy].astype(np.int64, copy=False))
    return pos_chunks, neg_chunks


def sample_sets(pos_chunks, neg_chunks, rng, negs_per_pos=2):
    base = sum(len(p) for p in pos_chunks)
    pos = np.empty(base, dtype=np.int64)
    negs = np.empty((base, negs_per_pos), dtype=np.int64)
    off = 0
    for p, n in zip(pos_chunks, neg_chunks):
        m = len(p)
        pos[off:off + m] = p
        for j in range(negs_per_pos):
            negs[off:off + m, j] = n[rng.integers(0, len(n), size=m)]
        off += m
    perm = rng.permutation(base)
    return pos[perm], negs[perm]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, aux_weight=0.025, negs_per_pos=2):
    splits2, fields = add_time_features(splits, verbose=verbose)
    enc, dim = encode_with_fields(splits2, fields)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_chunks, neg_chunks = build_user_pair_pools(ytr, utr)
    if verbose:
        print(f'fields={fields}')
        print(f"eligible users={len(pos_chunks):,d}; sets/epoch={sum(len(p) for p in pos_chunks):,d}; "
              f"negs_per_pos={negs_per_pos}; aux_weight={aux_weight}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        pidx, nidx = sample_sets(pos_chunks, neg_chunks, rng, negs_per_pos=negs_per_pos)
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(pidx), bs):
            ps_np = pidx[i:i + bs]
            ns_np = nidx[i:i + bs]
            ps = torch.from_numpy(ps_np)
            ns = torch.from_numpy(ns_np.reshape(-1))
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn).view(-1, negs_per_pos)
            logits = torch.cat([sp[:, None], sn], dim=1)
            rank_loss = -torch.nn.functional.log_softmax(logits, dim=1)[:, 0].mean()
            bce_pos = torch.nn.functional.softplus(-sp).mean()
            bce_neg = torch.nn.functional.softplus(sn).mean()
            loss = rank_loss + aux_weight * 0.5 * (bce_pos + bce_neg)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | softmax+bce {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, enc


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
    print({k_: len(v) for k_, v in splits.items()}, f"base_fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== time_softmax_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
