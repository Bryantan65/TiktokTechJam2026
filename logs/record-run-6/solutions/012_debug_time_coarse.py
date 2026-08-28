"""Debug node 11: coarse, key-aligned temporal context only.

Node 11 added high-cardinality absolute date/day fields and aligned time by a
fragile raw-row slice.  This version keeps the same best history+aux FM but adds
only low-cardinality hour-of-day and 4-hour block fields, aligned to load() rows
by the same duplicate-preserving key/deque method used for raw auxiliary data.
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

AUX_NAMES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']


def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def _raw_log_paths(data_dir):
    paths = [
        os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv'),
    ]
    if all(os.path.exists(p) for p in paths):
        return paths
    return [
        os.path.join(data_dir, 'data', 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'data', 'log_standard_4_22_to_5_08_pure.csv'),
    ]


def _row_key_from_raw(r):
    return (_to_int(r.get('date')), _to_int(r.get('user_id')), _to_int(r.get('video_id')),
            _to_int(r.get('author_id')), _to_int(r.get('tab')), _to_int(r.get('duration_ms')))


def _row_key(row):
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))


def read_time_features(data_dir, splits):
    """Return split -> int matrix [hour, 4-hour block], duplicate-order aligned."""
    by_key = defaultdict(deque)
    for path in _raw_log_paths(data_dir):
        if not os.path.exists(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                hm = _to_int(r.get('hourmin'), 0)
                hour = max(0, min(23, hm // 100))
                by_key[_row_key_from_raw(r)].append((hour, hour // 4))
    out = {}
    for sp, rows in splits.items():
        arr = np.zeros((len(rows), 2), dtype=np.int64)
        miss = 0
        for i, row in enumerate(rows):
            q = by_key.get(_row_key(row))
            if q:
                arr[i] = q.popleft()
            else:
                miss += 1
                arr[i] = (0, 0)
        if miss:
            print(f"warning: {miss} raw time rows missing for {sp}")
        out[sp] = arr
    return out


def add_time_fields(enc, splits, data_dir):
    times = read_time_features(data_dir, splits)
    offset = int(max(v[0].max() for v in enc.values())) + 1
    maps = []
    for j in range(times['train'].shape[1]):
        vals = np.unique(times['train'][:, j])
        mp = {int(v): offset + i + 1 for i, v in enumerate(vals)}
        unk = offset
        maps.append((mp, unk))
        offset += len(vals) + 1
    out = {}
    for sp, (X, y, u) in enc.items():
        extra = np.zeros((len(X), len(maps)), dtype=np.int64)
        for j, (mp, unk) in enumerate(maps):
            col = times[sp][:, j]
            extra[:, j] = [mp.get(int(v), unk) for v in col]
        out[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, u)
    return out, offset


def read_aux_targets(data_dir, splits):
    by_key = defaultdict(deque)
    for path in _raw_log_paths(data_dir):
        if not os.path.exists(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd:
                dur = max(_to_int(r.get('duration_ms')), 1)
                play = max(_to_int(r.get('play_time_ms')), 0)
                vals = []
                for name in AUX_NAMES:
                    vals.append(float(_to_int(r.get(name), 0) > 0))
                ratio = play / float(dur)
                vals.append(float(ratio >= 0.25))
                vals.append(float(ratio >= 0.50))
                vals.append(float(ratio >= 1.00))
                by_key[_row_key_from_raw(r)].append(np.asarray(vals, dtype=np.float32))
    out = {}
    dim = len(AUX_NAMES) + 3
    for sp, rows in splits.items():
        arr = np.zeros((len(rows), dim), dtype=np.float32)
        miss = 0
        for i, row in enumerate(rows):
            q = by_key.get(_row_key(row))
            if q:
                arr[i] = q.popleft()
            else:
                miss += 1
        if miss:
            print(f"warning: {miss} raw auxiliary rows missing for {sp}")
        out[sp] = arr
    return out


class HistState:
    def __init__(self):
        self.ui = defaultdict(int)
        self.up = defaultdict(int)
        self.uai = defaultdict(int)
        self.uap = defaultdict(int)
        self.uvi = defaultdict(int)
        self.uvp = defaultdict(int)
        self.uti = defaultdict(int)
        self.utp = defaultdict(int)
        self.udi = defaultdict(int)
        self.udp = defaultdict(int)
        self.ai = defaultdict(int)
        self.ap = defaultdict(int)
        self.vi = defaultdict(int)
        self.vp = defaultdict(int)

    def copy(self):
        other = HistState()
        for name in ('ui', 'up', 'uai', 'uap', 'uvi', 'uvp', 'uti', 'utp',
                     'udi', 'udp', 'ai', 'ap', 'vi', 'vp'):
            setattr(other, name, defaultdict(int, getattr(self, name).copy()))
        return other

    def features_one(self, row):
        u, v, a, tab = row[1], row[2], row[3], row[4]
        dur = int(row[5]) // 10000
        ui, up = self.ui[u], self.up[u]
        uai, uap = self.uai[(u, a)], self.uap[(u, a)]
        uvi, uvp = self.uvi[(u, v)], self.uvp[(u, v)]
        uti, utp = self.uti[(u, tab)], self.utp[(u, tab)]
        udi, udp = self.udi[(u, dur)], self.udp[(u, dur)]
        ai, ap = self.ai[a], self.ap[a]
        vi, vp = self.vi[v], self.vp[v]
        return [
            np.log1p(ui), (up + 1.0) / (ui + 2.0),
            np.log1p(uai), (uap + 0.5) / (uai + 2.0), uap / (up + 1.0),
            np.log1p(uvi), (uvp + 0.5) / (uvi + 2.0),
            np.log1p(uti), (utp + 0.5) / (uti + 2.0),
            np.log1p(udi), (udp + 0.5) / (udi + 2.0),
            np.log1p(ai), (ap + 1.0) / (ai + 2.0),
            np.log1p(vi), (vp + 0.5) / (vi + 2.0),
        ]

    def update(self, row):
        u, v, a, tab = row[1], row[2], row[3], row[4]
        dur = int(row[5]) // 10000
        y = 1 if row[6] > 0 else 0
        self.ui[u] += 1; self.up[u] += y
        self.uai[(u, a)] += 1; self.uap[(u, a)] += y
        self.uvi[(u, v)] += 1; self.uvp[(u, v)] += y
        self.uti[(u, tab)] += 1; self.utp[(u, tab)] += y
        self.udi[(u, dur)] += 1; self.udp[(u, dur)] += y
        self.ai[a] += 1; self.ap[a] += y
        self.vi[v] += 1; self.vp[v] += y


def make_history_features(splits):
    state = HistState()
    feats = {}
    tr = np.empty((len(splits['train']), 15), dtype=np.float32)
    for i, row in enumerate(splits['train']):
        tr[i] = state.features_one(row)
        state.update(row)
    feats['train'] = tr
    train_state = state.copy()
    for sp in ('valid', 'test'):
        st = train_state.copy()
        arr = np.empty((len(splits[sp]), 15), dtype=np.float32)
        for i, row in enumerate(splits[sp]):
            arr[i] = st.features_one(row)
        feats[sp] = arr
    mu = feats['train'].mean(axis=0)
    sd = feats['train'].std(axis=0) + 1e-6
    for sp in feats:
        feats[sp] = (feats[sp] - mu) / sd
    return feats


class MultiTaskHistFM(torch.nn.Module):
    def __init__(self, dim, hist_dim, aux_dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.H = torch.nn.Linear(hist_dim, 1)
        torch.nn.init.zeros_(self.H.weight)
        torch.nn.init.zeros_(self.H.bias)
        self.aux = torch.nn.Sequential(
            torch.nn.Linear(k + hist_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, aux_dim),
        )

    def forward(self, X, H, aux=False):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        main = self.b + self.W[X].sum(1) + inter + self.H(H).squeeze(1)
        if aux:
            return main, self.aux(torch.cat([S, H], dim=1))
        return main

    @torch.no_grad()
    def predict(self, X, H, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            hb = torch.from_numpy(H[i:i + bs].astype(np.float32)).to(device)
            out.append(self(xb, hb).cpu().numpy())
        return np.concatenate(out)


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    base_enc, _ = encode(splits)
    enc, dim = add_time_fields(base_enc, splits, data_dir)
    hist = make_history_features(splits)
    aux_t = read_aux_targets(data_dir, splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Htr, Hva = hist['train'], hist['valid']
    Atr = aux_t['train']

    model = MultiTaskHistFM(dim, Htr.shape[1], Atr.shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0},
                            {'params': model.H.parameters(), 'weight_decay': 1e-5},
                            {'params': model.aux.parameters(), 'weight_decay': 1e-5}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bce = torch.nn.BCEWithLogitsLoss()
    aux_bce = torch.nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Htr_t = torch.from_numpy(Htr.astype(np.float32))
    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    Atr_t = torch.from_numpy(Atr.astype(np.float32))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    aux_weight = 0.08

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            hb = Htr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            ab = Atr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            main_logit, aux_logit = model(xb, hb, aux=True)
            loss = bce(main_logit, yb) + aux_weight * aux_bce(aux_logit, ab)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, Hva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
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
    return model, enc, hist


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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+coarse_time")

    model, enc, hist = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                           seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[a.split]
    scores = model.predict(X, hist[a.split], device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== debug_time_coarse (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, hist[sp], device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
