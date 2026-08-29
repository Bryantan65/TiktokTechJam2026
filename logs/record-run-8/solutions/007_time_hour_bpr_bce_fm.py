"""BPR+BCE FM with raw-log time-of-day features.

Starts from 003_bpr_bce_fm.py and changes only the feature matrix: append hour
and coarse hour bucket read from the raw KuaiRand log CSVs.  The starter-kit
row tuples do not contain hourmin, so we align the raw logs back to split rows by
(date,user_id,video_id,tab) with FIFO duplicate handling.
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

LOG_FILES = [
    'log_standard_4_08_to_4_21_pure.csv',
    'log_standard_4_22_to_5_08_pure.csv',
]


def _first(row, names):
    for n in names:
        if n in row:
            return row[n]
    return None


def _to_int(x, default=0):
    try:
        if x is None or x == '':
            return default
        return int(float(x))
    except Exception:
        return default


def _parse_hour(x):
    if x is None or x == '':
        return 0
    s = str(x).strip()
    try:
        # Common KuaiRand form is HHMM as an int/string, e.g. 930 or 0930.
        if ':' in s:
            h = int(s.split(':', 1)[0])
        else:
            v = int(float(s))
            h = v // 100 if v >= 100 else v
        if 0 <= h <= 23:
            return h
    except Exception:
        pass
    return 0


def key_tuple(r):
    # data.load row: (date, user_id, video_id, author_id, tab, duration_ms, label)
    return (_to_int(r[0]), str(r[1]), str(r[2]), _to_int(r[4]))


def key_csv(row):
    return (_to_int(_first(row, ['date', 'request_date', 'day'])),
            str(_first(row, ['user_id', 'userId', 'user']) or ''),
            str(_first(row, ['video_id', 'photo_id', 'item_id', 'videoId']) or ''),
            _to_int(_first(row, ['tab', 'tab_id'])))


def load_hours(data_dir, splits):
    by_key = defaultdict(deque)
    seen = 0
    have_hour = False
    for fn in LOG_FILES:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            path = os.path.join(data_dir, 'data', fn)
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            names = reader.fieldnames or []
            hname = 'hourmin' if 'hourmin' in names else None
            if hname is None:
                for cand in ['hour_min', 'time', 'timestamp']:
                    if cand in names:
                        hname = cand
                        break
            have_hour = have_hour or (hname is not None)
            for row in reader:
                seen += 1
                by_key[key_csv(row)].append(_parse_hour(row.get(hname)) if hname else 0)

    out = {}
    used = 0
    miss = 0
    for sp in [s for s in ('train', 'valid', 'test') if s in splits]:
        arr = np.zeros(len(splits[sp]), dtype=np.int16)
        for i, r in enumerate(splits[sp]):
            q = by_key.get(key_tuple(r))
            if q:
                arr[i] = q.popleft()
                used += 1
            else:
                miss += 1
        out[sp] = arr
    print(f"hour feature: have_hour={have_hour} used={used:,d} miss={miss:,d} csv_rows={seen:,d}")
    return out


def append_time_features(splits, enc, dim, data_dir):
    hours = load_hours(data_dir, splits)
    # Fit categorical ids over all provided splits, matching starter-kit encode's
    # transductive treatment of ids while keeping only train-learnable hour bins.
    vocabs = [dict(), dict(), dict()]
    vals_by_split = {}
    for sp, rows in splits.items():
        h = hours[sp]
        vals = []
        for i, r in enumerate(rows):
            hour = int(h[i])
            h6 = hour // 4                  # six 4-hour buckets
            tab_h6 = f"{_to_int(r[4])}_{h6}"
            vals.append((f"h{hour}", f"b{h6}", tab_h6))
            for j, v in enumerate(vals[-1]):
                if v not in vocabs[j]:
                    vocabs[j][v] = len(vocabs[j])
        vals_by_split[sp] = vals
    offsets = []
    cur = dim
    for v in vocabs:
        offsets.append(cur)
        cur += len(v)
    out = {}
    for sp in enc:
        X, y, u = enc[sp]
        extra = np.empty((len(X), 3), dtype=np.int64)
        for i, vals in enumerate(vals_by_split[sp]):
            for j, v in enumerate(vals):
                extra[i, j] = offsets[j] + vocabs[j][v]
        out[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, u)
    print(f"added time fields: hour={len(vocabs[0])}, hour6={len(vocabs[1])}, tab_hour6={len(vocabs[2])}; dim {dim}->{cur}")
    return out, cur


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


def build_user_groups(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)
    groups = []
    for uu, p in pos.items():
        n = neg.get(uu)
        if n:
            groups.append((np.asarray(p, dtype=np.int64), np.asarray(n, dtype=np.int64)))
    return groups


def sample_pairs(groups, rng):
    pos_parts = []
    neg_parts = []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=len(p), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True, bce_weight=0.15):
    base_enc, base_dim = encode(splits)
    enc, dim = append_time_features(splits, base_enc, base_dim, data_dir)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups = build_user_groups(ytr, utr)
    if verbose:
        n_pairs = sum(len(p) for p, _ in groups)
        print(f"BPR eligible users={len(groups):,d}, sampled pairs/epoch={n_pairs:,d}, bce_weight={bce_weight}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        model.train()
        losses = []
        bprs = []
        bces = []
        for i in range(0, len(pos_idx), bs):
            psel = torch.from_numpy(pos_idx[i:i + bs])
            nsel = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            bpr = -torch.nn.functional.logsigmoid(sp - sn).mean()
            bce = 0.5 * (torch.nn.functional.softplus(-sp).mean() +
                         torch.nn.functional.softplus(sn).mean())
            loss = bpr + bce_weight * bce
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bprs.append(bpr.item())
            bces.append(bce.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} bpr {np.mean(bprs):.4f} bce {np.mean(bces):.4f} | valid "
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
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--bce_weight', type=float, default=0.15)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+['hour','hour6','tab_hour6']")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device,
                     verbose=a.out is None, bce_weight=a.bce_weight)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== time_hour_bpr_bce_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
