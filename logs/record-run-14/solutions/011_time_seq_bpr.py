import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402
from evaluate import evaluate  # noqa: E402


def find_log_files(data_dir):
    pats = [os.path.join(data_dir, 'log_standard*.csv'), os.path.join(data_dir, '*log_standard*.csv')]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    files = sorted(set(f for f in files if 'random' not in os.path.basename(f).lower()))
    def order(f):
        b = os.path.basename(f)
        if '4_08_to_4_21' in b:
            return (0, b)
        if '4_22_to_5_08' in b:
            return (1, b)
        return (2, b)
    return sorted(files, key=order)


def parse_hour(v):
    if v is None or v == '':
        return 24
    try:
        x = int(float(v))
        # KuaiRand hourmin is normally HHMM, but tolerate an hour already.
        h = x // 100 if x >= 100 else x
        return h if 0 <= h <= 23 else 24
    except Exception:
        return 24


def get_first(rec, names, default=''):
    for n in names:
        if n in rec and rec[n] not in (None, ''):
            return rec[n]
    return default


def date_to_dow(d):
    try:
        return datetime.strptime(str(int(float(d))), '%Y%m%d').weekday()
    except Exception:
        return 7


def read_time_order_rows(data_dir):
    """Read raw logs in the same order as data.load: train log then valid/test log.

    Node 10 attempted a key join and missed every row.  The contract says row order is
    preserved, and the two raw logs contain exactly train+valid+test rows, so sequence
    alignment is the robust way to attach non-label time columns.
    """
    rows = []
    per_date_count = defaultdict(int)
    for fn in find_log_files(data_dir):
        with open(fn, newline='') as f:
            rdr = csv.DictReader(f)
            for rec in rdr:
                date = get_first(rec, ['date', 'request_date', 'day'], '')
                hour = parse_hour(get_first(rec, ['hourmin', 'hour_min', 'time', 'timestamp'], ''))
                local = per_date_count[str(date)]
                per_date_count[str(date)] += 1
                rows.append((str(date), hour, local))
    totals = dict(per_date_count)
    out = []
    total = max(1, len(rows))
    for gi, (date, hour, local) in enumerate(rows):
        gbin20 = min(19, int(20 * gi / total))
        gbin50 = min(49, int(50 * gi / total))
        dbin10 = min(9, int(10 * local / max(1, totals.get(str(date), 1))))
        dbin20 = min(19, int(20 * local / max(1, totals.get(str(date), 1))))
        out.append((date, hour, gbin20, gbin50, dbin10, dbin20))
    return out


def build_augmented(splits, data_dir):
    enc0, dim0 = encode(splits)
    raw_aux = read_time_order_rows(data_dir)
    need = sum(len(splits[k]) for k in ['train', 'valid', 'test'] if k in splits)
    use_seq = (len(raw_aux) == need)
    aux = {}
    ptr = 0
    miss_hour = 0
    for sp in ['train', 'valid', 'test']:
        if sp not in splits:
            continue
        vals = []
        for r in splits[sp]:
            if use_seq:
                raw_date, hour, gbin20, gbin50, dbin10, dbin20 = raw_aux[ptr]
                ptr += 1
                # Prefer tuple date for consistency; raw_date is only diagnostic.
            else:
                hour, gbin20, gbin50, dbin10, dbin20 = 24, 20, 50, 10, 20
            if hour == 24:
                miss_hour += 1
            date = str(r[0])
            tab = str(r[4])
            dow = date_to_dow(date)
            hb4 = hour // 4 if hour < 24 else 6
            hb6 = hour // 6 if hour < 24 else 4
            # All are categorical fields for FM interactions with user/video/author/tab.
            vals.append((
                'date=' + date,
                'dow=' + str(dow),
                'h=' + str(hour),
                'hb4=' + str(hb4),
                'hb6=' + str(hb6),
                'tab_h=' + tab + '_' + str(hour),
                'tab_hb4=' + tab + '_' + str(hb4),
                'tab_date=' + tab + '_' + date,
                'g20=' + str(gbin20),
                'g50=' + str(gbin50),
                'd10=' + date + '_' + str(dbin10),
                'd20=' + date + '_' + str(dbin20),
            ))
        aux[sp] = vals
    n_extra = len(next(iter(aux.values()))[0]) if aux and len(next(iter(aux.values()))) else 0
    maps = []
    off = dim0
    for j in range(n_extra):
        mp = {}
        for sp in aux:
            for v in aux[sp]:
                if v[j] not in mp:
                    mp[v[j]] = off + len(mp)
        maps.append(mp)
        off += len(mp)
    enc = {}
    for sp in splits:
        X0, y, u = enc0[sp]
        if sp in aux:
            H = np.empty((len(X0), n_extra), dtype=np.int64)
            for i, v in enumerate(aux[sp]):
                for j in range(n_extra):
                    H[i, j] = maps[j][v[j]]
            X = np.concatenate([X0.astype(np.int64), H], axis=1)
        else:
            X = X0.astype(np.int64)
        enc[sp] = (X, y, u)
    print(f'time_seq raw_rows={len(raw_aux)} need={need} use_seq={use_seq} missing_hour={miss_hour} extra_fields={n_extra} dim={off}', flush=True)
    return enc, off


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0.0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]
        s = E.sum(1)
        inter = 0.5 * ((s * s).sum(1) - (E * E).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)
            out.append(self(xb).detach().cpu().numpy())
        return np.concatenate(out)


def build_pairs(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0.5:
            pos[u].append(i)
        else:
            neg[u].append(i)
    pidx, pools = [], []
    for u, ps in pos.items():
        ns = neg.get(u)
        if not ns:
            continue
        arr = np.asarray(ns, dtype=np.int64)
        for p in ps:
            pidx.append(p)
            pools.append(arr)
    return np.asarray(pidx, dtype=np.int64), pools


def train_bpr(enc, dim, seed, k=16, lr=0.001, l2=3e-6, epochs=24, bs=8192, patience=4, device='cpu'):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([
        {'params': [model.V, model.W], 'weight_decay': l2},
        {'params': [model.b], 'weight_decay': 0.0},
    ], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    pos_idx, neg_pools = build_pairs(ytr, utr)
    rng = np.random.default_rng(seed + 17)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    n_steps = max(len(ytr), len(pos_idx))
    best = -1.0
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.integers(0, len(pos_idx), size=n_steps, dtype=np.int64)
        for st in range(0, len(order), bs):
            which = order[st:st+bs]
            pp = pos_idx[which]
            nn = np.empty(len(which), dtype=np.int64)
            for j, w in enumerate(which):
                pool = neg_pools[int(w)]
                nn[j] = pool[rng.integers(0, len(pool))]
            both = np.concatenate([pp, nn])
            xb = Xtr_t[torch.from_numpy(both)].to(device)
            logits = model(xb)
            lp, ln = logits[:len(pp)], logits[len(pp):]
            loss = torch.nn.functional.softplus(-(lp - ln)).mean()
            yb = ytr_t[torch.from_numpy(both)].to(device)
            loss = loss + 0.02 * bce_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        pred = model.predict(Xva, device=device)
        m = evaluate(uva, yva, pred)['primary']
        print(f'bpr epoch={ep} valid_primary={m:.6f}', flush=True)
        if m > best + 1e-5:
            best = m
            bad = 0
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    return model


def z_by_user(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(scores)
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    for idx in groups.values():
        v = scores[idx]
        sd = v.std()
        out[idx] = (v - v.mean()) / sd if sd > 1e-8 else (v - v.mean())
    return out


def cached_bpr(enc, dim, target, seed, split_name, device):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'011_time_seq_bpr_{split_name}_seed{seed}.npy')
    if os.path.isfile(path):
        return np.load(path)
    model = train_bpr(enc, dim, seed=seed, device=device)
    X, _, _ = enc[target]
    p = model.predict(X, device=device).astype(np.float64)
    np.save(path, p)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
        split_name = 'dev'
    else:
        splits = load(a.data_dir)
        target = a.split
        split_name = a.split
    print({k: len(v) for k, v in splits.items()}, 'base_fields=', FIELDS, flush=True)
    t0 = time.time()
    enc, dim = build_augmented(splits, a.data_dir)
    X, _, users = enc[target]
    pred = cached_bpr(enc, dim, target, a.seed, split_name, a.device)
    # Ranking metrics are invariant to monotone transforms within user; z-scoring
    # removes user-scale drift from the extra time fields before output.
    scores = z_by_user(pred, users)
    print(f'done in {time.time()-t0:.1f}s', flush=True)
    np.save(a.out, scores.astype(np.float64))
