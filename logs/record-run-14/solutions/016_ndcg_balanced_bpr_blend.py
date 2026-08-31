import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402
from evaluate import evaluate  # noqa: E402


def _s(x):
    if x is None:
        return ''
    if isinstance(x, bytes):
        x = x.decode('utf8')
    try:
        if isinstance(x, (float, np.floating)) and np.isfinite(x) and abs(x - int(x)) < 1e-6:
            return str(int(x))
    except Exception:
        pass
    return str(x)


def row_key(r):
    return (_s(r[0]), _s(r[1]), _s(r[2]), _s(r[3]), _s(r[4]), _s(r[5]))


def csv_key(rec):
    def g(*names):
        for n in names:
            if n in rec:
                return rec.get(n)
        return ''
    return (_s(g('date')), _s(g('user_id', 'user')), _s(g('video_id', 'photo_id', 'item_id')),
            _s(g('author_id')), _s(g('tab')), _s(g('duration_ms', 'duration')))


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
        h = x // 100 if x >= 100 else x
        return h if 0 <= h <= 23 else 24
    except Exception:
        return 24


def date_to_dow(d):
    try:
        return datetime.strptime(_s(d), '%Y%m%d').weekday()
    except Exception:
        return 7


def read_time_aux(data_dir):
    q = defaultdict(deque)
    files = find_log_files(data_dir)
    global_idx = 0
    per_date_count = defaultdict(int)
    raw = []
    for fn in files:
        try:
            with open(fn, newline='') as f:
                rdr = csv.DictReader(f)
                for rec in rdr:
                    k = csv_key(rec)
                    hm = rec.get('hourmin', rec.get('time_ms', rec.get('timestamp', '')))
                    hour = parse_hour(hm)
                    d = k[0]
                    local = per_date_count[d]
                    per_date_count[d] += 1
                    raw.append((k, hour, global_idx, local))
                    global_idx += 1
        except FileNotFoundError:
            continue
    totals_by_date = dict(per_date_count)
    total = max(1, global_idx)
    for k, hour, gi, local in raw:
        d = k[0]
        gbin = min(19, int(20 * gi / total))
        dbin = min(9, int(10 * local / max(1, totals_by_date.get(d, 1))))
        q[k].append((hour, gbin, dbin))
    return q, len(raw)


def build_augmented(splits, data_dir):
    enc0, dim0 = encode(splits)
    auxq, nraw = read_time_aux(data_dir)
    aux = {}
    miss = 0
    for sp, rows in splits.items():
        vals = []
        for r in rows:
            k = row_key(r)
            if auxq.get(k):
                hour, gbin, dbin = auxq[k].popleft()
            else:
                hour, gbin, dbin = 24, 20, 10
                miss += 1
            tab = _s(r[4])
            date = _s(r[0])
            dow = date_to_dow(date)
            hb = hour // 4 if hour < 24 else 6
            vals.append((
                'd=' + date,
                'dow=' + str(dow),
                'h=' + str(hour),
                'hb=' + str(hb),
                'th=' + tab + '_' + str(hour),
                'td=' + tab + '_' + date,
                'gb=' + str(gbin),
                'db=' + date + '_' + str(dbin),
            ))
        aux[sp] = vals
    maps = []
    n_extra = len(next(iter(aux.values()))[0]) if aux and len(next(iter(aux.values()))) else 0
    off = dim0
    for j in range(n_extra):
        cats = {}
        for sp in aux:
            for v in aux[sp]:
                if v[j] not in cats:
                    cats[v[j]] = off + len(cats)
        maps.append(cats)
        off += len(cats)
    enc = {}
    for sp in splits:
        X0, y, u = enc0[sp]
        H = np.empty((len(X0), n_extra), dtype=np.int64)
        for i, v in enumerate(aux[sp]):
            for j in range(n_extra):
                H[i, j] = maps[j][v[j]]
        enc[sp] = (np.concatenate([X0.astype(np.int64), H], axis=1), y, u)
    print(f'time_aux raw_rows={nraw} missing_matches={miss} extra_fields={n_extra} dim={off}')
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


def build_pairs_and_weights(y, users, balanced=False):
    pos = defaultdict(list); neg = defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        (pos if yy > 0.5 else neg)[u].append(i)
    pidx, pools, weights = [], [], []
    for u, ps in pos.items():
        ns = neg.get(u)
        if not ns:
            continue
        arr = np.asarray(ns, dtype=np.int64)
        # GAUC weights users by positive count; nDCG@5 is per-user.  This mild cap makes a
        # second member care less about users with many positives, without discarding GAUC signal.
        w = 1.0 / float(min(len(ps), 5)) if balanced else 1.0
        for p in ps:
            pidx.append(p); pools.append(arr); weights.append(w)
    weights = np.asarray(weights, dtype=np.float32)
    if len(weights) and balanced:
        weights /= max(1e-6, weights.mean())
        print(f'balanced_pair_weights n={len(weights)} min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}')
    return np.asarray(pidx, dtype=np.int64), pools, weights


def train_model(enc, dim, seed, balanced=False, k=16, lr=0.001, l2=3e-6, epochs=40, bs=8192, patience=4, device='cpu'):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([
        {'params': [model.V, model.W], 'weight_decay': l2},
        {'params': [model.b], 'weight_decay': 0.0},
    ], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed + 17)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    pos_idx, neg_pools, pair_w = build_pairs_and_weights(ytr, utr, balanced=balanced)
    pair_w_t = torch.from_numpy(pair_w.astype(np.float32))
    n_steps = max(len(ytr), len(pos_idx))
    best = -1.0; best_state = None; bad = 0
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
            if balanced:
                wb = pair_w_t[torch.from_numpy(which)].to(device)
                loss_pair = (torch.nn.functional.softplus(-(lp - ln)) * wb).sum() / torch.clamp(wb.sum(), min=1e-6)
            else:
                loss_pair = torch.nn.functional.softplus(-(lp - ln)).mean()
            yb = ytr_t[torch.from_numpy(both)].to(device)
            loss = loss_pair + 0.02 * bce_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
        pred = model.predict(Xva, device=device)
        m = evaluate(uva, yva, pred)['primary']
        print(f'bpr balanced={balanced} seed={seed} epoch={ep} valid_primary={m:.6f}', flush=True)
        if m > best + 1e-5:
            best = m; bad = 0
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


def member_pred(enc, dim, target, mseed, split_name, device, balanced=False):
    os.makedirs('pred_cache', exist_ok=True)
    prefix = '016_time_bpr_ndcgbal_v1' if balanced else '010_time_bpr_v1'
    path = os.path.join('pred_cache', f'{prefix}_{split_name}_seed{mseed}.npy')
    if os.path.isfile(path):
        print('load', path, flush=True)
        return np.load(path)
    model = train_model(enc, dim, seed=mseed, balanced=balanced, device=device)
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
        splits = load_dev(a.data_dir); target = 'valid'; split_name = 'dev'
    else:
        splits = load(a.data_dir); target = a.split; split_name = a.split
    print({k: len(v) for k, v in splits.items()}, 'base_fields=', FIELDS)
    t0 = time.time()
    enc, dim = build_augmented(splits, a.data_dir)
    _, _, users = enc[target]
    seeds = [a.seed, a.seed + 101, a.seed + 202, a.seed + 303, a.seed + 404]
    old_z, bal_z = [], []
    for ms in seeds:
        old_z.append(z_by_user(member_pred(enc, dim, target, ms, split_name, a.device, balanced=False), users))
        bal_z.append(z_by_user(member_pred(enc, dim, target, ms, split_name, a.device, balanced=True), users))
    old_ens = z_by_user(np.mean(old_z, axis=0), users)
    bal_ens = z_by_user(np.mean(bal_z, axis=0), users)
    # Keep the strong GAUC-oriented incumbent dominant, but let the balanced member move rankings.
    scores = 0.65 * old_ens + 0.35 * bal_ens
    print(f'blend old=0.65 ndcg_balanced=0.35 members_each={len(seeds)} done in {time.time()-t0:.1f}s')
    np.save(a.out, scores.astype(np.float64))
