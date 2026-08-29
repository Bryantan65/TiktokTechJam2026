"""BCE+BPR rank blend with time features and a multi-task BCE member.

Branches from 009.  The BPR member is unchanged.  The BCE member is replaced
by a shared-embedding FM trained on the target label plus auxiliary feedback
columns from the raw logs (click/like/follow/comment/forward and log playtime).
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, FIELDS                         # noqa: E402
from evaluate import evaluate                         # noqa: E402


EXT_FIELDS = FIELDS + ['dow', 'hour', 'tod4']
BIN_AUX_CANDIDATES = [
    ('is_click', ['is_click', 'click']),
    ('is_like', ['is_like', 'like']),
    ('is_follow', ['is_follow', 'follow']),
    ('is_comment', ['is_comment', 'comment']),
    ('is_forward', ['is_forward', 'forward']),
]
PLAY_CANDIDATES = ['play_time_ms', 'play_time', 'playing_time']


def yyyymmdd_to_dow(d):
    y = int(d) // 10000
    m = (int(d) // 100) % 100
    day = int(d) % 100
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    if m < 3:
        y -= 1
    return (y + y // 4 - y // 100 + y // 400 + t[m - 1] + day + 6) % 7


def dur_bucket(ms):
    try:
        ms = int(ms)
    except Exception:
        ms = 0
    if ms < 7_000:
        return 0
    if ms < 15_000:
        return 1
    if ms < 30_000:
        return 2
    if ms < 60_000:
        return 3
    return 4


def norm_int_str(x):
    s = str(x)
    try:
        return str(int(float(s)))
    except Exception:
        return s


def parse_hour(v):
    try:
        h = int(float(v))
    except Exception:
        return 0
    if h >= 100:
        h = h // 100
    return max(0, min(23, h))


def parse_float(v, default=0.0):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except Exception:
        return default


def raw_log_paths(data_dir):
    return [
        os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv'),
    ]


def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_raw_queues(data_dir):
    q = defaultdict(deque)
    total = 0
    chosen_bin = None
    chosen_play = None
    for path in raw_log_paths(data_dir):
        if not os.path.exists(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            cols = set(rdr.fieldnames or [])
            hour_col = 'hourmin' if 'hourmin' in cols else ('hour' if 'hour' in cols else None)
            date_col = 'date' if 'date' in cols else ('time_ms' if 'time_ms' in cols else None)
            bin_cols = [find_col(cols, cand) for _, cand in BIN_AUX_CANDIDATES]
            play_col = find_col(cols, PLAY_CANDIDATES)
            if chosen_bin is None:
                chosen_bin = bin_cols
                chosen_play = play_col
                print('aux columns:', [c for c in bin_cols if c is not None], 'play:', play_col)
            if hour_col is None or date_col is None:
                continue
            for r in rdr:
                key = (norm_int_str(r.get(date_col, '')),
                       norm_int_str(r.get('user_id', '')),
                       norm_int_str(r.get('video_id', '')),
                       norm_int_str(r.get('tab', '')))
                hour = parse_hour(r.get(hour_col, 0))
                aux = [1.0 if parse_float(r.get(c, 0.0)) > 0.0 else 0.0
                       for c in bin_cols if c is not None]
                play = parse_float(r.get(play_col, 0.0), 0.0) if play_col is not None else 0.0
                q[key].append((hour, aux, play))
                total += 1
    n_aux = len([c for c in (chosen_bin or []) if c is not None])
    print(f"loaded raw rows {total:,d} across {len(q):,d} keys; n_bin_aux={n_aux}")
    return q, n_aux


def make_raw_lookup(splits, data_dir):
    queues, n_aux = load_raw_queues(data_dir)
    lookup = {}
    miss = 0
    used = 0
    for name, rows in splits.items():
        hours = np.zeros(len(rows), dtype=np.int16)
        aux = np.zeros((len(rows), n_aux), dtype=np.float32)
        play = np.zeros(len(rows), dtype=np.float32)
        for i, r in enumerate(rows):
            key = (norm_int_str(r[0]), norm_int_str(r[1]), norm_int_str(r[2]), norm_int_str(r[4]))
            dq = queues.get(key)
            if dq:
                h, a, p = dq.popleft()
                hours[i] = h
                if n_aux:
                    aux[i, :] = np.asarray(a, dtype=np.float32)
                play[i] = p
                used += 1
            else:
                miss += 1
        lookup[name] = (hours, aux, play)
    print(f"matched raw rows for {used:,d} tuple rows; missing {miss:,d}")
    return lookup, n_aux


def row_features(row, hour):
    return (row[1], row[2], row[3], row[4], dur_bucket(row[5]),
            yyyymmdd_to_dow(row[0]), int(hour), int(hour) // 4)


def encode_ext(splits, raw_lookup):
    vocabs = [{} for _ in EXT_FIELDS]
    for name, rows in splits.items():
        hours = raw_lookup[name][0]
        for r, h in zip(rows, hours):
            vals = row_features(r, h)
            for j, v in enumerate(vals):
                if v not in vocabs[j]:
                    vocabs[j][v] = len(vocabs[j])
    offsets = np.cumsum([0] + [len(v) for v in vocabs[:-1]], dtype=np.int64)
    dim = int(offsets[-1] + len(vocabs[-1]))
    enc = {}
    for name, rows in splits.items():
        hours, aux, play = raw_lookup[name]
        X = np.empty((len(rows), len(EXT_FIELDS)), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=object)
        for i, (r, h) in enumerate(zip(rows, hours)):
            vals = row_features(r, h)
            for j, v in enumerate(vals):
                X[i, j] = vocabs[j][v] + offsets[j]
            y[i] = float(r[6])
            users[i] = r[1]
        enc[name] = (X, y, users, aux, play)
    return enc, dim


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


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_aux=0, use_play=True, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.n_aux = int(n_aux)
        self.use_play = bool(use_play)
        if self.n_aux:
            self.W_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dim, dtype=torch.float32))
            self.b_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dtype=torch.float32))
            self.scale_aux = torch.nn.Parameter(torch.ones(self.n_aux, dtype=torch.float32))
        else:
            self.register_parameter('W_aux', None)
            self.register_parameter('b_aux', None)
            self.register_parameter('scale_aux', None)
        if self.use_play:
            self.W_play = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.b_play = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.scale_play = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
        else:
            self.register_parameter('W_play', None)
            self.register_parameter('b_play', None)
            self.register_parameter('scale_play', None)

    def interaction(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def forward_main(self, X):
        inter = self.interaction(X)
        return self.b + self.W[X].sum(1) + inter

    def forward_all(self, X):
        inter = self.interaction(X)
        main = self.b + self.W[X].sum(1) + inter
        aux_logits = None
        play_pred = None
        if self.n_aux:
            aux_lin = self.W_aux[:, X].sum(2).transpose(0, 1) + self.b_aux
            aux_logits = aux_lin + inter[:, None] * self.scale_aux[None, :]
        if self.use_play:
            play_pred = self.b_play + self.W_play[X].sum(1) + inter * self.scale_play
        return main, aux_logits, play_pred

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self.forward_main(xb).cpu().numpy())
        return np.concatenate(out)


def build_user_pair_sources(y, users):
    buckets = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if u not in buckets:
            buckets[u] = [[], []]
        buckets[u][1 if yy > 0.5 else 0].append(i)
    pos_lists, neg_lists, weights = [], [], []
    for neg, pos in buckets.values():
        if pos and neg:
            pos_a = np.asarray(pos, dtype=np.int64); neg_a = np.asarray(neg, dtype=np.int64)
            pos_lists.append(pos_a); neg_lists.append(neg_a)
            weights.append(min(len(pos_a) * len(neg_a), 2000))
    weights = np.asarray(weights, dtype=np.float64); weights /= weights.sum()
    return pos_lists, neg_lists, weights


def sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng):
    uids = rng.choice(len(pos_lists), size=n_pairs, replace=True, p=weights)
    pos_idx = np.empty(n_pairs, dtype=np.int64); neg_idx = np.empty(n_pairs, dtype=np.int64)
    for u in np.unique(uids):
        m = np.nonzero(uids == u)[0]
        pos = pos_lists[u]; neg = neg_lists[u]
        pos_idx[m] = pos[rng.integers(0, len(pos), size=len(m))]
        neg_idx[m] = neg[rng.integers(0, len(neg), size=len(m))]
    return pos_idx, neg_idx


def train_bpr(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train'][:3]
    Xva, yva, uva = enc['valid'][:3]
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_lists, neg_lists, weights = build_user_pair_sources(ytr, utr)
    n_pairs = max(1, int((ytr > 0.5).sum()))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        pidx, nidx = sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng)
        order = rng.permutation(n_pairs)
        model.train(); losses = []
        for i in range(0, n_pairs, bs):
            sel = order[i:i + bs]
            xb_pos = Xtr_t[torch.from_numpy(pidx[sel])].to(device)
            xb_neg = Xtr_t[torch.from_numpy(nidx[sel])].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.softplus(-(model(xb_pos) - model(xb_neg))).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BPR epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def train_mtl_bce(enc, dim, n_aux, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                  patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _, aux_tr, play_tr = enc['train']
    Xva, yva, uva = enc['valid'][:3]
    play_log = np.log1p(np.maximum(play_tr, 0.0)).astype(np.float32)
    p_mean = float(play_log.mean())
    p_std = float(play_log.std() + 1e-6)
    play_z = (play_log - p_mean) / p_std
    use_play = bool(np.any(play_tr > 0))
    model = MultiTaskFM(dim, n_aux=n_aux, use_play=use_play, k=k, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    aux_t = torch.from_numpy(aux_tr.astype(np.float32)) if n_aux else None
    play_t = torch.from_numpy(play_z.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        model.train(); losses = []
        for i in range(0, len(idx), bs):
            sel_np = idx[i:i + bs]
            sel = torch.from_numpy(sel_np)
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            main, aux_logits, play_pred = model.forward_all(xb)
            loss = F.binary_cross_entropy_with_logits(main, yb)
            if n_aux:
                ab = aux_t[sel].to(device)
                loss = loss + 0.08 * F.binary_cross_entropy_with_logits(aux_logits, ab)
            if use_play:
                pb = play_t[sel].to(device)
                loss = loss + 0.02 * F.mse_loss(play_pred, pb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  MTL epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def weighted_within_user_rank(scores_a, scores_b, users, wa=0.60):
    users = np.asarray(users)
    out = np.zeros(len(users), dtype=np.float64)
    order_u = np.argsort(users, kind='mergesort')
    start = 0
    while start < len(users):
        end = start + 1
        u = users[order_u[start]]
        while end < len(users) and users[order_u[end]] == u:
            end += 1
        idx = order_u[start:end]
        n = len(idx)
        if n == 1:
            out[idx[0]] = 0.0
        else:
            ra = np.empty(n, dtype=np.float64); rb = np.empty(n, dtype=np.float64)
            ra[np.argsort(scores_a[idx], kind='mergesort')] = np.linspace(0.0, 1.0, n)
            rb[np.argsort(scores_b[idx], kind='mergesort')] = np.linspace(0.0, 1.0, n)
            out[idx] = wa * ra + (1.0 - wa) * rb
        start = end
    return out


def run(splits, data_dir, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True):
    raw_lookup, n_aux = make_raw_lookup(splits, data_dir)
    enc, dim = encode_ext(splits, raw_lookup)
    bpr = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                    device=device, verbose=verbose)
    mtl = train_mtl_bce(enc, dim, n_aux=n_aux, k=k, lr=lr, epochs=epochs, seed=seed + 1009,
                        device=device, verbose=verbose)
    return bpr, mtl, enc


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
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={EXT_FIELDS}")

    bpr, mtl, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                        seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target][:3]
    s_bpr = bpr.predict(X, device=a.device)
    s_mtl = mtl.predict(X, device=a.device)
    scores = weighted_within_user_rank(s_bpr, s_mtl, users, wa=0.60)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        r = evaluate(users, y, scores)
        print(f"\n=== mtl_bce_blend (seed={a.seed}, device={a.device}) ===")
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
