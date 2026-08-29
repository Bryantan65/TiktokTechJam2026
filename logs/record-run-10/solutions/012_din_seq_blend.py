"""BPR + DIN-style sequence BCE rank blend with day/hour features.

Branches from 009.  The BPR member is unchanged.  The BCE member is replaced by
an FM with attention over each user's recent positive video/author history,
using only prior train interactions for valid/test histories.
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
SEQ_LEN = 20


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


def raw_log_paths(data_dir):
    return [
        os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv'),
    ]


def load_hour_queues(data_dir):
    q = defaultdict(deque)
    total = 0
    for path in raw_log_paths(data_dir):
        if not os.path.exists(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            cols = rdr.fieldnames or []
            hour_col = 'hourmin' if 'hourmin' in cols else ('hour' if 'hour' in cols else None)
            date_col = 'date' if 'date' in cols else ('time_ms' if 'time_ms' in cols else None)
            if hour_col is None or date_col is None:
                continue
            for r in rdr:
                key = (norm_int_str(r.get(date_col, '')),
                       norm_int_str(r.get('user_id', '')),
                       norm_int_str(r.get('video_id', '')),
                       norm_int_str(r.get('tab', '')))
                q[key].append(parse_hour(r.get(hour_col, 0)))
                total += 1
    print(f"loaded raw hourmin for {total:,d} rows across {len(q):,d} keys")
    return q


def make_hour_lookup(splits, data_dir):
    queues = load_hour_queues(data_dir)
    lookup = {}
    miss = 0
    used = 0
    for name, rows in splits.items():
        arr = np.zeros(len(rows), dtype=np.int16)
        for i, r in enumerate(rows):
            key = (norm_int_str(r[0]), norm_int_str(r[1]), norm_int_str(r[2]), norm_int_str(r[4]))
            dq = queues.get(key)
            if dq:
                arr[i] = dq.popleft()
                used += 1
            else:
                arr[i] = 0
                miss += 1
        lookup[name] = arr
    print(f"matched raw hourmin for {used:,d} tuple rows; missing {miss:,d}")
    return lookup


def row_features(row, hour):
    return (row[1], row[2], row[3], row[4], dur_bucket(row[5]),
            yyyymmdd_to_dow(row[0]), int(hour), int(hour) // 4)


def encode_ext(splits, hour_lookup):
    vocabs = [{} for _ in EXT_FIELDS]
    for name, rows in splits.items():
        hours = hour_lookup[name]
        for r, h in zip(rows, hours):
            vals = row_features(r, h)
            for j, v in enumerate(vals):
                if v not in vocabs[j]:
                    vocabs[j][v] = len(vocabs[j])
    offsets = np.cumsum([0] + [len(v) for v in vocabs[:-1]], dtype=np.int64)
    dim = int(offsets[-1] + len(vocabs[-1]))
    enc = {}
    for name, rows in splits.items():
        hours = hour_lookup[name]
        X = np.empty((len(rows), len(EXT_FIELDS)), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=object)
        for i, (r, h) in enumerate(zip(rows, hours)):
            vals = row_features(r, h)
            for j, v in enumerate(vals):
                X[i, j] = vocabs[j][v] + offsets[j]
            y[i] = float(r[6])
            users[i] = r[1]
        enc[name] = (X, y, users)
    return enc, dim


def build_histories(enc, seq_len=SEQ_LEN):
    """Recent positive video/author feature ids; valid/test see train history only."""
    hist = {}
    user_hist = defaultdict(lambda: deque(maxlen=seq_len))

    def fill_from_deques(users, X, y=None, update=False):
        n = len(users)
        hv = np.full((n, seq_len), -1, dtype=np.int64)
        ha = np.full((n, seq_len), -1, dtype=np.int64)
        for i, u in enumerate(users):
            h = user_hist[u]
            if h:
                vals = list(h)[-seq_len:]
                start = seq_len - len(vals)
                hv[i, start:] = [p[0] for p in vals]
                ha[i, start:] = [p[1] for p in vals]
            if update and y is not None and y[i] > 0.5:
                h.append((int(X[i, 1]), int(X[i, 2])))
        return hv, ha

    Xtr, ytr, utr = enc['train']
    hist['train'] = fill_from_deques(utr, Xtr, ytr, update=True)
    # Freeze the post-train history for all evaluation splits; do not consume valid/test labels.
    frozen = {u: deque(v, maxlen=seq_len) for u, v in user_hist.items()}
    for name, (X, y, users) in enc.items():
        if name == 'train':
            continue
        user_hist = {u: deque(v, maxlen=seq_len) for u, v in frozen.items()}
        hist[name] = fill_from_deques(users, X, None, update=False)
    return hist


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


class DINFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.att1 = torch.nn.Linear(4 * k, 32)
        self.att2 = torch.nn.Linear(32, 1)
        self.act = torch.nn.PReLU()
        self.seq_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.att1.weight)
        torch.nn.init.zeros_(self.att1.bias)
        torch.nn.init.xavier_uniform_(self.att2.weight)
        torch.nn.init.zeros_(self.att2.bias)

    def fm_part(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    def forward(self, X, Hvid, Hauth):
        base = self.fm_part(X)
        mask = Hvid.ge(0)
        hv = Hvid.clamp_min(0)
        ha = Hauth.clamp_min(0)
        target = self.V[X[:, 1]] + self.V[X[:, 2]]
        h = self.V[hv] + self.V[ha]
        t = target[:, None, :].expand_as(h)
        z = torch.cat([h, t, h - t, h * t], dim=2)
        att = self.att2(self.act(self.att1(z))).squeeze(-1)
        att = att.masked_fill(~mask, -1e9)
        w = torch.softmax(att, dim=1) * mask.float()
        ctx = (w[:, :, None] * h).sum(1)
        seq_logit = (ctx * target).sum(1)
        return base + self.seq_scale * seq_logit

    @torch.no_grad()
    def predict(self, X, Hvid, Hauth, bs=100_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            hv = torch.from_numpy(Hvid[i:i + bs].astype(np.int64)).to(device)
            ha = torch.from_numpy(Hauth[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb, hv, ha).cpu().numpy())
        return np.concatenate(out)


def train_din_bce(enc, hist, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                  patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Hvt, Hat = hist['train']
    Hvv, Hav = hist['valid']
    model = DINFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W, model.att1.weight, model.att1.bias,
                                        model.att2.weight, model.att2.bias, model.act.weight,
                                        model.seq_scale], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Hvt_t = torch.from_numpy(Hvt.astype(np.int64))
    Hat_t = torch.from_numpy(Hat.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        model.train(); losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device); yb = ytr_t[sel].to(device)
            hv = Hvt_t[sel].to(device); ha = Hat_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb, hv, ha), yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va_scores = model.predict(Xva, Hvv, Hav, device=device)
        va = evaluate(uva, yva, va_scores)
        if verbose:
            print(f"  DIN epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


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
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
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
    hour_lookup = make_hour_lookup(splits, data_dir)
    enc, dim = encode_ext(splits, hour_lookup)
    hist = build_histories(enc, seq_len=SEQ_LEN)
    bpr = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                    device=device, verbose=verbose)
    din = train_din_bce(enc, hist, dim, k=k, lr=lr, epochs=epochs, seed=seed + 1009,
                        device=device, verbose=verbose)
    return bpr, din, enc, hist


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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={EXT_FIELDS}, seq_len={SEQ_LEN}")

    bpr, din, enc, hist = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                              seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    Hvid, Hauth = hist[target]
    s_bpr = bpr.predict(X, device=a.device)
    s_din = din.predict(X, Hvid, Hauth, device=a.device)
    scores = weighted_within_user_rank(s_bpr, s_din, users, wa=0.60)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        r = evaluate(users, y, scores)
        print(f"\n=== din_seq_blend (seed={a.seed}, device={a.device}) ===")
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
