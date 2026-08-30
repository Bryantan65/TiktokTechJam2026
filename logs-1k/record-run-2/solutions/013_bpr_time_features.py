"""FM BPR with raw-log time context features.

Draft direction 6: the tuple loader strips hourmin/time context.  Align raw
log_standard rows to each split by ordered (date,user_id,video_id,tab), add
categorical date/hour/10-minute/tab-hour fields to the FM, and keep the proven
two-negative within-user BPR objective from node 2.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, FIELDS          # noqa: E402
from evaluate import evaluate          # noqa: E402

DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)


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


def norm(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def row_key(r):
    return (norm(r[DATE]), norm(r[USER]), norm(r[VIDEO]), norm(r[TAB]))


def raw_key(rec):
    return (norm(rec.get('date', '')), norm(rec.get('user_id', '')),
            norm(rec.get('video_id', '')), norm(rec.get('tab', '')))


def find_log_files(data_dir):
    names = ['log_standard_4_08_to_4_21_1k.csv', 'log_standard_4_22_to_5_08_1k.csv']
    files = [os.path.join(data_dir, n) for n in names]
    if all(os.path.isfile(p) for p in files):
        return files
    out = []
    for pat in ['log_standard_4_08_to_4_21*.csv', 'log_standard_4_22_to_5_08*.csv']:
        g = sorted(glob.glob(os.path.join(data_dir, pat)))
        if g:
            out.append(g[0])
    return out


def parse_hourmin(x):
    try:
        hm = int(float(x))
    except Exception:
        return -1, -1, -1
    h = hm // 100
    m = hm % 100
    if h < 0 or h > 23 or m < 0 or m > 59:
        return -1, -1, -1
    ten = h * 6 + (m // 10)
    block = h // 4
    return h, ten, block


def read_time_ordered(data_dir, rows, name='split'):
    n = len(rows)
    hour = np.full(n, -1, dtype=np.int16)
    ten = np.full(n, -1, dtype=np.int16)
    block = np.full(n, -1, dtype=np.int16)
    files = find_log_files(data_dir)
    if not files or not rows:
        print(f'warning: raw logs not found for {name}; time features missing', flush=True)
        return hour, ten, block
    i = 0
    cur = row_key(rows[0])
    seen = 0
    t0 = time.time()
    for path in files:
        if i >= n:
            break
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for rec in reader:
                seen += 1
                if raw_key(rec) == cur:
                    h, te, bl = parse_hourmin(rec.get('hourmin', ''))
                    hour[i], ten[i], block[i] = h, te, bl
                    i += 1
                    if i >= n:
                        break
                    cur = row_key(rows[i])
    miss = n - i
    print(f'aligned hourmin for {name}: {i:,d}/{n:,d} rows after scanning {seen:,d} raw rows in {time.time()-t0:.1f}s', flush=True)
    if miss:
        print(f'warning: {miss:,d} {name} rows missing time features', flush=True)
    return hour, ten, block


def dur_bucket(x):
    return int(np.log1p(float(x)) // 1)


def make_encoded(splits, aux):
    # Fields: user, video, author, tab, duration bucket, date, hour, 10-min,
    # 4-hour block, tab-hour, tab-10min.  Maps are built over all loaded splits,
    # matching starter-kit behavior and avoiding unknown categories at test time.
    maps = [{} for _ in range(11)]
    feats_by_split = {}
    ys_by_split = {}
    users_by_split = {}
    for sp, rows in splits.items():
        h, ten, block = aux[sp]
        feats = []
        ys = np.empty(len(rows), dtype=np.float32)
        us_raw = []
        for i, r in enumerate(rows):
            t = r[TAB]
            vals = [
                r[USER], r[VIDEO], r[AUTHOR], t, dur_bucket(r[DUR]), r[DATE],
                int(h[i]), int(ten[i]), int(block[i]), (t, int(h[i])), (t, int(ten[i])),
            ]
            feats.append(vals)
            ys[i] = float(r[LABEL])
            us_raw.append(r[USER])
            for j, v in enumerate(vals):
                d = maps[j]
                if v not in d:
                    d[v] = len(d)
        feats_by_split[sp] = feats
        ys_by_split[sp] = ys
        users_by_split[sp] = us_raw

    offsets = np.cumsum([0] + [len(m) for m in maps[:-1]]).astype(np.int64)
    dim = int(sum(len(m) for m in maps))
    user_map = {}
    enc = {}
    for sp, feats in feats_by_split.items():
        X = np.empty((len(feats), len(maps)), dtype=np.int64)
        for i, vals in enumerate(feats):
            for j, v in enumerate(vals):
                X[i, j] = maps[j][v] + offsets[j]
        u = np.empty(len(feats), dtype=np.int64)
        for i, raw_u in enumerate(users_by_split[sp]):
            if raw_u not in user_map:
                user_map[raw_u] = len(user_map)
            u[i] = user_map[raw_u]
        enc[sp] = (X, ys_by_split[sp], u)
    return enc, dim


def make_pair_sampler(y, users):
    users = np.asarray(users)
    y = np.asarray(y)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    pos_rows = []
    neg_by_user = {}
    start = 0
    n = len(users)
    while start < n:
        end = start + 1
        while end < n and su[end] == su[start]:
            end += 1
        rows = order[start:end]
        pos = rows[y[rows] > 0.5]
        neg = rows[y[rows] <= 0.5]
        if len(pos) and len(neg):
            pos_rows.append(pos)
            neg_by_user[su[start]] = neg.astype(np.int64)
        start = end
    if not pos_rows:
        raise RuntimeError('No users with both positive and negative examples for BPR')
    pos_rows = np.concatenate(pos_rows).astype(np.int64)
    pos_users = users[pos_rows]
    return pos_rows, pos_users, neg_by_user


def sample_pairs(pos_rows, pos_users, neg_by_user, rng, n_neg=2):
    perm = rng.permutation(len(pos_rows))
    p = pos_rows[perm]
    pu = pos_users[perm]
    negs = np.empty((len(p), n_neg), dtype=np.int64)
    for i, u in enumerate(pu):
        pool = neg_by_user[u]
        negs[i] = pool[rng.integers(len(pool), size=n_neg)]
    return p, negs


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    aux = {sp: read_time_ordered(data_dir, rows, sp) for sp, rows in splits.items()}
    enc, dim = make_encoded(splits, aux)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_rows, pos_users, neg_by_user = make_pair_sampler(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_neg = 2

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pidx, nidx = sample_pairs(pos_rows, pos_users, neg_by_user, rng, n_neg=n_neg)
        model.train()
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs]).long()
            ns = torch.from_numpy(nidx[i:i + bs].reshape(-1)).long()
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).repeat_interleave(n_neg)
            sn = model(xn)
            loss = -torch.nn.functional.logsigmoid(sp - sn).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr_time {np.mean(losses):.4f} | valid "
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
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...", flush=True)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}", flush=True)

    model, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(scores[:10])
