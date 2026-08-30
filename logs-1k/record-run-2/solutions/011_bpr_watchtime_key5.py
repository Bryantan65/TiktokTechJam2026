"""FM BPR with watch-time confidence weights, raw-log alignment by log keys.

Debug of 009/010: 010 aligned zero rows because duration_ms in the tuple may come
from video features or be represented differently from the raw log.  Align the
ordered train split to raw log_standard rows using the stable impression keys
(date, user_id, video_id, author_id, tab) only, then use log(play_time_ms) as a
mild positive-pair weight.
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
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)


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


def norm(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def row_key(r):
    return (norm(r[DATE]), norm(r[USER]), norm(r[VIDEO]), norm(r[AUTHOR]), norm(r[TAB]))


def raw_key(rec):
    return (norm(rec.get('date', '')), norm(rec.get('user_id', '')),
            norm(rec.get('video_id', '')), norm(rec.get('author_id', '')),
            norm(rec.get('tab', '')))


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


def get_playtime(rec):
    for c in ('play_time_ms', 'play_ms', 'watch_time_ms', 'play_time'):
        if c in rec and rec[c] != '':
            try:
                return float(rec[c])
            except Exception:
                return 0.0
    return 0.0


def read_train_playtime_ordered(data_dir, train_rows):
    files = find_log_files(data_dir)
    pt = np.zeros(len(train_rows), dtype=np.float32)
    if not files or not train_rows:
        print('warning: raw logs not found or empty train; using unit watch-time weights', flush=True)
        return pt
    i = 0
    cur = row_key(train_rows[0])
    seen = 0
    first_raw = None
    t0 = time.time()
    for path in files:
        if i >= len(train_rows):
            break
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            print(f'raw columns from {os.path.basename(path)}: {reader.fieldnames[:12]}', flush=True)
            for rec in reader:
                seen += 1
                if first_raw is None:
                    first_raw = raw_key(rec)
                if raw_key(rec) == cur:
                    pt[i] = get_playtime(rec)
                    i += 1
                    if i >= len(train_rows):
                        break
                    cur = row_key(train_rows[i])
    print(f'first train key={row_key(train_rows[0])} first raw key={first_raw}', flush=True)
    print(f'aligned play_time_ms for {i:,d}/{len(train_rows):,d} train rows after scanning {seen:,d} raw rows in {time.time()-t0:.1f}s', flush=True)
    return pt


def make_pair_sampler(y, users, pos_weight):
    users = np.asarray(users)
    y = np.asarray(y)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    pos_rows, neg_by_user = [], {}
    start, n = 0, len(users)
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
    return pos_rows, users[pos_rows], pos_weight[pos_rows].astype(np.float32), neg_by_user


def sample_pairs(pos_rows, pos_users, pos_w, neg_by_user, rng, n_neg=2):
    perm = rng.permutation(len(pos_rows))
    p = pos_rows[perm]
    pu = pos_users[perm]
    w = pos_w[perm]
    negs = np.empty((len(p), n_neg), dtype=np.int64)
    for i, u in enumerate(pu):
        pool = neg_by_user[u]
        negs[i] = pool[rng.integers(len(pool), size=n_neg)]
    return p, negs, w


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pt = read_train_playtime_ordered(data_dir, splits['train'])
    pos_mask = ytr > 0.5
    if np.any(pt[pos_mask] > 0):
        # Mild confidence range (roughly 1.0 to 1.5 before clipping) so the
        # label definition still dominates and watch time only ranks positives.
        w = 1.0 + 0.5 * np.log1p(np.maximum(pt, 0.0)) / np.log(60001.0)
        cap = np.percentile(w[pos_mask], 99.5)
        w = np.minimum(w, cap).astype(np.float32)
    else:
        w = np.ones(len(ytr), dtype=np.float32)
    print(f'positive pair weight: mean={w[pos_mask].mean():.3f} max={w[pos_mask].max():.3f}', flush=True)

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_rows, pos_users, pos_w, neg_by_user = make_pair_sampler(ytr, utr, w)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_neg = 2

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pidx, nidx, pw = sample_pairs(pos_rows, pos_users, pos_w, neg_by_user, rng, n_neg=n_neg)
        model.train()
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs]).long()
            ns = torch.from_numpy(nidx[i:i + bs].reshape(-1)).long()
            wb = torch.from_numpy(pw[i:i + bs]).to(device).repeat_interleave(n_neg)
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            raw = -torch.nn.functional.logsigmoid(model(xp).repeat_interleave(n_neg) - model(xn))
            loss = (raw * wb).sum() / (wb.sum() + 1e-8)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr_wt {np.mean(losses):.4f} | valid "
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
