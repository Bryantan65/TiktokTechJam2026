"""FM trained with within-user BPR, weighting positive pairs by watch time.

Improve node 2 / direction 4: BPR treats every positive equally, but long-view
positives with more actual watch time should be stronger preferences.  Read the
raw standard logs in the same order as the starter-kit loader, align by tuple
keys, and use log-scaled play_time_ms as a per-positive pair weight.
"""
import argparse
import csv
import glob
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


def _as_int_str(x):
    if x is None:
        return ''
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def _label_from_raw(rec):
    for name in ('label', 'long_view', 'is_long_view'):
        if name in rec and rec[name] != '':
            try:
                return int(float(rec[name]))
            except Exception:
                pass
    # Fallback used only if a variant lacks the explicit long-view column.
    return 0


def _find_log_files(data_dir):
    f1 = os.path.join(data_dir, 'log_standard_4_08_to_4_21_1k.csv')
    f2 = os.path.join(data_dir, 'log_standard_4_22_to_5_08_1k.csv')
    if os.path.isfile(f1) and os.path.isfile(f2):
        return [f1, f2]
    files = []
    for pat in ['log_standard_4_08_to_4_21*.csv', 'log_standard_4_22_to_5_08*.csv']:
        g = sorted(glob.glob(os.path.join(data_dir, pat)))
        if g:
            files.append(g[0])
    return files


def read_playtime_lookup(data_dir):
    """Map each tuple key occurrence to play_time_ms, preserving duplicates."""
    lookup = defaultdict(deque)
    files = _find_log_files(data_dir)
    if len(files) < 2:
        print('warning: raw log files not found; watch-time weights disabled')
        return lookup
    n = 0
    for path in files:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for rec in reader:
                key = (
                    _as_int_str(rec.get('date')),
                    _as_int_str(rec.get('user_id')),
                    _as_int_str(rec.get('video_id')),
                    _as_int_str(rec.get('author_id')),
                    _as_int_str(rec.get('tab')),
                    _as_int_str(rec.get('duration_ms')),
                    str(_label_from_raw(rec)),
                )
                pt = 0.0
                for col in ('play_time_ms', 'play_ms', 'watch_time_ms'):
                    if col in rec and rec[col] != '':
                        try:
                            pt = float(rec[col])
                        except Exception:
                            pt = 0.0
                        break
                lookup[key].append(pt)
                n += 1
    print(f'read play_time_ms for {n:,d} raw rows from {len(files)} files')
    return lookup


def align_playtime(rows, lookup):
    out = np.zeros(len(rows), dtype=np.float32)
    missed = 0
    for i, r in enumerate(rows):
        key = tuple(str(x) for x in r[:7])
        q = lookup.get(key)
        if q:
            out[i] = float(q.popleft())
        else:
            missed += 1
    if missed:
        print(f'warning: missed play_time alignment for {missed:,d}/{len(rows):,d} rows')
    return out


def make_pair_sampler(y, users, pos_weight):
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
    pos_w = pos_weight[pos_rows].astype(np.float32)
    return pos_rows, pos_users, pos_w, neg_by_user


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

    play_lookup = read_playtime_lookup(data_dir)
    pt = align_playtime(splits['train'], play_lookup)
    # Robust log weight.  If alignment failed, all positives get weight 1.
    pos_mask = ytr > 0.5
    if np.any(pt[pos_mask] > 0):
        w = 1.0 + np.log1p(np.maximum(pt, 0.0)) / np.log(60001.0)
        cap = np.percentile(w[pos_mask], 99.5)
        w = np.minimum(w, cap).astype(np.float32)
    else:
        w = np.ones(len(ytr), dtype=np.float32)
    print(f'positive pair weight: mean={w[pos_mask].mean():.3f} max={w[pos_mask].max():.3f}')

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
            ns_np = nidx[i:i + bs].reshape(-1)
            ns = torch.from_numpy(ns_np).long()
            wb = torch.from_numpy(pw[i:i + bs]).to(device).repeat_interleave(n_neg)
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).repeat_interleave(n_neg)
            sn = model(xn)
            raw = -torch.nn.functional.logsigmoid(sp - sn)
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
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_watchtime_weight (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
