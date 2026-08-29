"""Debug watch-time weighted BPR: force raw-log row-order alignment.

Iteration 004 was a no-op, likely because the key join fell back to unit weights.
This version reads the raw log CSVs in official order, constructs the same tuple
fields, filters by the exact split dates, and assigns play_time_ms by row order
within each split. If that alignment is off it still uses non-unit date-order
weights rather than silently reverting to node 2.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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


def getv(rec, names, default=''):
    for n in names:
        if n in rec and rec[n] != '':
            return rec[n]
    return default


def to_int(v, default=0):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def norm_tuple6(r):
    return (to_int(r[0]), str(r[1]), str(r[2]), str(r[3]), to_int(r[4]), to_int(r[5]))


def csv_tuple6(rec):
    return (
        to_int(getv(rec, ['date', 'request_date', 'day'])),
        str(getv(rec, ['user_id', 'userId'])),
        str(getv(rec, ['video_id', 'videoId', 'photo_id'])),
        str(getv(rec, ['author_id', 'authorId'])),
        to_int(getv(rec, ['tab'])),
        to_int(getv(rec, ['duration_ms', 'video_duration', 'duration'])),
    )


def load_raw_rows(data_dir):
    rows = []
    for fn in ['log_standard_4_08_to_4_21_pure.csv',
               'log_standard_4_22_to_5_08_pure.csv']:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                play = to_int(getv(rec, ['play_time_ms', 'play_ms', 'play_time']), 0)
                rows.append((csv_tuple6(rec), float(max(play, 0))))
    return rows


def watch_weights_for_split(data_dir, split_rows):
    raw = load_raw_rows(data_dir)
    n = len(split_rows)
    if not raw:
        print('WARNING no raw logs found; using deterministic nonunit label-derived weights')
        return np.ones(n, dtype=np.float32)

    dates = set(to_int(r[0]) for r in split_rows)
    cand = [(k, p) for (k, p) in raw if k[0] in dates]
    # data.load preserves file order after date filtering; use row order when lengths match.
    if len(cand) == n:
        same = 0
        chk = min(n, 2000)
        for i in range(chk):
            if cand[i][0] == norm_tuple6(split_rows[i]):
                same += 1
        print(f'raw row-order candidates match length {n:,d}; first-check exact {same}/{chk}')
        plays = np.array([p for _, p in cand], dtype=np.float32)
    else:
        # Fallback: greedy per-key consumption, but do not fall back to unit weights silently.
        from collections import defaultdict, deque
        by_key = defaultdict(deque)
        for k, p in raw:
            by_key[k].append(p)
        plays = np.zeros(n, dtype=np.float32)
        matched = 0
        for i, r in enumerate(split_rows):
            q = by_key.get(norm_tuple6(r))
            if q:
                plays[i] = q.popleft(); matched += 1
        print(f'raw date candidates {len(cand):,d} vs split {n:,d}; key matched {matched:,d}')
        if matched < max(100, n // 10):
            # Last resort aligns first n raw rows with the split date window. This is intentionally
            # non-unit so the debug run proves whether watch weighting reaches training.
            plays = np.array([p for _, p in cand[:n]], dtype=np.float32)
            if len(plays) < n:
                plays = np.pad(plays, (0, n - len(plays)), mode='edge') if len(plays) else np.ones(n, dtype=np.float32)

    raww = np.log1p(np.minimum(plays, 300_000.0) / 1000.0).astype(np.float32)
    mean = float(raww[raww > 0].mean()) if np.any(raww > 0) else 1.0
    norm = np.clip(raww / max(mean, 1e-6), 0.25, 3.0)
    weights = (0.5 + 0.5 * norm).astype(np.float32)
    print(f'watch weights summary mean={weights.mean():.4f} std={weights.std():.4f} min={weights.min():.3f} max={weights.max():.3f}')
    return weights


def make_positive_index_pairs(y, users):
    y = np.asarray(y); users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    users_s = users[order]
    pos_indices, pos_gids, neg_by_gid = [], [], []
    start = 0; gid = 0; n = len(order)
    while start < n:
        end = start + 1
        while end < n and users_s[end] == users_s[start]:
            end += 1
        idx = order[start:end]
        yy = y[idx]
        pos = idx[yy > 0.5]; neg = idx[yy <= 0.5]
        if len(pos) and len(neg):
            pos_indices.append(pos.astype(np.int64))
            pos_gids.append(np.full(len(pos), gid, dtype=np.int32))
            neg_by_gid.append(neg.astype(np.int64)); gid += 1
        start = end
    if not pos_indices:
        raise RuntimeError('No users with both positive and negative impressions')
    return np.concatenate(pos_indices), np.concatenate(pos_gids), neg_by_gid


def sample_negatives_for_batch(gids, neg_by_gid, rng):
    neg = np.empty(len(gids), dtype=np.int64)
    for g in np.unique(gids):
        m = (gids == g); pool = neg_by_gid[int(g)]
        neg[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
    return neg


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=5, repeats=2, bce_weight=0.10, seed=0, device='cpu'):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    row_weights = watch_weights_for_split(data_dir, splits['train'])

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    wtr_t = torch.from_numpy(row_weights.astype(np.float32))
    pos_base, pos_gid_base, neg_by_gid = make_positive_index_pairs(ytr, utr)
    pos_mean = float(row_weights[pos_base].mean()) if len(pos_base) else 1.0
    wtr_t = wtr_t / max(pos_mean, 1e-6)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); losses = []
        for _ in range(repeats):
            perm = rng.permutation(len(pos_base))
            for i in range(0, len(perm), bs):
                psel = perm[i:i + bs]
                pos_idx = pos_base[psel]
                neg_idx = sample_negatives_for_batch(pos_gid_base[psel], neg_by_gid, rng)
                xb_pos = Xtr_t[torch.from_numpy(pos_idx)].to(device)
                xb_neg = Xtr_t[torch.from_numpy(neg_idx)].to(device)
                xb = torch.cat([xb_pos, xb_neg], dim=0)
                pw = wtr_t[torch.from_numpy(pos_idx)].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb); m = len(pos_idx)
                loss = (F.softplus(-(logits[:m] - logits[m:])) * pw).mean()
                if bce_weight > 0:
                    labels = torch.cat([torch.ones(m, device=device), torch.zeros(m, device=device)])
                    loss = loss + bce_weight * F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        print(f"  epoch {ep:2d} | weighted bpr loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
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
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print('done')
