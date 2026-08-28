"""Node15 time+BPR FM plus leakage-safe item/author target-CTR bins.

Smoothed video and author CTR bins are computed from train labels. For train rows
we use leave-one-out statistics so the row's own label is not encoded; for
valid/test we use full train statistics. The features are compact categorical
priors intended to help rare/cold IDs beyond raw video_id/author_id embeddings.
"""
import argparse
import csv
from collections import defaultdict, deque
from datetime import datetime
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


def yyyymmdd_to_weekday(d):
    return datetime.strptime(str(int(d)), '%Y%m%d').weekday()


def parse_hour(hourmin):
    hm = int(float(hourmin))
    h = hm // 100
    if h < 0:
        h = 0
    if h > 23:
        h = h % 24
    return h


def _get(row, *names):
    for n in names:
        if n in row and row[n] != '':
            return row[n]
    raise KeyError('none of columns present: ' + ','.join(names))


def build_raw_time_lookup(data_dir):
    files = [
        'log_standard_4_08_to_4_21_pure.csv',
        'log_standard_4_22_to_5_08_pure.csv',
    ]
    lookup = defaultdict(deque)
    for fn in files:
        path = os.path.join(data_dir, fn)
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                d = int(float(_get(r, 'date')))
                key = (
                    d,
                    int(float(_get(r, 'user_id'))),
                    int(float(_get(r, 'video_id'))),
                    int(float(_get(r, 'tab'))),
                    int(float(_get(r, 'duration_ms'))),
                )
                hour = parse_hour(_get(r, 'hourmin'))
                lookup[key].append((yyyymmdd_to_weekday(d), hour))
    return lookup


def build_stats(rows, key_idx):
    cnt = defaultdict(int)
    pos = defaultdict(int)
    for r in rows:
        k = int(r[key_idx])
        y = 1 if float(r[6]) > 0.5 else 0
        cnt[k] += 1
        pos[k] += y
    return cnt, pos


def smoothed_ctr_value(k, cnt, pos, global_mean, alpha, subtract_y=None):
    c = cnt.get(k, 0)
    p = pos.get(k, 0)
    if subtract_y is not None and c > 0:
        c -= 1
        p -= int(subtract_y)
    return (p + alpha * global_mean) / (c + alpha)


def make_quantile_edges(values, n_bins=10):
    if len(values) == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(np.asarray(values, dtype=np.float64), qs)
    # np.searchsorted handles duplicate edges; keep them, which intentionally
    # collapses low-variance ranges into fewer effective buckets.
    return edges.astype(np.float64)


def ctr_bin(x, edges):
    return int(np.searchsorted(edges, float(x), side='right'))


def build_target_ctr_features(splits, n_bins=10, alpha_video=20.0, alpha_author=20.0):
    train = splits['train']
    ytr = np.asarray([1 if float(r[6]) > 0.5 else 0 for r in train], dtype=np.int8)
    global_mean = float(ytr.mean()) if len(ytr) else 0.5
    vcnt, vpos = build_stats(train, 2)
    acnt, apos = build_stats(train, 3)

    # Use leave-one-out values for train to choose balanced, non-leaky bins.
    v_train_vals = []
    a_train_vals = []
    for r in train:
        y = 1 if float(r[6]) > 0.5 else 0
        v_train_vals.append(smoothed_ctr_value(int(r[2]), vcnt, vpos, global_mean, alpha_video, y))
        a_train_vals.append(smoothed_ctr_value(int(r[3]), acnt, apos, global_mean, alpha_author, y))
    v_edges = make_quantile_edges(v_train_vals, n_bins)
    a_edges = make_quantile_edges(a_train_vals, n_bins)

    feats = {}
    for sp, rows in splits.items():
        arr = np.empty((len(rows), 2), dtype=np.int64)
        is_train = (sp == 'train')
        for i, r in enumerate(rows):
            y = 1 if float(r[6]) > 0.5 else 0
            sub = y if is_train else None
            vv = smoothed_ctr_value(int(r[2]), vcnt, vpos, global_mean, alpha_video, sub)
            av = smoothed_ctr_value(int(r[3]), acnt, apos, global_mean, alpha_author, sub)
            arr[i, 0] = ctr_bin(vv, v_edges)
            arr[i, 1] = ctr_bin(av, a_edges)
        feats[sp] = arr
    return feats


def encode_with_time_ctr(splits, data_dir):
    enc, dim = encode(splits)
    raw_time = build_raw_time_lookup(data_dir)
    ctr_feats = build_target_ctr_features(splits, n_bins=10, alpha_video=20.0, alpha_author=20.0)
    out = {}
    missing = 0
    off_dow = dim
    off_hour = off_dow + 7
    off_tab_hour = off_hour + 24
    off_vctr = off_tab_hour + 5 * 24
    off_actr = off_vctr + 10
    final_dim = off_actr + 10
    ctr_offsets = np.asarray([off_vctr, off_actr], dtype=np.int64)

    for sp, rows in splits.items():
        X, y, u = enc[sp]
        time_feats = np.empty((len(rows), 3), dtype=np.int64)
        for i, row in enumerate(rows):
            tab = int(row[4])
            key = (int(row[0]), int(row[1]), int(row[2]), tab, int(row[5]))
            if raw_time.get(key):
                dow, hour = raw_time[key].popleft()
            else:
                missing += 1
                dow, hour = yyyymmdd_to_weekday(row[0]), 0
            tab_bucket = tab
            if tab_bucket < 0:
                tab_bucket = 0
            if tab_bucket > 4:
                tab_bucket = 4
            time_feats[i, 0] = off_dow + dow
            time_feats[i, 1] = off_hour + hour
            time_feats[i, 2] = off_tab_hour + tab_bucket * 24 + hour
        c = ctr_feats[sp] + ctr_offsets.reshape(1, -1)
        X2 = np.concatenate([X.astype(np.int64), time_feats, c], axis=1)
        out[sp] = (X2, y, u)
    if missing:
        print(f"warning: {missing} rows missing raw hourmin; used hour=0 fallback")
    return out, final_dim


def make_user_pair_pools(y, users):
    by_user = {}
    for i, (uu, yy) in enumerate(zip(users, y)):
        if uu not in by_user:
            by_user[uu] = [[], []]  # neg, pos
        by_user[uu][1 if yy > 0.5 else 0].append(i)

    pos_rows = []
    neg_pools = []
    for negs, poss in by_user.values():
        if len(poss) and len(negs):
            neg_arr = np.asarray(negs, dtype=np.int64)
            for p in poss:
                pos_rows.append(p)
                neg_pools.append(neg_arr)
    return np.asarray(pos_rows, dtype=np.int64), neg_pools


def train_one(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=30, bs=8192,
              patience=3, neg_k=4, seed=0, device='cpu', verbose=True,
              tag='m'):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_rows, neg_pools = make_user_pair_pools(ytr, utr)
    if len(pos_rows) == 0:
        raise RuntimeError('no same-user positive/negative pairs in training data')

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        neg_rows = np.empty((len(pos_rows), neg_k), dtype=np.int64)
        for j, pool in enumerate(neg_pools):
            neg_rows[j] = pool[rng.integers(len(pool), size=neg_k)]

        order = rng.permutation(len(pos_rows))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            xp = Xtr_t[torch.from_numpy(pos_rows[sel])].to(device)
            xn = Xtr_t[torch.from_numpy(neg_rows[sel].reshape(-1))].to(device)

            opt.zero_grad(set_to_none=True)
            sp = model(xp).view(-1, 1)
            sn = model(xn).view(len(sel), neg_k)
            loss = F.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | targetctr_bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  {tag} early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model


def run(splits, data_dir, k=16, lr=0.001, epochs=30, neg_k=4, seed=0,
        device='cpu', verbose=True, n_models=3):
    enc, dim = encode_with_time_ctr(splits, data_dir)
    models = []
    seeds = [seed + 1009 * i for i in range(n_models)]
    for i, s in enumerate(seeds):
        torch.manual_seed(s)
        models.append(train_one(enc, dim, k=k, lr=lr, epochs=epochs, neg_k=neg_k,
                                seed=s, device=device, verbose=verbose,
                                tag=f"ens{i+1}/{n_models}"))
    return models, enc


@torch.no_grad()
def predict_ensemble(models, X, device='cpu'):
    preds = [m.predict(X, device=device).astype(np.float64) for m in models]
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--neg_k', type=int, default=4)
    ap.add_argument('--n_models', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}+['weekday','hour','tab_hour','video_ctr','author_ctr']; targetctr")

    models, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                      neg_k=a.neg_k, seed=a.seed, device=a.device,
                      verbose=a.out is None, n_models=a.n_models)

    X, y, users = enc[a.split]
    scores = predict_ensemble(models, X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_time_targetctr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, predict_ensemble(models, Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
