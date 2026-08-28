"""Mixed 3-model BPR FM ensemble: two tab-hour time models plus one history model.

Node 26 showed a 33% diverse member can improve over three identical tab-hour
models. This keeps the same 2/3 incumbent tab-hour vote but replaces the simple
weekday+hour member with node 23's compact leakage-safe user history features,
so the history signal is tested at a readable 33% weight.
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
                lookup[key].append((yyyymmdd_to_weekday(d), parse_hour(_get(r, 'hourmin'))))
    return lookup


def ctr_bin(pos, cnt):
    if cnt <= 0:
        return 0
    r = pos / cnt
    if pos <= 0:
        return 1
    if r < 0.34:
        return 2
    if r < 0.67:
        return 3
    return 4


def uv_state(pos, cnt):
    if cnt <= 0:
        return 0
    if pos <= 0:
        return 1
    if pos >= cnt:
        return 2
    return 3


def build_light_history(splits):
    ua = {}
    uv = {}
    hist = {}
    for sp in ('train', 'valid', 'test'):
        rows = splits[sp]
        feats = np.empty((len(rows), 2), dtype=np.int64)
        update = (sp == 'train')
        for i, row in enumerate(rows):
            u = int(row[1]); v = int(row[2]); a = int(row[3])
            c, p = ua.get((u, a), (0, 0))
            c2, p2 = uv.get((u, v), (0, 0))
            feats[i, 0] = ctr_bin(p, c)
            feats[i, 1] = uv_state(p2, c2)
            if update:
                y = 1 if float(row[6]) > 0.5 else 0
                ua[(u, a)] = (c + 1, p + y)
                uv[(u, v)] = (c2 + 1, p2 + y)
        hist[sp] = feats
    return hist


def encode_with_time(splits, data_dir, include_history=False):
    enc, dim = encode(splits)
    raw_time = build_raw_time_lookup(data_dir)
    hist = build_light_history(splits) if include_history else None
    out = {}
    missing = 0
    off_dow = dim
    off_hour = off_dow + 7
    off_tab_hour = off_hour + 24
    if include_history:
        off_ua_ctr = off_tab_hour + 5 * 24
        off_uv_state = off_ua_ctr + 5
        final_dim = off_uv_state + 4
        n_extra = 5
        hist_offsets = np.asarray([off_ua_ctr, off_uv_state], dtype=np.int64)
    else:
        final_dim = off_tab_hour + 5 * 24
        n_extra = 3
        hist_offsets = None

    for sp, rows in splits.items():
        X, y, u = enc[sp]
        feats = np.empty((len(rows), n_extra), dtype=np.int64)
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
            feats[i, 0] = off_dow + dow
            feats[i, 1] = off_hour + hour
            feats[i, 2] = off_tab_hour + tab_bucket * 24 + hour
        if include_history:
            feats[:, 3:5] = hist[sp] + hist_offsets.reshape(1, -1)
        X2 = np.concatenate([X.astype(np.int64), feats], axis=1)
        out[sp] = (X2, y, u)
    if missing:
        print(f"warning: {missing} rows missing raw hourmin; used hour=0 fallback")
    return out, final_dim


def make_user_pair_pools(y, users):
    by_user = {}
    for i, (uu, yy) in enumerate(zip(users, y)):
        if uu not in by_user:
            by_user[uu] = [[], []]
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


def train_one(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, neg_k=4, seed=0, device='cpu', verbose=True,
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
            print(f"  {tag} epoch {ep:2d} | mixed_hist_bpr {np.mean(losses):.4f} | valid "
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


def run(splits, data_dir, k=16, lr=0.001, epochs=40, neg_k=4, seed=0,
        device='cpu', verbose=True):
    enc_full, dim_full = encode_with_time(splits, data_dir, include_history=False)
    enc_hist, dim_hist = encode_with_time(splits, data_dir, include_history=True)
    members = []
    plan = [
        ('tabhour1/3', enc_full, dim_full, seed),
        ('tabhour2/3', enc_full, dim_full, seed + 1009),
        ('history3/3', enc_hist, dim_hist, seed + 2018),
    ]
    for tag, enc_i, dim_i, s in plan:
        torch.manual_seed(s)
        m = train_one(enc_i, dim_i, k=k, lr=lr, epochs=epochs, neg_k=neg_k,
                      seed=s, device=device, verbose=verbose, tag=tag)
        members.append((m, enc_i))
    return members


@torch.no_grad()
def predict_mixed(members, split, device='cpu'):
    preds = []
    for model, enc in members:
        X, y, users = enc[split]
        preds.append(model.predict(X, device=device).astype(np.float64))
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--neg_k', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}+two tab-hour members and one tab-hour+history member")

    members = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                  neg_k=a.neg_k, seed=a.seed, device=a.device,
                  verbose=a.out is None)

    scores = predict_mixed(members, a.split, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== mixed_tabhour_history_ensemble (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            _, ys, us = members[0][1][sp]
            r = evaluate(us, ys, predict_mixed(members, sp, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
