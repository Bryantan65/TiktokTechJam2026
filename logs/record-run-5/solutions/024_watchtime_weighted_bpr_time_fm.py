"""3-model multi-negative BPR FM with time features and watch-time weighted positives.

This starts from node 15 (weekday/hour/tab-hour FM) and uses raw train
play_time_ms only to weight the BPR loss for positive rows. The prediction-time
features are unchanged, so raw feedback is not used as an inference feature.
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


def _get(row, *names, default=None):
    for n in names:
        if n in row and row[n] != '':
            return row[n]
    if default is not None:
        return default
    raise KeyError('none of columns present: ' + ','.join(names))


def build_raw_lookup(data_dir):
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
                play = float(_get(r, 'play_time_ms', 'playtime_ms', default='0'))
                lookup[key].append((yyyymmdd_to_weekday(d), hour, play))
    return lookup


def encode_with_time_watch(splits, data_dir):
    enc, dim = encode(splits)
    raw = build_raw_lookup(data_dir)
    out = {}
    play_by_split = {}
    missing = 0
    off_dow = dim
    off_hour = off_dow + 7
    off_tab_hour = off_hour + 24
    final_dim = off_tab_hour + 5 * 24
    for sp, rows in splits.items():
        X, y, u = enc[sp]
        feats = np.empty((len(rows), 3), dtype=np.int64)
        play = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            tab = int(row[4])
            key = (int(row[0]), int(row[1]), int(row[2]), tab, int(row[5]))
            if raw.get(key):
                dow, hour, p = raw[key].popleft()
                play[i] = p
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
        X2 = np.concatenate([X.astype(np.int64), feats], axis=1)
        out[sp] = (X2, y, u)
        play_by_split[sp] = play
    if missing:
        print(f"warning: {missing} rows missing raw columns; used fallback")
    return out, final_dim, play_by_split


def make_user_pair_pools(y, users, play_time, rows):
    by_user = {}
    for i, (uu, yy) in enumerate(zip(users, y)):
        if uu not in by_user:
            by_user[uu] = [[], []]  # neg, pos
        by_user[uu][1 if yy > 0.5 else 0].append(i)

    # Completion/watch weight is computed only for positive anchors.  Keep it
    # bounded and normalized so the experiment changes preference emphasis, not
    # the effective learning rate.
    duration = np.asarray([max(1.0, float(r[5])) for r in rows], dtype=np.float32)
    comp = np.minimum(play_time.astype(np.float32) / duration, 3.0) / 3.0
    raw_w = 0.60 + 0.80 * comp

    pos_rows = []
    neg_pools = []
    weights = []
    for negs, poss in by_user.values():
        if len(poss) and len(negs):
            neg_arr = np.asarray(negs, dtype=np.int64)
            for p in poss:
                pos_rows.append(p)
                neg_pools.append(neg_arr)
                weights.append(float(raw_w[p]))
    weights = np.asarray(weights, dtype=np.float32)
    if len(weights):
        weights /= max(1e-6, float(weights.mean()))
    return np.asarray(pos_rows, dtype=np.int64), neg_pools, weights


def train_one(enc, dim, play, train_rows, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, neg_k=4, seed=0, device='cpu', verbose=True,
              tag='m'):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_rows, neg_pools, pos_w = make_user_pair_pools(ytr, utr, play['train'], train_rows)
    if len(pos_rows) == 0:
        raise RuntimeError('no same-user positive/negative pairs in training data')

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    w_t = torch.from_numpy(pos_w.astype(np.float32))
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
            wb = w_t[torch.from_numpy(sel)].to(device).view(-1, 1)

            opt.zero_grad(set_to_none=True)
            sp = model(xp).view(-1, 1)
            sn = model(xn).view(len(sel), neg_k)
            loss_mat = F.softplus(-(sp - sn))
            loss = (loss_mat * wb).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | watch_weighted_bpr {np.mean(losses):.4f} | valid "
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
        device='cpu', verbose=True, n_models=3):
    enc, dim, play = encode_with_time_watch(splits, data_dir)
    models = []
    seeds = [seed + 1009 * i for i in range(n_models)]
    for i, s in enumerate(seeds):
        torch.manual_seed(s)
        models.append(train_one(enc, dim, play, splits['train'], k=k, lr=lr,
                                epochs=epochs, neg_k=neg_k, seed=s,
                                device=device, verbose=verbose,
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
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--neg_k', type=int, default=4)
    ap.add_argument('--n_models', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}+['weekday','hour','tab_hour']; watch-time weighted positives")

    models, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                      neg_k=a.neg_k, seed=a.seed, device=a.device,
                      verbose=a.out is None, n_models=a.n_models)

    X, y, users = enc[a.split]
    scores = predict_ensemble(models, X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== watchtime_weighted_bpr_time_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, predict_ensemble(models, Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
