"""Refined multi-task BPR time FM with play-time auxiliary signal.

Node 16 used sparse binary feedback auxiliaries and was slightly worse than the
best time BPR model. This version keeps the same main multi-negative BPR head but
uses a more directly related continuous watch-time target (log play_time_ms,
standardized on train) plus click as auxiliary representation learning.
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


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        # task 0 = main rank; task 1 = click BCE; task 2 = log play-time MSE
        self.W = torch.nn.Parameter(torch.zeros((3, dim), dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((3,), dtype=torch.float32))

    def shared_interaction(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def score(self, X, task=0):
        inter = self.shared_interaction(X)
        return self.b[task] + self.W[task, X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self.score(xb, 0).cpu().numpy())
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


def _get_float_default(row, names, default=0.0):
    if isinstance(names, str):
        names = [names]
    for name in names:
        v = row.get(name, '')
        if v != '' and v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default


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
                click = 1.0 if _get_float_default(r, 'is_click', 0.0) > 0 else 0.0
                play = _get_float_default(r, ['play_time_ms', 'play_ms', 'play_time'], 0.0)
                lookup[key].append((yyyymmdd_to_weekday(d), hour, click, np.log1p(max(0.0, play))))
    return lookup


def encode_with_time_aux(splits, data_dir):
    enc, dim = encode(splits)
    raw_lookup = build_raw_lookup(data_dir)
    out = {}
    missing = 0
    off_dow = dim
    off_hour = off_dow + 7
    off_tab_hour = off_hour + 24
    final_dim = off_tab_hour + 5 * 24
    train_play = []

    tmp = {}
    for sp, rows in splits.items():
        X, y, u = enc[sp]
        feats = np.empty((len(rows), 3), dtype=np.int64)
        click = np.zeros(len(rows), dtype=np.float32)
        playlog = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            tab = int(row[4])
            key = (int(row[0]), int(row[1]), int(row[2]), tab, int(row[5]))
            if raw_lookup.get(key):
                dow, hour, c, p = raw_lookup[key].popleft()
                click[i] = c
                playlog[i] = p
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
        tmp[sp] = (X2, y, u, click, playlog)
        if sp == 'train':
            train_play = playlog

    mu = float(np.mean(train_play))
    sd = float(np.std(train_play) + 1e-6)
    for sp, (X2, y, u, click, playlog) in tmp.items():
        playz = ((playlog - mu) / sd).astype(np.float32)
        out[sp] = (X2, y, u, click.astype(np.float32), playz)
    if missing:
        print(f"warning: {missing} rows missing raw fields; used hour=0/aux=0 fallback")
    print(f"play_time log mean={mu:.4f} std={sd:.4f}")
    return out, final_dim


def make_user_pair_pools(y, users):
    by_user = {}
    for i, (uu, yy) in enumerate(zip(users, y)):
        if uu not in by_user:
            by_user[uu] = [[], []]
        by_user[uu][1 if yy > 0.5 else 0].append(i)
    pos_rows, neg_pools = [], []
    for negs, poss in by_user.values():
        if len(poss) and len(negs):
            neg_arr = np.asarray(negs, dtype=np.int64)
            for p in poss:
                pos_rows.append(p)
                neg_pools.append(neg_arr)
    return np.asarray(pos_rows, dtype=np.int64), neg_pools


def train_one(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, neg_k=4, click_weight=0.01, play_weight=0.02,
              seed=0, device='cpu', verbose=True, tag='m'):
    Xtr, ytr, utr, click_tr, play_tr = enc['train']
    Xva, yva, uva, _, _ = enc['valid']
    pos_rows, neg_pools = make_user_pair_pools(ytr, utr)
    if len(pos_rows) == 0:
        raise RuntimeError('no same-user positive/negative pairs in training data')

    model = MultiTaskFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    click_t = torch.from_numpy(click_tr.astype(np.float32))
    play_t = torch.from_numpy(play_tr.astype(np.float32))
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
            pos_sel = pos_rows[sel]
            neg_sel = neg_rows[sel]
            xp = Xtr_t[torch.from_numpy(pos_sel)].to(device)
            xn = Xtr_t[torch.from_numpy(neg_sel.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model.score(xp, 0).view(-1, 1)
            sn = model.score(xn, 0).view(len(sel), neg_k)
            loss = F.softplus(-(sp - sn)).mean()

            aux_rows = np.concatenate([pos_sel, neg_sel[:, 0]])
            xa = Xtr_t[torch.from_numpy(aux_rows)].to(device)
            yc = click_t[torch.from_numpy(aux_rows)].to(device)
            yp = play_t[torch.from_numpy(aux_rows)].to(device)
            sc = model.score(xa, 1)
            spm = model.score(xa, 2)
            loss = loss + click_weight * F.binary_cross_entropy_with_logits(sc, yc)
            loss = loss + play_weight * F.smooth_l1_loss(spm, yp)

            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | bpr_click_play {np.mean(losses):.4f} | valid "
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
    enc, dim = encode_with_time_aux(splits, data_dir)
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
          f"fields={FIELDS}+['weekday','hour','tab_hour']+click/play_aux")
    models, enc = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                      neg_k=a.neg_k, seed=a.seed, device=a.device,
                      verbose=a.out is None, n_models=a.n_models)
    X, y, users, click, play = enc[a.split]
    scores = predict_ensemble(models, X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== mtl_playtime_bpr_time_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us, _, _ = enc[sp]
            r = evaluate(us, ys, predict_ensemble(models, Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
