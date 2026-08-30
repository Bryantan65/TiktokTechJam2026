"""MTL FM plus aggregate-rate heuristic, fused per user.

Branch from node 12.  The MTL FM is the current best model; node 22 showed a
transparent aggregate-rate heuristic is much weaker standalone but aligned with
within-user ranking.  This script tests whether it adds complementary memorized
user-video/user-author signal when blended at a readable 30% weight after
per-user z-score normalization.
"""
import argparse
import csv
import math
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

AUX_BIN_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_tasks, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(n_tasks, dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))

    def forward_all(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W[:, X].sum(2).transpose(0, 1)
        return inter[:, None] + lin + self.b[None, :]

    def forward(self, X):
        return self.forward_all(X)[:, 0]

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        outs = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            outs.append(self(xb).cpu().numpy())
        return np.concatenate(outs)


def _int_val(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def _float_val(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _find_col(rec, names):
    for n in names:
        if n in rec:
            return n
    return None


def read_aux_lookup(data_dir):
    paths = [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
             os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]
    if not all(os.path.isfile(p) for p in paths):
        return None, AUX_BIN_COLS + ['watch_ratio']
    q = defaultdict(deque)
    cols = None
    bad_header = False
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                if cols is None:
                    date_c = _find_col(rec, ['date'])
                    user_c = _find_col(rec, ['user_id', 'user'])
                    video_c = _find_col(rec, ['video_id', 'video'])
                    author_c = _find_col(rec, ['author_id', 'author'])
                    tab_c = _find_col(rec, ['tab'])
                    dur_c = _find_col(rec, ['duration_ms', 'video_duration', 'duration'])
                    play_c = _find_col(rec, ['play_time_ms', 'playtime_ms', 'play_time'])
                    aux_cols = [_find_col(rec, [c]) for c in AUX_BIN_COLS]
                    cols = (date_c, user_c, video_c, author_c, tab_c, dur_c, play_c, aux_cols)
                    bad_header = any(c is None for c in [date_c, user_c, video_c, author_c, tab_c, dur_c])
                date_c, user_c, video_c, author_c, tab_c, dur_c, play_c, aux_cols = cols
                if bad_header:
                    continue
                dur = max(_int_val(rec[dur_c]), 1)
                vals = []
                for c in aux_cols:
                    vals.append(np.nan if c is None else float(_int_val(rec[c]) > 0))
                if play_c is None:
                    vals.append(np.nan)
                else:
                    vals.append(float(np.clip(_float_val(rec[play_c]) / dur, 0.0, 1.0)))
                key = (_int_val(rec[date_c]), str(rec[user_c]), str(rec[video_c]),
                       str(rec[author_c]), _int_val(rec[tab_c]), dur)
                q[key].append(np.asarray(vals, dtype=np.float32))
    return q, AUX_BIN_COLS + ['watch_ratio']


def attach_aux(splits, data_dir):
    lookup, names = read_aux_lookup(data_dir)
    n_aux = len(names)
    out = {}
    if lookup is None:
        for sp, rows in splits.items():
            out[sp] = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        return out, names
    miss = 0
    for sp, rows in splits.items():
        arr = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            key = (_int_val(row[0]), str(row[1]), str(row[2]), str(row[3]),
                   _int_val(row[4]), max(_int_val(row[5]), 1))
            if lookup.get(key):
                arr[i] = lookup[key].popleft()
            else:
                miss += 1
        out[sp] = arr
    print(f"raw auxiliaries joined for MTL; missing rows={miss}; aux={names}")
    return out, names


def run_mtl(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, seed=0, device='cpu', aux_weight=0.15):
    enc, dim = encode(splits)
    aux, aux_names = attach_aux(splits, data_dir)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Yaux = aux['train']

    model = MultiTaskFM(dim, n_tasks=1 + Yaux.shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    aux_t = torch.from_numpy(Yaux.astype(np.float32))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            ab = aux_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model.forward_all(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], yb)
            aux_loss = 0.0
            aux_count = 0
            for j in range(ab.shape[1]):
                tgt = ab[:, j]
                mask = torch.isfinite(tgt)
                if bool(mask.any()):
                    aux_loss = aux_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[mask, j + 1], tgt[mask])
                    aux_count += 1
            if aux_count:
                loss = loss + aux_weight * aux_loss / aux_count
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, enc


def dur_bucket(duration_ms):
    d = int(duration_ms)
    if d < 7000:
        return 0
    if d < 15000:
        return 1
    if d < 30000:
        return 2
    if d < 60000:
        return 3
    return 4


def logit(p):
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def row_vals(r):
    return (str(r[1]), str(r[2]), str(r[3]), int(r[4]), dur_bucket(r[5]), int(r[0]))


def build_stats(rows, specs):
    cnts = [defaultdict(int) for _ in specs]
    sums = [defaultdict(float) for _ in specs]
    for r in rows:
        y = float(r[6])
        vals = row_vals(r)
        for j, cols in enumerate(specs):
            key = tuple(vals[c] for c in cols)
            if len(key) == 1:
                key = key[0]
            cnts[j][key] += 1
            sums[j][key] += y
    return cnts, sums


def agg_predict(rows, train_rows):
    prior = float(np.mean([float(r[6]) for r in train_rows]))
    specs = [(0, 1), (0, 2), (0, 2, 3), (0, 3), (1,), (2,), (2, 3), (3,), (4,)]
    weights = [1.20, 1.00, 0.45, 0.35, 0.75, 0.55, 0.25, 0.15, 0.10]
    alphas = [8.0, 20.0, 30.0, 50.0, 25.0, 35.0, 50.0, 100.0, 100.0]
    cnts, sums = build_stats(train_rows, specs)
    print(f"prior={prior:.5f}; built {len(specs)} aggregate tables")
    out = np.empty(len(rows), dtype=np.float64)
    prior_log = logit(prior)
    for i, r in enumerate(rows):
        vals = row_vals(r)
        s = 0.15 * prior_log
        wsum = 0.15
        for j, cols in enumerate(specs):
            key = tuple(vals[c] for c in cols)
            if len(key) == 1:
                key = key[0]
            c = cnts[j].get(key, 0)
            sm = sums[j].get(key, 0.0)
            rate = (sm + alphas[j] * prior) / (c + alphas[j])
            s += weights[j] * logit(rate)
            wsum += abs(weights[j])
        out[i] = s / max(wsum, 1e-6)
    return out


def zscore_by_user(scores, users):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    _, inv = np.unique(users, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sm = np.bincount(inv, weights=scores)
    sm2 = np.bincount(inv, weights=scores * scores)
    mean = sm / np.maximum(cnt, 1.0)
    var = sm2 / np.maximum(cnt, 1.0) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return (scores - mean[inv]) / std[inv]


def cached_mtl_predictions(splits, data_dir, target, seed, device, split_name):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache', f'node12_mtl_{split_name}_seed{int(seed)}.npy')
    if os.path.isfile(cache_path):
        preds = np.load(cache_path)
        enc, _ = encode(splits)
        if len(preds) == len(enc[target][0]):
            print(f"loaded cached MTL predictions: {cache_path}")
            return preds.astype(np.float64), enc
        print(f"ignoring stale cache length {len(preds)} at {cache_path}")
    model, enc = run_mtl(splits, data_dir=data_dir, seed=seed, device=device)
    X = enc[target][0]
    preds = model.predict(X, device=device).astype(np.float64)
    np.save(cache_path, preds)
    print(f"saved cached MTL predictions: {cache_path}")
    return preds, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+raw_aux_mtl+agg_zblend")

    mtl, enc = cached_mtl_predictions(splits, a.data_dir, target, a.seed, a.device, a.split)
    agg = agg_predict(splits[target], splits['train'])
    users = enc[target][2]
    zm = zscore_by_user(mtl, users)
    za = zscore_by_user(agg, users)
    preds = 0.70 * zm + 0.30 * za
    # Seeded microscopic noise only breaks exact ties after normalization.
    rng = np.random.default_rng(int(a.seed))
    preds = preds + rng.normal(0.0, 1e-9, size=len(preds))

    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else:
        print(preds[:10])
