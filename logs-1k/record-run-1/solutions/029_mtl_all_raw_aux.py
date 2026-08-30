"""FM with a wider set of raw-feedback auxiliary tasks.

Node 12 used click/like/follow/comment/forward plus clipped watch ratio.  This
variant keeps the same shared FM/main-head architecture but automatically adds
all usable numeric feedback columns from the raw logs as auxiliary BCE targets
(excluding id/time/item metadata), with vectorized masked auxiliary loss so the
extra tasks are cheap.
"""
import argparse
import csv
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

PREFERRED_AUX = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
META_COLS = set([
    'date', 'user_id', 'user', 'video_id', 'video', 'author_id', 'author', 'tab',
    'duration_ms', 'video_duration', 'duration', 'hourmin', 'time_ms', 'time',
    'request_id', 'index', 'row_id'
])
EXCLUDE_AUX = set(['label', 'is_long_view', 'long_view'])


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


def _raw_paths(data_dir):
    paths = [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
             os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]
    if all(os.path.isfile(p) for p in paths):
        return paths
    return None


def _choose_aux_cols(header):
    cols = []
    for c in PREFERRED_AUX:
        if c in header and c not in cols:
            cols.append(c)
    for c in header:
        cl = c.lower()
        if c in cols or cl in META_COLS or cl in EXCLUDE_AUX:
            continue
        # KuaiRand feedbacks are mostly is_* binaries; include play time as a
        # normalized continuous confidence target below.
        if cl.startswith('is_'):
            cols.append(c)
    play_c = None
    for c in ['play_time_ms', 'playtime_ms', 'play_time']:
        if c in header:
            play_c = c
            break
    return cols, play_c


def read_aux_lookup(data_dir):
    paths = _raw_paths(data_dir)
    # Fallback to node-12 task shape if raw files are unavailable; standalone still works.
    fallback_names = PREFERRED_AUX + ['watch_ratio']
    if paths is None:
        return None, fallback_names
    q = defaultdict(deque)
    cols = None
    bad_header = False
    names = None
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for rec in reader:
                if cols is None:
                    header = reader.fieldnames or list(rec.keys())
                    date_c = _find_col(rec, ['date'])
                    user_c = _find_col(rec, ['user_id', 'user'])
                    video_c = _find_col(rec, ['video_id', 'video'])
                    author_c = _find_col(rec, ['author_id', 'author'])
                    tab_c = _find_col(rec, ['tab'])
                    dur_c = _find_col(rec, ['duration_ms', 'video_duration', 'duration'])
                    aux_cols, play_c = _choose_aux_cols(header)
                    cols = (date_c, user_c, video_c, author_c, tab_c, dur_c, aux_cols, play_c)
                    names = list(aux_cols) + (['watch_ratio'] if play_c is not None else [])
                    bad_header = any(c is None for c in [date_c, user_c, video_c, author_c, tab_c, dur_c])
                    if not names:
                        names = fallback_names
                date_c, user_c, video_c, author_c, tab_c, dur_c, aux_cols, play_c = cols
                if bad_header:
                    continue
                dur = max(_int_val(rec[dur_c]), 1)
                vals = []
                for c in aux_cols:
                    vals.append(float(_int_val(rec.get(c, 0)) > 0))
                if play_c is not None:
                    vals.append(float(np.clip(_float_val(rec.get(play_c, 0.0)) / dur, 0.0, 1.0)))
                key = (_int_val(rec[date_c]), str(rec[user_c]), str(rec[video_c]),
                       str(rec[author_c]), _int_val(rec[tab_c]), dur)
                q[key].append(np.asarray(vals, dtype=np.float32))
    return q, names


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
    print(f"raw all-aux joined; missing rows={miss}; n_aux={n_aux}; aux={names}")
    return out, names


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=35, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True, aux_weight=0.10):
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
            if ab.numel() > 0 and aux_weight > 0:
                mask = torch.isfinite(ab)
                if bool(mask.any()):
                    aux_loss_all = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[:, 1:], torch.nan_to_num(ab, nan=0.0), reduction='none')
                    loss = loss + aux_weight * aux_loss_all[mask].mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
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
    ap.add_argument('--epochs', type=int, default=35)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+all_raw_aux_mtl")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device,
                     verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== multitask_fm_all_raw_aux (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
