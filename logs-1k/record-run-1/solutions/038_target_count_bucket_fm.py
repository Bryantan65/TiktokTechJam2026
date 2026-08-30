"""Multi-task FM plus leakage-safe target/count encoding buckets.

Kaggle CTR practice often uses smoothed target encoding and count encoding for
high-cardinality categorical ids; here the encodings are converted to categorical
buckets and appended as additional FM fields.  Train rows use leave-one-out
statistics, while valid/test rows use statistics from the full training split.
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

AUX_BIN_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
STAT_SPECS = [
    ('v', lambda r: str(r[2])),
    ('a', lambda r: str(r[3])),
    ('vt', lambda r: (str(r[2]), int(r[4]))),
    ('at', lambda r: (str(r[3]), int(r[4]))),
    ('td', lambda r: (int(r[4]), dur_bucket(r[5]))),
]


def dur_bucket(x):
    try:
        v = int(float(x))
    except Exception:
        v = 0
    # coarse log-duration buckets, independent of starter-kit internals
    return int(np.clip(np.searchsorted([5_000, 10_000, 20_000, 30_000, 45_000, 60_000,
                                        90_000, 120_000, 180_000, 300_000], v), 0, 10))


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
        print('raw auxiliaries unavailable; running as single-task plus TE buckets')
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


def build_stat_tables(train_rows):
    tables = []
    y = np.asarray([float(r[6]) for r in train_rows], dtype=np.float32)
    glob = float(y.mean()) if len(y) else 0.0
    for name, fn in STAT_SPECS:
        cnt = defaultdict(int)
        pos = defaultdict(float)
        for r in train_rows:
            k = fn(r)
            cnt[k] += 1
            pos[k] += float(r[6])
        tables.append((name, fn, cnt, pos))
    return tables, glob


def rate_bucket(pos, cnt, glob, alpha=20.0):
    rate = (pos + alpha * glob) / (cnt + alpha) if cnt > 0 else glob
    return int(np.clip(np.floor(rate * 30.0), 0, 29))


def count_bucket(cnt):
    return int(np.clip(np.floor(np.log2(max(cnt, 0) + 1.0)), 0, 20))


def extra_tokens_for_split(rows, tables, glob, split_name):
    tokens = []
    is_train = split_name == 'train'
    for r in rows:
        y = float(r[6]) if is_train else 0.0
        feats = []
        for name, fn, cnt, pos in tables:
            k = fn(r)
            c = cnt.get(k, 0)
            p = pos.get(k, 0.0)
            if is_train:
                c = max(c - 1, 0)
                p = max(p - y, 0.0)
            feats.append(f'{name}_cb={count_bucket(c)}')
            feats.append(f'{name}_rb={rate_bucket(p, c, glob)}')
        tokens.append(feats)
    return tokens


def append_extra_fields(Xbase, splits):
    tables, glob = build_stat_tables(splits['train'])
    all_tokens = {sp: extra_tokens_for_split(rows, tables, glob, sp) for sp, rows in splits.items()}
    n_extra = len(next(iter(all_tokens.values()))[0]) if splits['train'] else 0
    maps = [dict() for _ in range(n_extra)]
    next_id = int(Xbase['dim'])
    # Fit vocabulary on train buckets; unseen buckets get their own deterministic id if needed.
    for feats in all_tokens['train']:
        for j, tok in enumerate(feats):
            if tok not in maps[j]:
                maps[j][tok] = next_id
                next_id += 1
    Xout = {}
    for sp, feats_list in all_tokens.items():
        arr = np.empty((len(feats_list), n_extra), dtype=np.int64)
        for i, feats in enumerate(feats_list):
            for j, tok in enumerate(feats):
                if tok not in maps[j]:
                    maps[j][tok] = next_id
                    next_id += 1
                arr[i, j] = maps[j][tok]
        X, y, u = Xbase[sp]
        Xout[sp] = (np.concatenate([X.astype(np.int64), arr], axis=1), y, u)
    print(f"added {n_extra} target/count bucket fields; global rate={glob:.4f}; dim {Xbase['dim']}->{next_id}")
    return Xout, next_id


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True, aux_weight=0.15):
    base_enc, base_dim = encode(splits)
    base_enc['dim'] = base_dim
    enc, dim = append_extra_fields(base_enc, splits)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+TE_count_buckets+raw_aux_mtl")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device,
                     verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== target_count_bucket_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
