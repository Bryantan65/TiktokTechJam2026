"""Debug raw auxiliary join for multi-task FM.

The previous debug showed the raw CSVs were not located under --data_dir, so all
auxiliary labels were NaN and aux_weight changes no-oped.  This version searches
near --data_dir and the repository for the KuaiRand raw log files, then runs the
intended node-12 auxiliary BCE loss (weight 0.15) with main-head-only inference.
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
LOG_NAMES = ['log_standard_4_08_to_4_21_pure.csv',
             'log_standard_4_22_to_5_08_pure.csv']


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


def _ancestor_paths(path, n=5):
    cur = os.path.abspath(path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    out = []
    for _ in range(n):
        out.append(cur)
        par = os.path.dirname(cur)
        if par == cur:
            break
        cur = par
    return out


def find_raw_logs(data_dir):
    sol_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    roots = []
    # Direct and common sibling locations first.
    roots.extend([data_dir,
                  os.path.join(data_dir, '..'),
                  os.path.join(data_dir, '..', '..'),
                  os.path.join(data_dir, '..', '..', 'KuaiRand-Pure', 'data'),
                  os.path.join(data_dir, '..', '..', 'KuaiRand-1K', 'data'),
                  os.path.join(cwd, 'rec_datasets', 'KuaiRand-Pure', 'data'),
                  os.path.join(cwd, 'rec_datasets', 'KuaiRand-1K', 'data'),
                  os.path.join(cwd, '..', 'rec_datasets', 'KuaiRand-Pure', 'data'),
                  os.path.join(sol_dir, '..', 'rec_datasets', 'KuaiRand-Pure', 'data')])
    roots.extend(_ancestor_paths(data_dir, 5))
    roots.extend(_ancestor_paths(cwd, 4))

    seen = set()
    clean_roots = []
    for r in roots:
        ar = os.path.abspath(r)
        if ar not in seen and os.path.isdir(ar):
            seen.add(ar); clean_roots.append(ar)

    def direct_try(root, name):
        cands = [os.path.join(root, name),
                 os.path.join(root, 'data', name),
                 os.path.join(root, 'KuaiRand-Pure', 'data', name),
                 os.path.join(root, 'KuaiRand-1K', 'data', name),
                 os.path.join(root, 'rec_datasets', 'KuaiRand-Pure', 'data', name),
                 os.path.join(root, 'rec_datasets', 'KuaiRand-1K', 'data', name)]
        for p in cands:
            if os.path.isfile(p):
                return os.path.abspath(p)
        return None

    found = []
    for name in LOG_NAMES:
        hit = None
        for r in clean_roots:
            hit = direct_try(r, name)
            if hit:
                break
        if hit is None:
            # Bounded fallback search; skip hidden/cache directories.
            for r in clean_roots:
                for dirpath, dirnames, filenames in os.walk(r):
                    dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('pred_cache', '__pycache__')]
                    if name in filenames:
                        hit = os.path.abspath(os.path.join(dirpath, name))
                        break
                    # avoid very deep walks from ancestors
                    rel = os.path.relpath(dirpath, r)
                    if rel != '.' and rel.count(os.sep) >= 4:
                        dirnames[:] = []
                if hit:
                    break
        found.append(hit)
    print('raw log search roots:', clean_roots[:8])
    print('raw log paths:', found)
    return found if all(p is not None for p in found) else None


def read_aux_lookup(data_dir):
    paths = find_raw_logs(data_dir)
    if paths is None:
        print('raw auxiliary csv files not found after robust search; MTL auxiliaries disabled')
        return None, AUX_BIN_COLS + ['watch_ratio']
    q = defaultdict(deque)
    cols = None
    bad_header = False
    n_rows = 0
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
                    print('aux columns:', cols)
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
                n_rows += 1
    print(f'loaded raw aux rows={n_rows}, unique keys={len(q)}')
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
    total = 0
    for sp, rows in splits.items():
        arr = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            key = (_int_val(row[0]), str(row[1]), str(row[2]), str(row[3]),
                   _int_val(row[4]), max(_int_val(row[5]), 1))
            if lookup.get(key):
                arr[i] = lookup[key].popleft()
            else:
                miss += 1
            total += 1
        out[sp] = arr
    train_finite = np.isfinite(out['train']).mean(axis=0) if 'train' in out else np.zeros(n_aux)
    train_mean = np.nanmean(out['train'], axis=0) if 'train' in out else np.zeros(n_aux)
    print(f"raw auxiliaries joined for MTL rawpath; missing rows={miss}/{total}; aux={names}; train_finite={train_finite}; train_mean={train_mean}")
    return out, names


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True, aux_weight=0.15):
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
        aux_losses = []
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
                aux_loss = aux_loss / aux_count
                loss = loss + aux_weight * aux_loss
                aux_losses.append(float(aux_loss.detach().cpu()))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            al = np.mean(aux_losses) if aux_losses else float('nan')
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} aux {al:.4f} | valid "
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+raw_aux_mtl_rawpath")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device,
                     verbose=a.out is None, aux_weight=0.15)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== multitask_fm_rawpath_dev (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
