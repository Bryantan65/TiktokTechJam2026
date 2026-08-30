"""Debug node 26: robustly find raw KuaiRand CSVs and apply non-unit IPS weights.

Uses the random-exposure log only to estimate author-tab exposure weights;
random labels/rows are never used for training.  Also robustly finds standard
logs so this remains the same MTL setup as the best model plus the new weights.
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
STANDARD_LOGS = ['log_standard_4_08_to_4_21_pure.csv',
                 'log_standard_4_22_to_5_08_pure.csv']
RANDOM_LOG = 'log_random_4_22_to_5_08_pure.csv'


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


def candidate_roots(data_dir):
    roots = []
    p = os.path.abspath(data_dir)
    for _ in range(8):
        if p and p not in roots:
            roots.append(p)
        q = os.path.dirname(p)
        if q == p:
            break
        p = q
    # common sibling layout: rec_datasets/KuaiRand-1K/data -> rec_datasets/KuaiRand-Pure/data
    more = []
    for r in roots:
        more += [os.path.join(r, 'KuaiRand-Pure', 'data'),
                 os.path.join(r, 'rec_datasets', 'KuaiRand-Pure', 'data'),
                 os.path.join(os.path.dirname(r), 'KuaiRand-Pure', 'data'),
                 os.path.join(os.path.dirname(r), 'data')]
    for r in more:
        ar = os.path.abspath(r)
        if ar not in roots:
            roots.append(ar)
    return roots


def find_file(data_dir, filename):
    # First try cheap exact locations.
    for root in candidate_roots(data_dir):
        p = os.path.join(root, filename)
        if os.path.isfile(p):
            return os.path.abspath(p)
    # Then recursively search ancestors/sibling dataset dirs, but prune large irrelevant dirs.
    seen = set()
    for root in candidate_roots(data_dir):
        if not os.path.isdir(root):
            continue
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {'.git', '.venv', '__pycache__', 'solutions', 'pred_cache'}]
            if filename in filenames:
                return os.path.abspath(os.path.join(dirpath, filename))
    return None


def find_standard_logs(data_dir):
    paths = [find_file(data_dir, nm) for nm in STANDARD_LOGS]
    if any(p is None for p in paths):
        print("standard logs not fully found:", dict(zip(STANDARD_LOGS, paths)))
        return None
    print("standard logs:", paths)
    return paths


def read_aux_lookup(data_dir):
    paths = find_standard_logs(data_dir)
    n_aux = len(AUX_BIN_COLS) + 1
    if paths is None:
        return None, AUX_BIN_COLS + ['watch_ratio']
    q = defaultdict(deque)
    cols = None
    bad_header = False
    n = 0
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
                    print("raw columns:", cols, "bad_header=", bad_header)
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
                n += 1
    print(f"read {n} raw standard rows into aux lookup; keys={len(q)}")
    return q, AUX_BIN_COLS + ['watch_ratio']


def attach_aux(splits, data_dir):
    lookup, names = read_aux_lookup(data_dir)
    n_aux = len(names)
    out = {}
    if lookup is None:
        print("WARNING: no aux lookup; using NaN auxiliary targets")
        for sp, rows in splits.items():
            out[sp] = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        return out, names
    miss = 0
    hit = 0
    for sp, rows in splits.items():
        arr = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            key = (_int_val(row[0]), str(row[1]), str(row[2]), str(row[3]),
                   _int_val(row[4]), max(_int_val(row[5]), 1))
            if lookup.get(key):
                arr[i] = lookup[key].popleft()
                hit += 1
            else:
                miss += 1
        out[sp] = arr
    print(f"raw auxiliaries joined; hit={hit} missing={miss}; aux={names}")
    return out, names


def author_tab_key_from_row(r):
    return (str(r[3]), _int_val(r[4]))


def build_propensity_weights(train_rows, data_dir, clip_lo=0.35, clip_hi=2.5, alpha=50.0):
    rand_path = find_file(data_dir, RANDOM_LOG)
    if rand_path is None:
        print("WARNING: random exposure log not found after recursive search; using unit weights")
        return np.ones(len(train_rows), dtype=np.float32)
    print("random log:", rand_path)

    std = defaultdict(int)
    for r in train_rows:
        std[author_tab_key_from_row(r)] += 1

    rnd = defaultdict(int)
    n_rnd = 0
    with open(rand_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = None
        for rec in reader:
            if cols is None:
                author_c = _find_col(rec, ['author_id', 'author'])
                tab_c = _find_col(rec, ['tab'])
                print("random columns:", (author_c, tab_c))
                if author_c is None or tab_c is None:
                    print("random log missing author/tab; using unit weights")
                    return np.ones(len(train_rows), dtype=np.float32)
                cols = (author_c, tab_c)
            author_c, tab_c = cols
            rnd[(str(rec[author_c]), _int_val(rec[tab_c]))] += 1
            n_rnd += 1

    keys = set(std) | set(rnd)
    n_std = float(sum(std.values()))
    n_rnd_f = float(max(n_rnd, 1))
    k = float(max(len(keys), 1))
    table = {}
    for key in keys:
        ps = (std.get(key, 0) + alpha) / (n_std + alpha * k)
        pr = (rnd.get(key, 0) + alpha) / (n_rnd_f + alpha * k)
        table[key] = float(np.clip(pr / max(ps, 1e-12), clip_lo, clip_hi))

    w = np.asarray([table.get(author_tab_key_from_row(r), 1.0) for r in train_rows], dtype=np.float32)
    w /= max(float(np.mean(w)), 1e-6)
    qs = np.quantile(w, [0, .01, .1, .5, .9, .99, 1.0])
    print(f"propensity weights: random_rows={n_rnd} keys={len(keys)} "
          f"mean={float(w.mean()):.6f} std={float(w.std()):.6f} quantiles={qs}")
    return w


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True, aux_weight=0.15):
    enc, dim = encode(splits)
    aux, aux_names = attach_aux(splits, data_dir)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Yaux = aux['train']
    sw = build_propensity_weights(splits['train'], data_dir)

    model = MultiTaskFM(dim, n_tasks=1 + Yaux.shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    aux_t = torch.from_numpy(Yaux.astype(np.float32))
    sw_t = torch.from_numpy(sw.astype(np.float32))

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
            wb = sw_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model.forward_all(xb)
            main = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], yb, reduction='none')
            loss = (main * wb).sum() / torch.clamp(wb.sum(), min=1.0)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+raw_aux_mtl+author_tab_ips")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device,
                     verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(scores[:10])
