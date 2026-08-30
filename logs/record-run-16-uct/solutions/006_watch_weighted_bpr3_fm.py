"""FM with within-user BPR and watch-time weighted positive pairs.

Refines the 3-negative BPR run.  The extra negatives gave broader coverage but
made every positive equally important; here each positive's pairwise losses are
weighted by its raw play_time_ms / duration_ms signal read from the KuaiRand CSVs.
Weights are normalized inside each minibatch so the learning-rate scale stays
close to the unweighted BPR objective.
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


def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _row_key_from_tuple(r):
    return (_to_int(r[0]), str(r[1]), str(r[2]), str(r[3]), _to_int(r[4]), _to_int(r[5]))


def _first_present(d, names):
    for n in names:
        if n in d and d[n] != '':
            return d[n]
    return None


def _candidate_path(data_dir, filename):
    p = os.path.join(data_dir, filename)
    if os.path.isfile(p):
        return p
    p = os.path.join(data_dir, 'data', filename)
    if os.path.isfile(p):
        return p
    return os.path.join(data_dir, filename)


def load_watch_weights(data_dir, rows, alpha=0.5, cap_ratio=2.0):
    """Return one positive-pair weight per loaded train row.

    The starter data keeps row order from the two standard log CSVs, but split
    rows are tuples without play_time_ms.  Build queues by the tuple-visible key
    and pop in split order, which also handles duplicated impressions.
    """
    files = [
        'log_standard_4_08_to_4_21_pure.csv',
        'log_standard_4_22_to_5_08_pure.csv',
    ]
    q = defaultdict(deque)
    n_raw = 0
    for fn in files:
        path = _candidate_path(data_dir, fn)
        if not os.path.isfile(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for rec in reader:
                date = _to_int(_first_present(rec, ['date', 'upload_date', 'day']))
                user = str(_first_present(rec, ['user_id', 'userId', 'user']) or '')
                video = str(_first_present(rec, ['video_id', 'videoId', 'item_id', 'itemId']) or '')
                author = str(_first_present(rec, ['author_id', 'authorId']) or '')
                tab = _to_int(_first_present(rec, ['tab']))
                dur = _to_int(_first_present(rec, ['duration_ms', 'duration', 'video_duration']))
                play = _to_int(_first_present(rec, ['play_time_ms', 'play_time', 'playtime_ms', 'watch_time_ms']))
                key = (date, user, video, author, tab, dur)
                ratio = float(play) / float(max(dur, 1))
                ratio = max(0.0, min(cap_ratio, ratio))
                q[key].append(1.0 + alpha * ratio)
                n_raw += 1

    weights = np.ones(len(rows), dtype=np.float32)
    matched = 0
    for i, r in enumerate(rows):
        key = _row_key_from_tuple(r)
        if q.get(key):
            weights[i] = q[key].popleft()
            matched += 1
    return weights, matched, n_raw


def make_user_pairs(y, users, row_weights):
    by_user_pos = {}
    by_user_neg = {}
    for i, (yy, u) in enumerate(zip(y, users)):
        if yy > 0.5:
            by_user_pos.setdefault(u, []).append(i)
        else:
            by_user_neg.setdefault(u, []).append(i)

    pos_all = []
    pos_w = []
    neg_pools = []
    for u, pos in by_user_pos.items():
        neg = by_user_neg.get(u)
        if neg:
            neg_arr = np.asarray(neg, dtype=np.int64)
            for p in pos:
                pos_all.append(p)
                pos_w.append(row_weights[p])
                neg_pools.append(neg_arr)
    pos_w = np.asarray(pos_w, dtype=np.float32)
    # Keep the global objective scale equal to unweighted BPR.
    if len(pos_w) and float(pos_w.mean()) > 0:
        pos_w = pos_w / float(pos_w.mean())
    return np.asarray(pos_all, dtype=np.int64), pos_w, neg_pools


def sample_negatives(neg_pools, rng, n_neg):
    neg = np.empty((len(neg_pools), n_neg), dtype=np.int64)
    for i, pool in enumerate(neg_pools):
        neg[i] = pool[rng.integers(len(pool), size=n_neg)]
    return neg


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, n_neg=3, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    row_weights, matched, n_raw = load_watch_weights(data_dir, splits['train'])
    pos_idx, pos_weights, neg_pools = make_user_pairs(ytr, utr, row_weights)
    if len(pos_idx) == 0:
        raise RuntimeError('no within-user positive/negative pairs in train split')
    if verbose:
        print(f"watch rows matched: {matched:,d}/{len(splits['train']):,d} train rows "
              f"from {n_raw:,d} raw rows")
        print(f"weighted BPR positives: {len(pos_idx):,d}; negatives/positive={n_neg}; "
              f"weight mean={pos_weights.mean():.3f} std={pos_weights.std():.3f} "
              f"min={pos_weights.min():.3f} max={pos_weights.max():.3f}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    W_t = torch.from_numpy(pos_weights.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        neg_idx = sample_negatives(neg_pools, rng, n_neg)
        order = rng.permutation(len(pos_idx))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            psel_np = pos_idx[sel]
            nsel_np = neg_idx[sel].reshape(-1)
            xp = Xtr_t[torch.from_numpy(psel_np)].to(device)
            xn = Xtr_t[torch.from_numpy(nsel_np)].to(device)
            wb = W_t[torch.from_numpy(sel)].to(device).repeat_interleave(n_neg)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).repeat_interleave(n_neg)
            sn = model(xn)
            per_pair = torch.nn.functional.softplus(-(sp - sn))
            loss = (per_pair * wb).sum() / torch.clamp(wb.sum(), min=1e-6)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | wbpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
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
    ap.add_argument('--n_neg', type=int, default=3)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, n_neg=a.n_neg, seed=a.seed,
                     device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== watch_weighted_bpr3_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
