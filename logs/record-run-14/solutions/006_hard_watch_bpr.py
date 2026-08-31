"""BPR FM with harder within-user negatives and watch-time pair weights.

This refines node 2 rather than changing model capacity.  The parent sampled a
random negative from the same user for each positive.  Here most negatives are
sampled from the same user *and tab* when possible, making pairs less trivial,
and the pair loss is moderately weighted by the positive row's raw play_time_ms
(as an auxiliary training signal only, never as a prediction feature).
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
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


def _s(x):
    return str(x)


def _key_from_row_tuple(r):
    # Do not include duration: raw CSVs sometimes stringify integer ms
    # differently from the starter-kit tuple, and (date,user,video,author,tab)
    # is enough when consumed as a queue in file order.
    return (_s(r[0]), _s(r[1]), _s(r[2]), _s(r[3]), _s(r[4]))


def _find_log_file(data_dir, prefix):
    if not os.path.isdir(data_dir):
        return None
    for fn in os.listdir(data_dir):
        if fn.startswith(prefix) and fn.endswith('.csv'):
            return os.path.join(data_dir, fn)
    return None


def load_play_time_for_rows(data_dir, rows):
    """Return play_time_ms aligned to rows; missing values get 0.

    Only play_time_ms is read from raw logs and is used only to weight training
    pairs.  The raw long_view column is deliberately ignored.
    """
    files = []
    for pref in ('log_standard_4_08_to_4_21', 'log_standard_4_22_to_5_08'):
        p = _find_log_file(data_dir, pref)
        if p is not None:
            files.append(p)
    if not files:
        return np.zeros(len(rows), dtype=np.float32)

    need = defaultdict(int)
    for r in rows:
        need[_key_from_row_tuple(r)] += 1

    queues = defaultdict(deque)
    for path in files:
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                for rec in reader:
                    k = (rec.get('date', ''), rec.get('user_id', ''),
                         rec.get('video_id', ''), rec.get('author_id', ''),
                         rec.get('tab', ''))
                    if need.get(k, 0) > len(queues[k]):
                        try:
                            pt = float(rec.get('play_time_ms', 0.0) or 0.0)
                        except Exception:
                            pt = 0.0
                        queues[k].append(pt)
        except OSError:
            pass

    out = np.zeros(len(rows), dtype=np.float32)
    miss = 0
    for i, r in enumerate(rows):
        k = _key_from_row_tuple(r)
        if queues.get(k) and len(queues[k]) > 0:
            out[i] = queues[k].popleft()
        else:
            miss += 1
    if miss:
        print(f"play_time_ms missing for {miss}/{len(rows)} train rows; using weight 1 there")
    return out


def play_to_pair_weights(play_ms, y):
    """Moderate positive-row weights with mean approximately 1.

    We use log play time as a strength signal, but clip it so this remains a
    ranking-weight tweak rather than reconstructing the label rule.
    """
    y = np.asarray(y)
    play_ms = np.asarray(play_ms, dtype=np.float32)
    w = np.ones(len(y), dtype=np.float32)
    pos = y > 0.5
    if pos.sum() == 0:
        return w
    strength = np.log1p(np.maximum(play_ms[pos], 0.0)).astype(np.float32)
    good = strength > 0
    if good.sum() < 10:
        return w
    med = float(np.median(strength[good]))
    if med <= 0:
        return w
    wp = np.sqrt(np.maximum(strength, med * 0.25) / med)
    wp = np.clip(wp, 0.65, 2.25)
    wp = wp / max(float(wp.mean()), 1e-6)
    w[pos] = wp.astype(np.float32)
    return w


def build_pair_sampler(y, users, tabs, pair_weights):
    """Return positive indices plus easy and hard negative pools."""
    y = np.asarray(y)
    users = np.asarray(users)
    tabs = np.asarray(tabs)

    pos_by_user = defaultdict(list)
    neg_by_user = defaultdict(list)
    neg_by_user_tab = defaultdict(list)
    for i, (u, t, yy) in enumerate(zip(users, tabs, y)):
        if yy > 0.5:
            pos_by_user[u].append(i)
        else:
            neg_by_user[u].append(i)
            neg_by_user_tab[(u, t)].append(i)

    pos_idx = []
    neg_pools = []
    hard_pools = []
    pos_w = []
    for u, ps in pos_by_user.items():
        ns = neg_by_user.get(u)
        if not ns:
            continue
        ns_arr = np.asarray(ns, dtype=np.int64)
        for p in ps:
            pos_idx.append(p)
            neg_pools.append(ns_arr)
            hp = neg_by_user_tab.get((u, tabs[p]))
            if hp:
                hard_pools.append(np.asarray(hp, dtype=np.int64))
            else:
                hard_pools.append(ns_arr)
            pos_w.append(pair_weights[p])
    return (np.asarray(pos_idx, dtype=np.int64), neg_pools, hard_pools,
            np.asarray(pos_w, dtype=np.float32))


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        bce_weight=0.05, hard_prob=0.60, seed=0, device='cpu', verbose=True,
        data_dir=None):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    play_ms = load_play_time_for_rows(data_dir, splits['train']) if data_dir else np.zeros(len(ytr), dtype=np.float32)
    row_weights = play_to_pair_weights(play_ms, ytr)

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    w_t = torch.from_numpy(row_weights.astype(np.float32))

    # Field order is [user_id, video_id, author_id, tab, dur_bucket].
    pos_idx, neg_pools, hard_pools, pos_w_np = build_pair_sampler(
        ytr, utr, Xtr[:, 3], row_weights)
    if len(pos_idx) == 0:
        raise RuntimeError('no within-user positive/negative pairs available')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_pairs_per_epoch = max(len(ytr), len(pos_idx))
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    for ep in range(1, epochs + 1):
        order = rng.integers(0, len(pos_idx), size=n_pairs_per_epoch, dtype=np.int64)
        t0 = time.time()
        model.train()
        losses = []
        hard_draws = rng.random(len(order)) < hard_prob
        for i in range(0, len(order), bs):
            which = order[i:i + bs]
            use_hard = hard_draws[i:i + bs]
            p_np = pos_idx[which]
            n_np = np.empty(len(which), dtype=np.int64)
            for j, widx in enumerate(which):
                pool = hard_pools[int(widx)] if use_hard[j] else neg_pools[int(widx)]
                n_np[j] = pool[rng.integers(0, len(pool))]

            pair_np = np.concatenate([p_np, n_np])
            xb = Xtr_t[torch.from_numpy(pair_np)].to(device)
            logits = model(xb)
            sp, sn = logits[:len(p_np)], logits[len(p_np):]
            pw = torch.from_numpy(pos_w_np[which]).to(device)
            pair_loss = torch.nn.functional.softplus(-(sp - sn))
            loss = (pair_loss * pw).sum() / pw.sum().clamp_min(1e-6)

            if bce_weight > 0:
                idx_t = torch.from_numpy(pair_np)
                yb = ytr_t[idx_t].to(device)
                wb = w_t[idx_t].to(device)
                bce = bce_loss(logits, yb)
                loss = loss + bce_weight * (bce * wb).sum() / wb.sum().clamp_min(1e-6)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
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

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None, data_dir=a.data_dir)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== hard_watch_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
