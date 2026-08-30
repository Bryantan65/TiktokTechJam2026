"""Watch-time weighted 3-negative BPR FM seed bag with per-user z fusion.

This copies node 7's best BPR ensemble, but changes member training: each
positive/negative pair is weighted by a normalized log watch-time signal from the
raw KuaiRand log CSVs.  Cached predictions use a new name because the member
loss has changed.  If raw play_time_ms cannot be loaded/aligned, it falls back to
uniform weights so the script remains standalone.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

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
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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


def find_log_files(data_dir):
    cands = [
        data_dir,
        os.path.join(data_dir, 'data'),
        os.path.dirname(data_dir),
    ]
    names = ['log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv']
    for base in cands:
        paths = [os.path.join(base, n) for n in names]
        if all(os.path.isfile(p) for p in paths):
            return paths
    return None


def get_float(row, keys, default=0.0):
    for k in keys:
        if k in row and row[k] not in ('', 'None', 'nan'):
            try:
                return float(row[k])
            except Exception:
                pass
    return default


def load_train_watch_weights(data_dir, splits, verbose=True):
    """Return one weight per training row, aligned by preserved raw CSV order.

    data.load uses the two standard logs in order.  In KuaiRand-Pure the first
    file is the train window and the second is split into valid/test, so the
    first len(train) raw rows align to splits['train'].  A few defensive checks
    keep failures from corrupting the experiment silently.
    """
    paths = find_log_files(data_dir)
    ntr = len(splits['train'])
    if paths is None:
        if verbose:
            print('raw log CSVs not found; using uniform BPR weights')
        return np.ones(ntr, dtype=np.float32)

    vals = []
    total_needed = ntr
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                play = get_float(row, ['play_time_ms', 'play_time', 'watch_time_ms', 'time_ms'], 0.0)
                dur = get_float(row, ['duration_ms', 'video_duration', 'duration'], 0.0)
                # Use watch ratio when duration is available, but cap replays/outliers.
                if dur > 0:
                    ratio = max(0.0, min(play / dur, 5.0))
                    vals.append(np.log1p(ratio))
                else:
                    vals.append(np.log1p(max(0.0, play) / 1000.0))
                if len(vals) >= total_needed:
                    break
        if len(vals) >= total_needed:
            break

    if len(vals) < ntr:
        if verbose:
            print(f'only loaded {len(vals)} raw rows for {ntr} train rows; using uniform BPR weights')
        return np.ones(ntr, dtype=np.float32)

    w = np.asarray(vals[:ntr], dtype=np.float32)
    # Convert to a gentle confidence weight: mean 1 on positive rows, clipped so
    # one viral/replayed item cannot dominate BPR updates.
    y = np.asarray([r[6] for r in splits['train']], dtype=np.float32)
    pos = y > 0.5
    if pos.any() and np.isfinite(w[pos]).all() and w[pos].mean() > 1e-8:
        w = w / float(w[pos].mean())
        w = np.clip(w, 0.25, 3.0)
    else:
        w[:] = 1.0
    if verbose:
        print(f'watch BPR weights: mean_pos={w[pos].mean() if pos.any() else w.mean():.3f} '
              f'p10={np.percentile(w[pos] if pos.any() else w,10):.3f} '
              f'p90={np.percentile(w[pos] if pos.any() else w,90):.3f}')
    return w.astype(np.float32)


def make_pair_sampler(y, users):
    pos_by_u = defaultdict(list)
    neg_by_u = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u[uu].append(i)
        else:
            neg_by_u[uu].append(i)
    pos_idx, pos_users, neg_arrays = [], [], {}
    for uu, ps in pos_by_u.items():
        ns = neg_by_u.get(uu)
        if ns:
            neg_arrays[uu] = np.asarray(ns, dtype=np.int64)
            pos_idx.extend(ps)
            pos_users.extend([uu] * len(ps))
    return np.asarray(pos_idx, dtype=np.int64), np.asarray(pos_users, dtype=object), neg_arrays


def train_one_member(enc, dim, pair_weights, k=16, lr=0.001, l2=1e-6, epochs=50, bs=8192,
                     patience=5, seed=0, device='cpu', verbose=True, nneg=3):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    torch.manual_seed(seed)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    w_t = torch.from_numpy(pair_weights.astype(np.float32))
    pos_idx, pos_users, neg_by_u = make_pair_sampler(ytr, utr)
    if verbose:
        print(f"member seed={seed}: watch-weighted BPR positives {len(pos_idx):,d} from {len(neg_by_u):,d} mixed-label users; nneg={nneg}")
    if len(pos_idx) == 0:
        raise RuntimeError('no same-user positive/negative training pairs found')
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(pos_idx))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            pidx = pos_idx[sel]
            nidx = np.empty((len(sel), nneg), dtype=np.int64)
            for j, uu in enumerate(pos_users[sel]):
                ns = neg_by_u[uu]
                nidx[j] = ns[rng.integers(len(ns), size=nneg)]
            xp = Xtr_t[torch.from_numpy(pidx)].to(device)
            xn = Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            ww = w_t[torch.from_numpy(pidx)].to(device).repeat_interleave(nneg)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).repeat_interleave(nneg)
            sn = model(xn)
            raw = torch.nn.functional.softplus(-(sp - sn))
            loss = (raw * ww).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  member {seed} epoch {ep:2d} | wbpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  member {seed} early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def cached_member_predictions(enc, dim, pair_weights, target, split_name, k, lr, epochs, seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache', f'009_watchwbpr_member_seed{seed}_{split_name}_{target}.npy')
    if os.path.isfile(cache_path):
        if verbose:
            print(f"loading cached member predictions: {cache_path}")
        return np.load(cache_path)
    model = train_one_member(enc, dim, pair_weights, k=k, lr=lr, epochs=epochs, seed=seed,
                             device=device, verbose=verbose, nneg=3)
    X, y, users = enc[target]
    preds = model.predict(X, device=device).astype(np.float64)
    np.save(cache_path, preds)
    if verbose:
        print(f"saved cached member predictions: {cache_path}")
    return preds


def per_user_zscore(scores, users):
    scores = scores.astype(np.float64, copy=True)
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    for idxs in groups.values():
        idx = np.asarray(idxs, dtype=np.int64)
        vals = scores[idx]
        sd = vals.std()
        if sd > 1e-12:
            scores[idx] = (vals - vals.mean()) / sd
        else:
            scores[idx] = 0.0
    return scores


def run_ensemble(splits, target, split_name, data_dir, k=16, lr=0.001, epochs=50,
                 seed=0, device='cpu', verbose=True, members=3):
    enc, dim = encode(splits)
    X, y, users = enc[target]
    pair_weights = load_train_watch_weights(data_dir, splits, verbose=verbose)
    member_seeds = [seed + 1000 * m for m in range(members)]
    fused = []
    for ms in member_seeds:
        p = cached_member_predictions(enc, dim, pair_weights, target, split_name, k, lr, epochs, ms, device, verbose)
        fused.append(per_user_zscore(p, users))
    return np.mean(np.vstack(fused), axis=0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=50)
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

    scores = run_ensemble(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr,
                          epochs=a.epochs, seed=a.seed, device=a.device,
                          verbose=a.out is None, members=3)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== watchtime_weighted_bpr (seed={a.seed}, device={a.device}) ===")
        print(f"produced {len(scores):,d} predictions for split={a.split}")
