"""Blend the best node-13 ensemble with a multi-task FM member.

The incumbent is the cached node-13 BCE/BPR/rank five-seed ensemble.  This adds
one FM trained on the main label plus raw KuaiRand feedback auxiliaries
(is_click/is_like/is_follow/is_comment/is_forward).  Auxiliary heads have
separate linear terms but share the FM interaction embeddings, so engagement
signals can regularize item/user cross factors without changing the prediction
contract.  The new member is blended at a readable 30% after per-user z/rank
normalization.
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
from evaluate import evaluate                  # noqa: E402  early stopping only


AUX_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']


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


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_tasks=6, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros((n_tasks, dim), dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))

    def forward_all(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        # W[:, X] -> tasks x batch x fields, then transpose to batch x tasks.
        lin = self.W[:, X].sum(2).transpose(0, 1)
        return lin + self.b.unsqueeze(0) + inter.unsqueeze(1)

    def forward(self, X):
        return self.forward_all(X)[:, 0]

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def fit_bce(splits, enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); ytr_t = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time(); model.train(); losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device); yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BCE(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def make_user_pair_sources(y, users):
    pos = defaultdict(list); neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        (pos if yy > 0.5 else neg)[uu].append(i)
    pos_idx = []; neg_pools = []
    for uu, ps in pos.items():
        ns = neg.get(uu)
        if ns:
            arr = np.asarray(ns, dtype=np.int64)
            for p in ps:
                pos_idx.append(p); neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def sample_uniform_pairs(pos_idx, neg_pools, rng, neg_per_pos=3):
    total = len(pos_idx) * neg_per_pos
    p_out = np.empty(total, dtype=np.int64); n_out = np.empty(total, dtype=np.int64)
    k = 0
    for p, pool in zip(pos_idx, neg_pools):
        m = len(pool)
        for _ in range(neg_per_pos):
            p_out[k] = p; n_out[k] = pool[rng.integers(0, m)]; k += 1
    order = rng.permutation(total)
    return p_out[order], n_out[order]


def fit_bpr(splits, enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, neg_per_pos=3, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, neg_pools = make_user_pair_sources(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); p_idx, n_idx = sample_uniform_pairs(pos_idx, neg_pools, rng, neg_per_pos=neg_per_pos)
        model.train(); losses = []
        for i in range(0, len(p_idx), bs):
            ps = torch.from_numpy(p_idx[i:i + bs]); ns = torch.from_numpy(n_idx[i:i + bs])
            xp = Xtr_t[ps].to(device); xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp) - model(xn))).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BPR(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def read_raw_aux(data_dir):
    by_key = defaultdict(deque)
    paths = [
        os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
        os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv'),
    ]
    n = 0
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                key = (str(r.get('date', '')), str(r.get('user_id', '')),
                       str(r.get('video_id', '')), str(r.get('tab', '')))
                vals = []
                for c in AUX_COLS:
                    try:
                        vals.append(float(r.get(c, 0) or 0))
                    except Exception:
                        vals.append(0.0)
                by_key[key].append(np.asarray(vals, dtype=np.float32))
                n += 1
    return by_key, n


def align_aux_to_splits(splits, data_dir):
    queues, nraw = read_raw_aux(data_dir)
    out = {}
    hits = 0; total = 0
    # Consume in split order. The date is in the key, so train/valid/test do not
    # steal each other's rows; occurrence order handles repeated impressions.
    for name in ['train', 'valid', 'test']:
        if name not in splits:
            continue
        arr = np.zeros((len(splits[name]), len(AUX_COLS)), dtype=np.float32)
        for i, row in enumerate(splits[name]):
            key = (str(row[0]), str(row[1]), str(row[2]), str(row[4]))
            q = queues.get(key)
            if q:
                arr[i] = q.popleft(); hits += 1
            total += 1
        out[name] = arr
    print(f"raw aux rows {nraw} aligned {hits}/{total} using cols {AUX_COLS}")
    return out


def fit_multitask(splits, enc, dim, aux_train, k=16, lr=0.001, l2=1e-6,
                  epochs=40, bs=8192, patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = MultiTaskFM(dim, n_tasks=1 + len(AUX_COLS), k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    aux_t = torch.from_numpy(aux_train.astype(np.float32))
    rng = np.random.default_rng(seed + 17)
    # Click is common and informative; rarer social actions are useful but noisy.
    aux_w = torch.tensor([0.20, 0.08, 0.05, 0.05, 0.05], dtype=torch.float32, device=device)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time(); model.train(); losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device); yb = ytr_t[sel].to(device); ab = aux_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model.forward_all(xb)
            main_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], yb)
            aux_loss_each = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1:], ab, reduction='none').mean(0)
            loss = main_loss + (aux_w * aux_loss_each).sum()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  MTL(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def user_groups(users):
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def per_user_zscore(scores, groups):
    scores = scores.astype(np.float64, copy=False)
    out = np.empty_like(scores, dtype=np.float64)
    for idx in groups:
        s = scores[idx]; sd = s.std()
        out[idx] = (s - s.mean()) / sd if sd > 1e-12 else s - s.mean()
    return out


def per_user_rank01(scores, groups):
    scores = scores.astype(np.float64, copy=False)
    out = np.empty_like(scores, dtype=np.float64)
    for idx in groups:
        n = len(idx)
        if n <= 1:
            out[idx] = 0.0; continue
        order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64) / (n - 1.0)
        out[idx] = ranks
    return out


def cached_predict(cache_name, train_fn, enc, target, member_seed, device, use_cache=True):
    os.makedirs('pred_cache', exist_ok=True)
    X, _, _ = enc[target]
    cache_path = os.path.join('pred_cache', f'{cache_name}_{target}_seed{member_seed}.npy')
    if use_cache and os.path.isfile(cache_path):
        preds = np.load(cache_path)
        if len(preds) == len(X):
            return preds.astype(np.float64, copy=False)
    model = train_fn()
    preds = model.predict(X, device=device).astype(np.float64)
    if use_cache:
        np.save(cache_path, preds)
    return preds


def cached_predict_013(member_name, train_fn, enc, target, member_seed, device, use_cache=True):
    return cached_predict(f'006_{member_name}', train_fn, enc, target, member_seed, device, use_cache)


def node8_seed_score(bce_preds, bpr_preds, groups):
    zblend = 0.35 * per_user_zscore(bce_preds, groups) + 0.65 * per_user_zscore(bpr_preds, groups)
    rblend = 0.35 * per_user_rank01(bce_preds, groups) + 0.65 * per_user_rank01(bpr_preds, groups)
    return 0.70 * zblend + 0.30 * rblend


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--neg_per_pos', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({kk: len(vv) for kk, vv in splits.items()}, f"fields={FIELDS}")

    enc, dim = encode(splits)
    verbose = a.out is None
    use_cache = a.out is not None and a.split != 'dev'
    X, y, users = enc[target]
    groups = user_groups(users)

    # Exact node-13 base: unchanged cached node-8 members with seed weights.
    member_seeds = [0, 1, 2, 3, 4]
    weights = np.asarray([0.12, 0.12, 0.12, 0.32, 0.32], dtype=np.float64)
    seed_scores = []
    for ms in member_seeds:
        bce_preds = cached_predict_013(
            'bce',
            lambda ms=ms: fit_bce(splits, enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                  seed=ms, device=a.device, verbose=verbose),
            enc, target, ms, a.device, use_cache=use_cache)
        bpr_preds = cached_predict_013(
            'bpr_uniform_np3',
            lambda ms=ms: fit_bpr(splits, enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                  neg_per_pos=a.neg_per_pos, seed=ms, device=a.device,
                                  verbose=verbose),
            enc, target, ms, a.device, use_cache=use_cache)
        seed_scores.append(per_user_zscore(node8_seed_score(bce_preds, bpr_preds, groups), groups))
    base = weights @ np.vstack(seed_scores)
    cur_idx = member_seeds.index(a.seed) if a.seed in member_seeds else (a.seed % len(member_seeds))
    base = 0.995 * base + 0.005 * seed_scores[cur_idx]

    aux = align_aux_to_splits(splits, a.data_dir)
    mtl_preds = cached_predict(
        '022_multitask_fm_aux',
        lambda: fit_multitask(splits, enc, dim, aux['train'], k=a.k, lr=a.lr,
                              epochs=a.epochs, seed=a.seed, device=a.device,
                              verbose=verbose),
        enc, target, a.seed, a.device, use_cache=use_cache)
    mtl_score = 0.70 * per_user_zscore(mtl_preds, groups) + 0.30 * per_user_rank01(mtl_preds, groups)

    scores = 0.70 * per_user_zscore(base, groups) + 0.30 * per_user_zscore(mtl_score, groups)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== multitask_blend (seed={a.seed}, device={a.device}) ===")
        r = evaluate(users, y, scores)
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
