"""FM with hashed within-user cross fields.

Node 8's hand-smoothed history features were only a weak post-hoc correction.
This version gives the FM explicit first-order memorisation slots for the same
within-user preferences (user-video, user-author, user-tab) plus author-tab, so
BCE/BPR training can learn the smoothing and interactions instead of using fixed
weights.
"""
import argparse
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

# Bucket sizes keep Adam memory bounded while giving common crosses dedicated-ish ids.
CROSS_SPECS = [
    ('uv', 600_000),   # user-video repeated exposures / pair preference
    ('ua', 400_000),   # user-author preference
    ('ut', 80_000),    # user-tab preference
    ('at', 120_000),   # author-tab context
]


def h2(a, b, mod):
    s = (str(a) + '\x1f' + str(b)).encode('utf8')
    return zlib.crc32(s) % mod


def augment_encoded(splits):
    enc0, dim0 = encode(splits)
    enc = {}
    for sp, rows in splits.items():
        X0, y, u = enc0[sp]
        H = np.empty((len(rows), len(CROSS_SPECS)), dtype=np.int64)
        offsets = []
        off = dim0
        for _, m in CROSS_SPECS:
            offsets.append(off); off += m
        for i, r in enumerate(rows):
            user, video, author, tab = r[1], r[2], r[3], r[4]
            H[i, 0] = offsets[0] + h2(user, video, CROSS_SPECS[0][1])
            H[i, 1] = offsets[1] + h2(user, author, CROSS_SPECS[1][1])
            H[i, 2] = offsets[2] + h2(user, tab, CROSS_SPECS[2][1])
            H[i, 3] = offsets[3] + h2(author, tab, CROSS_SPECS[3][1])
        enc[sp] = (np.concatenate([X0.astype(np.int64), H], axis=1), y, u)
    return enc, off


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


def build_pair_sampler(y, users):
    pos_by_user = defaultdict(list); neg_by_user = defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0.5:
            pos_by_user[u].append(i)
        else:
            neg_by_user[u].append(i)
    pos_idx, neg_pools = [], []
    for u, ps in pos_by_user.items():
        ns = neg_by_user.get(u)
        if not ns:
            continue
        arr = np.asarray(ns, dtype=np.int64)
        for p in ps:
            pos_idx.append(p); neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def train_bce(splits, seed=0, k=16, lr=0.001, l2=3e-6, epochs=40, bs=8192,
              patience=4, device='cpu'):
    enc, dim = augment_encoded(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        model.train()
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device); yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc


def train_bpr(splits, seed=0, k=16, lr=0.001, l2=3e-6, epochs=40, bs=8192,
              patience=4, bce_weight=0.02, device='cpu'):
    enc, dim = augment_encoded(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    pos_idx, neg_pools = build_pair_sampler(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_pairs = max(len(ytr), len(pos_idx))
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        order = rng.integers(0, len(pos_idx), size=n_pairs, dtype=np.int64)
        model.train()
        for i in range(0, len(order), bs):
            which = order[i:i + bs]
            p_np = pos_idx[which]
            n_np = np.empty(len(which), dtype=np.int64)
            for j, w in enumerate(which):
                pool = neg_pools[int(w)]
                n_np[j] = pool[rng.integers(0, len(pool))]
            pair_np = np.concatenate([p_np, n_np])
            xb = Xtr_t[torch.from_numpy(pair_np)].to(device)
            logits = model(xb)
            sp, sn = logits[:len(p_np)], logits[len(p_np):]
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            if bce_weight > 0:
                yb = ytr_t[torch.from_numpy(pair_np)].to(device)
                loss = loss + bce_weight * bce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc


def z_by_user(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(scores)
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    for idxs in groups.values():
        vals = scores[idxs]
        sd = vals.std()
        out[idxs] = (vals - vals.mean()) / sd if sd > 1e-8 else 0.0
    return out


def cached_member(name, split_name, target, seed, train_fn, splits, device):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'009_{name}_{split_name}_seed{seed}.npy')
    if os.path.isfile(path):
        return np.load(path), None
    model, enc = train_fn(splits, seed=seed, device=device)
    X, _, users = enc[target]
    preds = model.predict(X, device=device).astype(np.float64)
    np.save(path, preds)
    return preds, users


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
        splits = load_dev(a.data_dir); target = 'valid'; split_name = 'dev'
    else:
        splits = load(a.data_dir); target = a.split; split_name = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}+hashed_crosses")
    t0 = time.time()
    bce_pred, users = cached_member('cross_bce', split_name, target, a.seed, train_bce, splits, a.device)
    bpr_pred, users2 = cached_member('cross_bpr02', split_name, target, a.seed, train_bpr, splits, a.device)
    if users is None:
        enc_tmp, _ = augment_encoded(splits)
        users = enc_tmp[target][2]
    # Equal readable blend: pointwise cross biases help nDCG, BPR keeps GAUC aligned.
    scores = 0.50 * z_by_user(bce_pred, users) + 0.50 * z_by_user(bpr_pred, users)
    print(f"built cross-FM blend in {time.time() - t0:.1f}s")
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        _, y, u = augment_encoded(splits)[0][target]
        print(evaluate(u, y, scores))
