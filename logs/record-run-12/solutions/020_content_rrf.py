"""Content-feature FM ensemble with reciprocal-rank fusion.

Improves node 17 by changing only the cached-member fusion rule.  The member
training code and cache names are identical to 018_content_features_rank.py, so
cached runs are cheap, while a fresh cache can still train from scratch.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

CONTENT_FIELDS = ['music_id', 'tag', 'video_type', 'upload_type', 'server_width', 'server_height', 'music_type']


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
        self.eval(); out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def norm_val(v):
    if v is None or v == '':
        return '__MISS__'
    s = str(v).strip()
    return s if s else '__MISS__'


def load_video_content(data_dir):
    path = os.path.join(data_dir, 'video_features_basic_pure.csv')
    mp = {}
    if not os.path.isfile(path):
        print('WARNING: no video_features_basic_pure.csv found; all content missing')
        return mp
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        vid_col = 'video_id' if 'video_id' in (rdr.fieldnames or []) else ('item_id' if 'item_id' in (rdr.fieldnames or []) else None)
        if vid_col is None:
            print('WARNING: video feature file has no video_id/item_id; all content missing')
            return mp
        for rec in rdr:
            vid = norm_val(rec.get(vid_col))
            mp[vid] = tuple(norm_val(rec.get(c)) for c in CONTENT_FIELDS)
    print('loaded video content rows', len(mp))
    return mp


def append_content_features(splits, enc, base_dim, data_dir):
    content = load_video_content(data_dir)
    rows_all = []
    for sp in ['train', 'valid', 'test']:
        if sp in splits:
            rows_all.extend(splits[sp])
    maps = [{ '__MISS__': 0 } for _ in CONTENT_FIELDS]
    for r in rows_all:
        vals = content.get(norm_val(r[2]), ('__MISS__',) * len(CONTENT_FIELDS))
        for j, v in enumerate(vals):
            if v not in maps[j]:
                maps[j][v] = len(maps[j])
    offsets = []
    off = base_dim
    for m in maps:
        offsets.append(off); off += len(m)
    print('content cardinalities', {CONTENT_FIELDS[i]: len(maps[i]) for i in range(len(CONTENT_FIELDS))}, 'new_dim', off)
    out = {}
    for sp, (X, y, u) in enc.items():
        C = np.empty((len(X), len(CONTENT_FIELDS)), dtype=np.int64)
        for i, r in enumerate(splits[sp]):
            vals = content.get(norm_val(r[2]), ('__MISS__',) * len(CONTENT_FIELDS))
            for j, v in enumerate(vals):
                C[i, j] = offsets[j] + maps[j].get(v, 0)
        out[sp] = (np.concatenate([X.astype(np.int64), C], axis=1), y, u)
    return out, off


def train_bce_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    X_t = torch.from_numpy(Xtr.astype(np.int64)); y_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); losses = []; t0 = time.time(); perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = torch.from_numpy(perm[i:i + bs])
            xb = X_t[idx].to(device); yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  bce seed {seed} epoch {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def make_positive_index_pairs(y, users):
    y = np.asarray(y); users = np.asarray(users)
    order = np.argsort(users, kind='mergesort'); us = users[order]
    pos_indices, pos_gids, neg_by_gid = [], [], []
    start = 0; gid = 0; n = len(order)
    while start < n:
        end = start + 1
        while end < n and us[end] == us[start]:
            end += 1
        idx = order[start:end]; yy = y[idx]
        pos = idx[yy > 0.5]; neg = idx[yy <= 0.5]
        if len(pos) and len(neg):
            pos_indices.append(pos.astype(np.int64))
            pos_gids.append(np.full(len(pos), gid, dtype=np.int32))
            neg_by_gid.append(neg.astype(np.int64)); gid += 1
        start = end
    if not pos_indices:
        raise RuntimeError('No users with both positives and negatives')
    return np.concatenate(pos_indices), np.concatenate(pos_gids), neg_by_gid


def sample_negatives_for_batch(gids, neg_by_gid, rng):
    neg = np.empty(len(gids), dtype=np.int64)
    for g in np.unique(gids):
        m = (gids == g); pool = neg_by_gid[int(g)]
        neg[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
    return neg


def train_bpr_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5, repeats=2, bce_weight=0.10, device='cpu', verbose=False):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2}, {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_base, pos_gid_base, neg_by_gid = make_positive_index_pairs(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); losses = []; t0 = time.time()
        for _ in range(repeats):
            perm = rng.permutation(len(pos_base))
            for i in range(0, len(perm), bs):
                psel = perm[i:i + bs]
                pos_idx = pos_base[psel]
                neg_idx = sample_negatives_for_batch(pos_gid_base[psel], neg_by_gid, rng)
                xb_pos = Xtr_t[torch.from_numpy(pos_idx)].to(device)
                xb_neg = Xtr_t[torch.from_numpy(neg_idx)].to(device)
                xb = torch.cat([xb_pos, xb_neg], dim=0)
                opt.zero_grad(set_to_none=True)
                logits = model(xb); m = len(pos_idx)
                loss = F.softplus(-(logits[:m] - logits[m:])).mean()
                if bce_weight > 0:
                    labels = torch.cat([torch.ones(m, device=device), torch.zeros(m, device=device)])
                    loss = loss + bce_weight * F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  bpr seed {seed} epoch {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def rrf_by_user(scores, users, k=5.0):
    scores = np.asarray(scores, dtype=np.float64); users = np.asarray(users)
    out = np.empty_like(scores, dtype=np.float64)
    order = np.argsort(users, kind='mergesort'); us = users[order]
    start = 0; n = len(order)
    while start < n:
        end = start + 1
        while end < n and us[end] == us[start]:
            end += 1
        idx = order[start:end]
        # rank 1 is the highest score; stable tie handling keeps deterministic output.
        ord2 = idx[np.argsort(-scores[idx], kind='mergesort')]
        ranks = np.arange(1, len(ord2) + 1, dtype=np.float64)
        out[ord2] = 1.0 / (k + ranks)
        start = end
    return out


def get_member_preds(name, train_fn, enc, dim, Xtar, seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True)
    cache_path = os.path.join('pred_cache', f'018_{name}_seed{seed}_n{len(Xtar)}.npy')
    if os.path.isfile(cache_path):
        print(f'loading cached {name} seed {seed} predictions {cache_path}')
        return np.load(cache_path).astype(np.float64)
    print(f'training {name} seed {seed} member')
    model = train_fn(enc, dim, seed=seed, device=device, verbose=verbose)
    preds = model.predict(Xtar, device=device).astype(np.float64)
    np.save(cache_path, preds)
    return preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(0)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+{CONTENT_FIELDS}")
    enc0, dim0 = encode(splits)
    enc, dim = append_content_features(splits, enc0, dim0, a.data_dir)
    Xtar, _, utar = enc[target]
    verbose = (a.out is None)

    blended = []
    for member_seed in (0, 1, 2):
        bpr = get_member_preds('content_bpr_anchor_v1', train_bpr_member, enc, dim, Xtar, member_seed, a.device, verbose)
        bce = get_member_preds('content_bce_v1', train_bce_member, enc, dim, Xtar, member_seed, a.device, verbose)
        blended.append(0.70 * rrf_by_user(bpr, utar, k=5.0) + 0.30 * rrf_by_user(bce, utar, k=5.0))
    scores = np.mean(blended, axis=0)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print('done')
