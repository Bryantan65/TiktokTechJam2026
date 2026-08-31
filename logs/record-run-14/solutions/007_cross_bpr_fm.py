"""BPR FM with train-observed user interaction cross fields.

Node 6 made negatives harder but the raw play_time join failed, and the harder
pairs alone mostly traded nDCG for GAUC.  This keeps the within-user BPR setup
but adds direct categorical memory for train-observed user-video, user-author and
user-tab preferences.  Unknown crosses at prediction time map to zero-effect OOV
features, so they do not inject random noise.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0, zero_indices=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        W0 = np.zeros(dim, dtype=np.float32)
        if zero_indices is not None and len(zero_indices):
            V0[np.asarray(zero_indices, dtype=np.int64)] = 0.0
            W0[np.asarray(zero_indices, dtype=np.int64)] = 0.0
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.from_numpy(W0))
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


def cross_values(row):
    # tuple: (date, user_id, video_id, author_id, tab, duration_ms, label)
    u = str(row[1]); v = str(row[2]); a = str(row[3]); t = str(row[4])
    return ((u, v), (u, a), (u, t), (a, t))


def add_cross_features(splits):
    """Append cross-field ids to starter-kit encoded base features.

    Cross vocabularies are fit on train only.  Values not seen during training
    map to a per-field OOV id whose embedding is fixed initially at zero and is
    never selected in training, so unseen crosses are neutral.
    """
    enc, base_dim = encode(splits)
    maps = [dict() for _ in range(4)]
    offsets = []
    next_id = base_dim
    for j in range(4):
        offsets.append(next_id)
        vals = set()
        for r in splits['train']:
            vals.add(cross_values(r)[j])
        # deterministic order independent of hash seed
        mp = {val: offsets[j] + i for i, val in enumerate(sorted(vals))}
        maps[j] = mp
        next_id = offsets[j] + len(mp) + 1  # final id is OOV
    oov_ids = [offsets[j] + len(maps[j]) for j in range(4)]

    out = {}
    for sp, rows in splits.items():
        X, y, users = enc[sp]
        C = np.empty((len(rows), 4), dtype=np.int64)
        for i, r in enumerate(rows):
            cvs = cross_values(r)
            for j in range(4):
                C[i, j] = maps[j].get(cvs[j], oov_ids[j])
        out[sp] = (np.concatenate([X.astype(np.int64), C], axis=1), y, users)
    return out, next_id, oov_ids


def build_pair_sampler(y, users, tabs):
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
    for u, ps in pos_by_user.items():
        ns = neg_by_user.get(u)
        if not ns:
            continue
        ns_arr = np.asarray(ns, dtype=np.int64)
        for p in ps:
            pos_idx.append(p)
            neg_pools.append(ns_arr)
            hp = neg_by_user_tab.get((u, tabs[p]))
            hard_pools.append(np.asarray(hp, dtype=np.int64) if hp else ns_arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools, hard_pools


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        bce_weight=0.05, hard_prob=0.30, seed=0, device='cpu', verbose=True):
    enc, dim, zero_ids = add_cross_features(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed, zero_indices=zero_ids).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    # Original starter fields are first; tab is column 3.
    pos_idx, neg_pools, hard_pools = build_pair_sampler(ytr, utr, Xtr[:, 3])
    if len(pos_idx) == 0:
        raise RuntimeError('no within-user positive/negative pairs available')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_pairs_per_epoch = max(len(ytr), len(pos_idx))
    bce_loss = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        order = rng.integers(0, len(pos_idx), size=n_pairs_per_epoch, dtype=np.int64)
        hard_draws = rng.random(len(order)) < hard_prob
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), bs):
            which = order[i:i + bs]
            use_hard = hard_draws[i:i + bs]
            p_np = pos_idx[which]
            n_np = np.empty(len(which), dtype=np.int64)
            for j, w in enumerate(which):
                pool = hard_pools[int(w)] if use_hard[j] else neg_pools[int(w)]
                n_np[j] = pool[rng.integers(0, len(pool))]

            pair_np = np.concatenate([p_np, n_np])
            xb = Xtr_t[torch.from_numpy(pair_np)].to(device)
            logits = model(xb)
            sp, sn = logits[:len(p_np)], logits[len(p_np):]
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            if bce_weight > 0:
                yb = ytr_t[torch.from_numpy(pair_np)].to(device)
                loss = loss + bce_weight * bce_loss(logits, yb)

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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+crosses")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== cross_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
