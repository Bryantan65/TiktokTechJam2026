"""Score-level blend of pointwise FM and pairwise-BPR FM.

Node 3 mixed BCE and BPR gradients into one model and regressed.  This keeps the
objectives separate, early-stops each model normally, then combines only their
within-user ranks so the stronger BPR model can keep its ordering while BCE can
vote on cases where the pairwise sampler is noisy.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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


def train_bce(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BCE epoch {ep:2d} | loss {np.mean(losses):.4f} | "
                  f"valid primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def build_user_pair_sources(y, users):
    buckets = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if u not in buckets:
            buckets[u] = [[], []]
        buckets[u][1 if yy > 0.5 else 0].append(i)
    pos_lists, neg_lists, weights = [], [], []
    for neg, pos in buckets.values():
        if pos and neg:
            pos_a = np.asarray(pos, dtype=np.int64)
            neg_a = np.asarray(neg, dtype=np.int64)
            pos_lists.append(pos_a)
            neg_lists.append(neg_a)
            weights.append(min(len(pos_a) * len(neg_a), 2000))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return pos_lists, neg_lists, weights


def sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng):
    uids = rng.choice(len(pos_lists), size=n_pairs, replace=True, p=weights)
    pos_idx = np.empty(n_pairs, dtype=np.int64)
    neg_idx = np.empty(n_pairs, dtype=np.int64)
    for u in np.unique(uids):
        m = np.nonzero(uids == u)[0]
        pos = pos_lists[u]
        neg = neg_lists[u]
        pos_idx[m] = pos[rng.integers(0, len(pos), size=len(m))]
        neg_idx[m] = neg[rng.integers(0, len(neg), size=len(m))]
    return pos_idx, neg_idx


def train_bpr(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_lists, neg_lists, weights = build_user_pair_sources(ytr, utr)
    n_pairs = max(1, int((ytr > 0.5).sum()))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        pidx, nidx = sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng)
        order = rng.permutation(n_pairs)
        for i in range(0, n_pairs, bs):
            sel = order[i:i + bs]
            pi = torch.from_numpy(pidx[sel])
            ni = torch.from_numpy(nidx[sel])
            xb_pos = Xtr_t[pi].to(device)
            xb_neg = Xtr_t[ni].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.softplus(-(model(xb_pos) - model(xb_neg))).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BPR epoch {ep:2d} | loss {np.mean(losses):.4f} | "
                  f"valid primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def weighted_within_user_rank(scores_a, scores_b, users, wa=0.70):
    """Return a label-free weighted rank blend; each input is monotone-invariant."""
    users = np.asarray(users)
    out = np.zeros(len(users), dtype=np.float64)
    order_u = np.argsort(users, kind='mergesort')
    start = 0
    while start < len(users):
        end = start + 1
        u = users[order_u[start]]
        while end < len(users) and users[order_u[end]] == u:
            end += 1
        idx = order_u[start:end]
        n = len(idx)
        if n == 1:
            out[idx[0]] = 0.0
        else:
            # Ordinal ranks are enough because only final ordering is scored.
            ra = np.empty(n, dtype=np.float64)
            rb = np.empty(n, dtype=np.float64)
            ra[np.argsort(scores_a[idx], kind='mergesort')] = np.linspace(0.0, 1.0, n)
            rb[np.argsort(scores_b[idx], kind='mergesort')] = np.linspace(0.0, 1.0, n)
            out[idx] = wa * ra + (1.0 - wa) * rb
        start = end
    return out


def run(splits, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    # Deliberately separate objectives.  Different initial seeds add a little
    # diversity without ignoring the harness seed.
    bpr = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                    device=device, verbose=verbose)
    bce = train_bce(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed + 1009,
                    device=device, verbose=verbose)
    return bpr, bce, enc


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

    bpr, bce, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                        device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    s_bpr = bpr.predict(X, device=a.device)
    s_bce = bce.predict(X, device=a.device)
    scores = weighted_within_user_rank(s_bpr, s_bce, users, wa=0.70)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        r = evaluate(users, y, scores)
        print(f"\n=== bce_bpr_rankblend (seed={a.seed}, device={a.device}) ===")
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
