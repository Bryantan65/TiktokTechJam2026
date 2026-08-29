"""FM trained with within-user BPR plus online hard-negative sampling.

Node 2's random same-user BPR improved the baseline.  This keeps the same model
and loss but, after a short warmup, samples most negatives from the currently
highest-scored negatives for that user so the gradient focuses on mistakes that
matter for top-k ranking.
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


def make_hard_neg_lists(model, Xtr, neg_lists, hard_k=32, device='cpu'):
    scores = model.predict(Xtr, device=device)
    hard = []
    for neg in neg_lists:
        if len(neg) <= hard_k:
            hard.append(neg)
        else:
            s = scores[neg]
            # Top currently over-ranked negatives for this user.
            part = np.argpartition(s, -hard_k)[-hard_k:]
            hard.append(neg[part].astype(np.int64, copy=False))
    return hard


def sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng,
                 hard_neg_lists=None, hard_prob=0.70):
    """Sample same-user (positive row, negative row) training pairs."""
    uids = rng.choice(len(pos_lists), size=n_pairs, replace=True, p=weights)
    pos_idx = np.empty(n_pairs, dtype=np.int64)
    neg_idx = np.empty(n_pairs, dtype=np.int64)
    for u in np.unique(uids):
        m = np.nonzero(uids == u)[0]
        pos = pos_lists[u]
        pos_idx[m] = pos[rng.integers(0, len(pos), size=len(m))]
        use_hard = (hard_neg_lists is not None) and (rng.random() < hard_prob)
        neg = hard_neg_lists[u] if use_hard else neg_lists[u]
        neg_idx[m] = neg[rng.integers(0, len(neg), size=len(m))]
    return pos_idx, neg_idx


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_lists, neg_lists, weights = build_user_pair_sources(ytr, utr)
    # More pair updates than node 2, but still less than one full pass over rows.
    n_pairs = min(len(ytr), max(1, int(2 * (ytr > 0.5).sum())))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    hard_neg_lists = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        # Two random-BPR warmup epochs, then refresh hard negatives each epoch.
        if ep >= 3:
            hard_neg_lists = make_hard_neg_lists(model, Xtr, neg_lists,
                                                 hard_k=32, device=device)
        pidx, nidx = sample_pairs(pos_lists, neg_lists, weights, n_pairs, rng,
                                  hard_neg_lists=hard_neg_lists,
                                  hard_prob=0.70 if ep >= 3 else 0.0)
        order = rng.permutation(n_pairs)
        for i in range(0, n_pairs, bs):
            sel = order[i:i + bs]
            pi = torch.from_numpy(pidx[sel])
            ni = torch.from_numpy(nidx[sel])
            xb_pos = Xtr_t[pi].to(device)
            xb_neg = Xtr_t[ni].to(device)
            opt.zero_grad(set_to_none=True)
            s_pos = model(xb_pos)
            s_neg = model(xb_neg)
            loss = F.softplus(-(s_pos - s_neg)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            hn = 'hard' if ep >= 3 else 'rand'
            print(f"  epoch {ep:2d} | {hn} bpr {np.mean(losses):.4f} | valid "
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
                     device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bpr_hardneg_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
