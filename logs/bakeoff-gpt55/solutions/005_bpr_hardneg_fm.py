"""FM trained with within-user BPR loss using sampled hard negatives.

Node 3's random 3-negative BPR was the best result. This keeps the same model
and number of trained pairs, but after the first epoch each negative is selected
as the highest-scoring item from a small same-user candidate set, concentrating
updates on mistakes that can affect top-of-user rankings.
"""
import argparse
import os
import sys
import time

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


@torch.no_grad()
def score_indices(model, Xtr_t, idx_np, device='cpu', bs=200_000):
    model.eval()
    out = np.empty(len(idx_np), dtype=np.float32)
    for i in range(0, len(idx_np), bs):
        j = min(i + bs, len(idx_np))
        xb = Xtr_t[torch.from_numpy(idx_np[i:j])].to(device)
        out[i:j] = model(xb).cpu().numpy().astype(np.float32)
    return out


def make_user_pairs(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    groups = []
    n_pos = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
            n_pos += len(pos)
    return groups, n_pos


def sample_epoch_pairs_random(groups, n_pos, negs_per_pos, rng):
    n_pairs = n_pos * negs_per_pos
    pos_all = np.empty(n_pairs, dtype=np.int64)
    neg_all = np.empty(n_pairs, dtype=np.int64)
    p = 0
    for pos, neg in groups:
        m = len(pos)
        mm = m * negs_per_pos
        pos_all[p:p + mm] = np.repeat(pos, negs_per_pos)
        neg_all[p:p + mm] = neg[rng.integers(0, len(neg), size=mm)]
        p += mm
    perm = rng.permutation(n_pairs)
    return pos_all[perm], neg_all[perm]


def sample_epoch_pairs_hard(groups, n_pos, negs_per_pos, cand_per_neg,
                            rng, model, Xtr_t, device):
    n_pairs = n_pos * negs_per_pos
    pos_all = np.empty(n_pairs, dtype=np.int64)
    cand_all = np.empty(n_pairs * cand_per_neg, dtype=np.int64)
    p = 0
    c = 0
    for pos, neg in groups:
        m = len(pos)
        mm = m * negs_per_pos
        pos_all[p:p + mm] = np.repeat(pos, negs_per_pos)
        cand_all[c:c + mm * cand_per_neg] = neg[rng.integers(0, len(neg), size=mm * cand_per_neg)]
        p += mm
        c += mm * cand_per_neg

    cand_scores = score_indices(model, Xtr_t, cand_all, device=device)
    cand_scores = cand_scores.reshape(n_pairs, cand_per_neg)
    cand_all_2d = cand_all.reshape(n_pairs, cand_per_neg)
    neg_all = cand_all_2d[np.arange(n_pairs), cand_scores.argmax(axis=1)]

    perm = rng.permutation(n_pairs)
    return pos_all[perm], neg_all[perm]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        negs_per_pos=3, cand_per_neg=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups, n_pos = make_user_pairs(ytr, utr)
    if verbose:
        print(f"hard-bpr users={len(groups):,d} positives={n_pos:,d} "
              f"epoch_pairs={n_pos * negs_per_pos:,d} cand_per_neg={cand_per_neg}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if ep == 1:
            pos_idx, neg_idx = sample_epoch_pairs_random(groups, n_pos, negs_per_pos, rng)
            mode = 'rand'
        else:
            pos_idx, neg_idx = sample_epoch_pairs_hard(
                groups, n_pos, negs_per_pos, cand_per_neg, rng, model, Xtr_t, device)
            mode = 'hard'
        n_pairs = len(pos_idx)
        model.train()
        losses = []
        for i in range(0, n_pairs, bs):
            ip = torch.from_numpy(pos_idx[i:i + bs])
            ineg = torch.from_numpy(neg_idx[i:i + bs])
            xp = Xtr_t[ip].to(device)
            xn = Xtr_t[ineg].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} {mode:4s} | bpr {np.mean(losses):.4f} | valid "
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
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
