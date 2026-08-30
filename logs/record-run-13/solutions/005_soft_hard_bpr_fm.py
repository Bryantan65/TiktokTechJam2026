"""Debug hard-negative BPR: delayed and low-rate hard mining.

Node 4 mined hard negatives from the start for most pairs and collapsed.  This
version keeps the same-user BPR setup but uses uniform negatives for a warmup,
then replaces only a small fraction of draws with sampled hard negatives.  It
saves/evaluates only after the hard-mining phase starts so the tested change is
not silently discarded in favour of a pure-uniform checkpoint.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402  early stopping only


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


def make_user_pair_sources(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)

    pos_idx = []
    neg_pools = []
    for uu, ps in pos.items():
        ns = neg.get(uu)
        if ns:
            arr = np.asarray(ns, dtype=np.int64)
            for p in ps:
                pos_idx.append(p)
                neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def sample_epoch_pairs(pos_idx, neg_pools, rng, neg_per_pos=3, scores=None,
                       hard_frac=0.10, candidates=4):
    total = len(pos_idx) * neg_per_pos
    p_out = np.empty(total, dtype=np.int64)
    n_out = np.empty(total, dtype=np.int64)
    k = 0
    use_hard = scores is not None and hard_frac > 0.0
    for p, pool in zip(pos_idx, neg_pools):
        m = len(pool)
        for _ in range(neg_per_pos):
            if use_hard and m > 1 and rng.random() < hard_frac:
                cand = pool[rng.integers(0, m, size=candidates)]
                n = cand[np.argmax(scores[cand])]
            else:
                n = pool[rng.integers(0, m)]
            p_out[k] = p
            n_out[k] = n
            k += 1
    order = rng.permutation(total)
    return p_out[order], n_out[order]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        neg_per_pos=3, warmup_epochs=4, hard_frac=0.10, candidates=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, neg_pools = make_user_pair_sources(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('No users with both positive and negative examples; cannot train BPR')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if ep <= warmup_epochs:
            tr_scores = None
            phase = 'uniform'
        else:
            tr_scores = model.predict(Xtr, device=device)
            phase = 'soft-hard'
        p_idx, n_idx = sample_epoch_pairs(
            pos_idx, neg_pools, rng, neg_per_pos=neg_per_pos, scores=tr_scores,
            hard_frac=hard_frac, candidates=candidates)

        model.train()
        losses = []
        for i in range(0, len(p_idx), bs):
            ps = torch.from_numpy(p_idx[i:i + bs])
            ns = torch.from_numpy(n_idx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn)
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} {phase:9s} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | pairs {len(p_idx):,d} | "
                  f"{time.time() - t0:.1f}s")

        # Do not let a warmup-only checkpoint win; this is a debug of hard mining.
        if ep > warmup_epochs:
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop at epoch {ep}")
                    break

    if best_state is None:
        best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
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
    ap.add_argument('--neg_per_pos', type=int, default=3)
    ap.add_argument('--warmup_epochs', type=int, default=4)
    ap.add_argument('--hard_frac', type=float, default=0.10)
    ap.add_argument('--candidates', type=int, default=4)
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

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                     neg_per_pos=a.neg_per_pos, warmup_epochs=a.warmup_epochs,
                     hard_frac=a.hard_frac, candidates=a.candidates,
                     seed=a.seed, device=a.device, verbose=a.out is None)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== soft_hard_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
