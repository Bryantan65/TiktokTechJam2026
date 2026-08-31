"""FM with BCE warm-up followed by within-user BPR fine-tuning.

Node 2 showed that BPR is a small positive signal, but from random initialisation
it must learn both calibration/popularity structure and pairwise ordering.  This
variant first learns the stable BCE baseline representation for a few epochs,
then switches to BPR and only checkpoints during the BPR phase so the tested
model is genuinely pairwise-finetuned rather than the warm-up model.
"""
import argparse
import os
import sys
import time

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


def build_pair_sampler(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    pos_by_user = {}
    neg_by_user = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0.5:
            pos_by_user.setdefault(u, []).append(i)
        else:
            neg_by_user.setdefault(u, []).append(i)
    pos_idx = []
    neg_pools = []
    for u, ps in pos_by_user.items():
        ns = neg_by_user.get(u)
        if ns:
            arr = np.asarray(ns, dtype=np.int64)
            for p in ps:
                pos_idx.append(p)
                neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def run(splits, k=16, lr=0.001, l2=1e-6, warm_epochs=6, bpr_epochs=30,
        bs=8192, patience=4, bce_weight=0.02, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)

    # Warm-up: learn the strong pointwise FM representation, but do not save it
    # as best_state; the experiment is the post-warmup BPR model.
    for ep in range(1, warm_epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = bce_loss(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if verbose:
            va = evaluate(uva, yva, model.predict(Xva, device=device))
            print(f"  warm {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

    # A slightly smaller LR avoids destroying the warm-started solution.
    for g in opt.param_groups:
        g['lr'] = lr * 0.5

    pos_idx, neg_pools = build_pair_sampler(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('no within-user positive/negative pairs available')
    n_pairs_per_epoch = max(len(ytr), len(pos_idx))
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, bpr_epochs + 1):
        order = rng.integers(0, len(pos_idx), size=n_pairs_per_epoch, dtype=np.int64)
        t0 = time.time()
        model.train()
        losses = []
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
                loss = loss + bce_weight * bce_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  bpr  {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at BPR epoch {ep}")
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

    model, enc = run(splits, k=a.k, lr=a.lr, seed=a.seed, device=a.device,
                     verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== warm_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
