"""FM with BCE warmup followed by same-user BPR fine-tuning.

Node 2 showed weak positive signal for BPR from random initialisation. This
variant first learns a stable FM with pointwise BCE for a few epochs, then
switches to the ranking-aligned same-user BPR objective. Checkpoints are saved
only during the BPR phase so the experiment cannot silently return the warmup
baseline unchanged.
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


def build_pair_pools(y, users):
    pos_by_u = {}
    neg_by_u = {}
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u.setdefault(uu, []).append(i)
        else:
            neg_by_u.setdefault(uu, []).append(i)

    neg_arrays = []
    pos_idx = []
    pos_gid = []
    for uu, ps in pos_by_u.items():
        ns = neg_by_u.get(uu)
        if ns:
            gid = len(neg_arrays)
            neg_arrays.append(np.asarray(ns, dtype=np.int64))
            pos_idx.extend(ps)
            pos_gid.extend([gid] * len(ps))
    return (np.asarray(pos_idx, dtype=np.int64),
            np.asarray(pos_gid, dtype=np.int64),
            neg_arrays)


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, warmup_epochs=5):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)

    # Phase 1: pointwise warmup, deliberately no checkpointing here.
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, warmup_epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if verbose:
            va = evaluate(uva, yva, model.predict(Xva, device=device))
            print(f"  warm {ep:2d} | bce {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

    # Phase 2: same-user BPR fine-tuning. Only this phase can set best_state.
    pos_idx, pos_gid, neg_arrays = build_pair_pools(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('no same-user positive/negative pairs found')
    pairs_per_epoch = len(ytr)
    steps_per_epoch = int(np.ceil(pairs_per_epoch / bs))

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            pick = rng.integers(0, len(pos_idx), size=bs)
            pidx_np = pos_idx[pick]
            gids = pos_gid[pick]
            nidx_np = np.empty(bs, dtype=np.int64)
            for j, gid in enumerate(gids):
                ns = neg_arrays[int(gid)]
                nidx_np[j] = ns[rng.integers(0, len(ns))]

            xp = Xtr_t[torch.from_numpy(pidx_np)].to(device)
            xn = Xtr_t[torch.from_numpy(nidx_np)].to(device)
            x = torch.cat([xp, xn], dim=0)
            opt.zero_grad(set_to_none=True)
            s = model(x)
            sp, sn = s[:len(xp)], s[len(xp):]
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

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
        print(f"\n=== bce_warm_bpr (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
