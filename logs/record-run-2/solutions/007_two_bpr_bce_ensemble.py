"""Blend two independently sampled BPR FMs with one pointwise BCE FM.

Node 6 showed complementary signal from BPR and BCE. This keeps the same blend
but reduces BPR negative-sampling/initialization noise by averaging two BPR FMs
trained with different seeds before adding the BCE component.
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


def build_user_pos_neg(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    groups = []
    start = 0
    n = len(users)
    while start < n:
        end = start + 1
        while end < n and us[end] == us[start]:
            end += 1
        idx = order[start:end]
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
        start = end
    return groups


def sample_pairs(groups, rng):
    pos_parts = []
    neg_parts = []
    for pos, neg in groups:
        p = pos.copy()
        rng.shuffle(p)
        n = rng.choice(neg, size=len(p), replace=(len(neg) < len(p)))
        pos_parts.append(p)
        neg_parts.append(n.astype(np.int64))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    perm = rng.permutation(len(pi))
    return pi[perm], ni[perm]


def train_bpr(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True, tag='bpr'):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    groups = build_user_pos_neg(ytr, utr)
    if verbose:
        npos = sum(len(p) for p, _ in groups)
        print(f"{tag} groups={len(groups):,d} paired_positives/epoch={npos:,d}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        pi, ni = sample_pairs(groups, rng)
        t0 = time.time()
        model.train(); losses = []
        for i in range(0, len(pi), bs):
            ps = torch.from_numpy(pi[i:i + bs])
            ns = torch.from_numpy(ni[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp) - model(xn))).mean()
            loss.backward(); opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  {tag} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def train_bce(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed + 10000).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    rng = np.random.default_rng(seed + 777)
    best, best_state, bad = -1.0, None, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train(); losses = []
        order = rng.permutation(n)
        for i in range(0, n, bs):
            idx = torch.from_numpy(order[i:i + bs])
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  bce epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


class BlendedModel:
    def __init__(self, bprs, bce, enc, device='cpu', w_bce=0.25):
        self.bprs = bprs
        self.bce = bce
        self.device = device
        self.w_bce = w_bce
        Xtr, _, _ = enc['train']
        self.bpr_stats = []
        for m in bprs:
            s = m.predict(Xtr, device=device)
            self.bpr_stats.append((float(s.mean()), float(s.std() + 1e-6)))
        s2 = bce.predict(Xtr, device=device)
        self.m2, self.sd2 = float(s2.mean()), float(s2.std() + 1e-6)

    def predict(self, X):
        bsum = None
        for m, (mu, sd) in zip(self.bprs, self.bpr_stats):
            z = (m.predict(X, device=self.device) - mu) / sd
            bsum = z if bsum is None else bsum + z
        zbpr = bsum / len(self.bprs)
        s2 = self.bce.predict(X, device=self.device)
        z2 = (s2 - self.m2) / self.sd2
        return zbpr + self.w_bce * z2


def run(splits, k=16, lr=0.001, epochs=40, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    bpr1 = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                     device=device, verbose=verbose, tag='bpr1')
    bpr2 = train_bpr(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed + 12345,
                     device=device, verbose=verbose, tag='bpr2')
    bce = train_bce(enc, dim, k=k, lr=lr, epochs=epochs, seed=seed,
                    device=device, verbose=verbose)
    return BlendedModel([bpr1, bpr2], bce, enc, device=device, w_bce=0.25), enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                     device=a.device, verbose=a.out is None)
    X, y, users = enc[a.split]
    scores = model.predict(X)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== two_bpr_bce_ensemble (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
