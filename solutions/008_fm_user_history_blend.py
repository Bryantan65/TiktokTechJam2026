"""Baseline FM plus small train-only user-history residuals.

This tests a cheap user-behavior feature: for a candidate row, add smoothed
historical positive-rate logits for the same user with the same video/author/tab
/duration bucket, computed from train only. These vary within a user across the
items being ranked, unlike pure user first-order effects.
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


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
           seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
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
        idx = rng.permutation(len(ytr))
        t0 = time.time()
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
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, enc


def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def pair_residual(train_a, train_b, y, pred_a, pred_b, prior, global_mean):
    train_a = train_a.astype(np.int64)
    train_b = train_b.astype(np.int64)
    pred_a = pred_a.astype(np.int64)
    pred_b = pred_b.astype(np.int64)
    base = int(max(train_b.max(initial=0), pred_b.max(initial=0))) + 1
    ktr = train_a * base + train_b
    kpr = pred_a * base + pred_b
    keys, inv = np.unique(ktr, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float32)
    sm = np.bincount(inv, weights=y.astype(np.float32)).astype(np.float32)
    rate = (sm + prior * global_mean) / (cnt + prior)
    vals = safe_logit(rate) - safe_logit(global_mean)
    pos = np.searchsorted(keys, kpr)
    # avoid indexing keys with len(keys) for missing right-edge values
    ok_range = pos < len(keys)
    out = np.zeros(len(kpr), dtype=np.float32)
    ok = np.zeros(len(kpr), dtype=bool)
    ok[ok_range] = keys[pos[ok_range]] == kpr[ok_range]
    out[ok] = vals[pos[ok]]
    return out


def user_history_residual(enc, split):
    Xtr, ytr, _ = enc['train']
    Xp, _, _ = enc[split]
    y = ytr.astype(np.float32)
    gm = float(np.mean(y))
    u_tr = Xtr[:, 0]
    u_p = Xp[:, 0]
    # FIELDS: user_id, video_id, author_id, tab, dur_bucket
    uv = pair_residual(u_tr, Xtr[:, 1], y, u_p, Xp[:, 1], prior=4.0, global_mean=gm)
    ua = pair_residual(u_tr, Xtr[:, 2], y, u_p, Xp[:, 2], prior=8.0, global_mean=gm)
    ut = pair_residual(u_tr, Xtr[:, 3], y, u_p, Xp[:, 3], prior=20.0, global_mean=gm)
    ud = pair_residual(u_tr, Xtr[:, 4], y, u_p, Xp[:, 4], prior=20.0, global_mean=gm)
    return (0.45 * uv + 0.35 * ua + 0.12 * ut + 0.08 * ud).astype(np.float32)


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

    model, enc = run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                        device=a.device, verbose=a.out is None)
    X, y, users = enc[a.split]
    fm_scores = model.predict(X, device=a.device)
    hist = user_history_residual(enc, a.split)
    scores = fm_scores + 0.05 * hist

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== fm_user_history_blend (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            sc = model.predict(Xs, device=a.device) + 0.05 * user_history_residual(enc, sp)
            r = evaluate(us, ys, sc)
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
