"""Fixed-epoch FM plus a small smoothed target-encoding residual.

The FM already uses video/author/tab/duration IDs, but a low-rank BCE model may
not fully capture simple empirical rate effects useful for within-user ranking.
This script trains the same FM for a fixed 9 epochs (near the known validation
peak, avoiding metric computation inside the solution) and adds a small residual
from train-only smoothed target logits for video, author, video-tab, author-tab,
and tab-duration.
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


def train_fixed_fm(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=9, bs=8192,
                   seed=0, device='cpu', verbose=False):
    Xtr, ytr, _ = enc['train']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
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
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | {time.time() - t0:.1f}s")
    return model


def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def one_field_logit(train_vals, y, pred_vals, prior, global_mean):
    train_vals = train_vals.astype(np.int64)
    pred_vals = pred_vals.astype(np.int64)
    maxv = int(max(train_vals.max(initial=0), pred_vals.max(initial=0))) + 1
    cnt = np.bincount(train_vals, minlength=maxv).astype(np.float32)
    sm = np.bincount(train_vals, weights=y.astype(np.float32), minlength=maxv).astype(np.float32)
    rate = (sm + prior * global_mean) / (cnt + prior)
    return safe_logit(rate[pred_vals]) - safe_logit(global_mean)


def pair_keys(a, b):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    base = int(max(b.max(initial=0), 0)) + 1
    return a * base + b, base


def pair_field_logit(train_a, train_b, y, pred_a, pred_b, prior, global_mean):
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
    ok = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == kpr)
    out = np.zeros(len(kpr), dtype=np.float32)
    out[ok] = vals[pos[ok]]
    return out


def target_residual(enc, split):
    Xtr, ytr, _ = enc['train']
    Xp, _, _ = enc[split]
    y = ytr.astype(np.float32)
    gm = float(np.mean(y))
    # FIELDS: [user_id, video_id, author_id, tab, dur_bucket]
    vid_tr, auth_tr, tab_tr, dur_tr = Xtr[:, 1], Xtr[:, 2], Xtr[:, 3], Xtr[:, 4]
    vid_p, auth_p, tab_p, dur_p = Xp[:, 1], Xp[:, 2], Xp[:, 3], Xp[:, 4]

    # Conservative priors: high-cardinality video needs more shrinkage; tab/dur
    # combos are dense and can use less.
    r_video = one_field_logit(vid_tr, y, vid_p, prior=50.0, global_mean=gm)
    r_author = one_field_logit(auth_tr, y, auth_p, prior=30.0, global_mean=gm)
    r_vtab = pair_field_logit(vid_tr, tab_tr, y, vid_p, tab_p, prior=80.0, global_mean=gm)
    r_atab = pair_field_logit(auth_tr, tab_tr, y, auth_p, tab_p, prior=50.0, global_mean=gm)
    r_tdur = pair_field_logit(tab_tr, dur_tr, y, tab_p, dur_p, prior=20.0, global_mean=gm)

    return (0.30 * r_video + 0.25 * r_author + 0.20 * r_vtab +
            0.15 * r_atab + 0.10 * r_tdur).astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    enc, dim = encode(splits)
    model = train_fixed_fm(enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                           seed=a.seed, device=a.device, verbose=(a.out is None))
    X, y, users = enc[a.split]
    fm_scores = model.predict(X, device=a.device)
    te = target_residual(enc, a.split)
    scores = fm_scores + 0.08 * te

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"wrote scores for split={a.split}; no metrics computed")
