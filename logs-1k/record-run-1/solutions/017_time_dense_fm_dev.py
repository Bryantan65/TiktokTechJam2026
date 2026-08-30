"""FM with tuple-derived time features (dev screen).

Branch from the current best FM-style training loop, but add date-derived dense
features to let the score learn global temporal drift: normalized day index,
day-of-week sin/cos, weekend flag, and interactions of day index with tab.
The dense features are appended as a linear term; categorical FM fields remain
[user_id, video_id, author_id, tab, dur_bucket].
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class DenseTimeFM(torch.nn.Module):
    def __init__(self, dim, n_dense, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.w = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.dw = torch.nn.Parameter(torch.zeros(n_dense, dtype=torch.float32))

    def forward(self, X, D):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.w[X].sum(1)
        dense_lin = (D * self.dw[None, :]).sum(1)
        return inter + lin + dense_lin + self.b

    @torch.no_grad()
    def predict(self, X, D, bs=200_000, device='cpu'):
        self.eval()
        outs = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            db = torch.from_numpy(D[i:i + bs].astype(np.float32)).to(device)
            outs.append(self(xb, db).cpu().numpy())
        return np.concatenate(outs)


def parse_date_int(x):
    try:
        s = str(int(x))
        return datetime.strptime(s, '%Y%m%d')
    except Exception:
        return datetime(2022, 4, 8)


def build_dense_time(splits):
    # Use only train dates for centering/scaling to avoid label leakage. Date
    # values themselves are part of the input row and available at prediction.
    train_days = np.array([parse_date_int(r[0]).toordinal() for r in splits['train']], dtype=np.float32)
    base = float(train_days.min())
    scale = max(float(train_days.max() - train_days.min()), 1.0)
    out = {}
    for sp, rows in splits.items():
        D = np.zeros((len(rows), 9), dtype=np.float32)
        for i, r in enumerate(rows):
            dt = parse_date_int(r[0])
            day = (dt.toordinal() - base) / scale
            dow = dt.weekday()  # 0=Mon
            ang = 2.0 * np.pi * dow / 7.0
            tab = int(r[4]) if str(r[4]).lstrip('-').isdigit() else 0
            D[i, 0] = day
            D[i, 1] = day * day
            D[i, 2] = np.sin(ang)
            D[i, 3] = np.cos(ang)
            D[i, 4] = 1.0 if dow >= 5 else 0.0
            # Coarse tab-specific temporal trends for the dominant tab mix drift.
            if 0 <= tab <= 3:
                D[i, 5 + tab] = day
        # standardize with train stats later
        out[sp] = D
    mu = out['train'].mean(axis=0)
    sd = out['train'].std(axis=0)
    sd[sd < 1e-6] = 1.0
    for sp in out:
        out[sp] = ((out[sp] - mu) / sd).astype(np.float32)
    print('dense time train mean/std before standardize:', mu, sd)
    return out


def run(splits, k=16, lr=0.001, l2=1e-6, dense_l2=1e-5, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    dense = build_dense_time(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Dtr = dense['train']
    Dva = dense['valid']

    model = DenseTimeFM(dim, n_dense=Dtr.shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([
        {'params': [model.V, model.w], 'weight_decay': l2},
        {'params': [model.dw], 'weight_decay': dense_l2},
        {'params': [model.b], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Dtr_t = torch.from_numpy(Dtr.astype(np.float32))
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
            db = Dtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, db)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        va_scores = model.predict(Xva, Dva, device=device)
        va = evaluate(uva, yva, va_scores)
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, enc, dense


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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+dense_time")

    model, enc, dense = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                            seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, dense[target], device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== dense_time_fm_dev (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, dense[sp], device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
