"""FM with categorical time bucket crosses (dev screen).

Refines node 17: dense global date terms may be too weak for an FM because they
only add a linear drift.  Here date-derived features are categorical fields so
FM interactions can learn (user/item/tab) x time effects while future days are
clipped into train-observed buckets.
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


class FM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.w = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.w[X].sum(1)
        return inter + lin + self.b

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        outs = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            outs.append(self(xb).cpu().numpy())
        return np.concatenate(outs)


def parse_date_int(x):
    try:
        return datetime.strptime(str(int(x)), '%Y%m%d')
    except Exception:
        return datetime(2022, 4, 8)


def _tab_val(x):
    try:
        t = int(x)
    except Exception:
        t = 0
    return max(0, min(6, t))


def add_time_cats(splits, enc, base_dim):
    # Derive train-time scale only, then clip future dates into the last bin so
    # every category used at validation/test has had training updates.
    train_ord = np.array([parse_date_int(r[0]).toordinal() for r in splits['train']], dtype=np.int64)
    lo = int(train_ord.min())
    hi = int(train_ord.max())
    span = max(hi - lo + 1, 1)
    n_bins = 5
    sizes = [n_bins, 7, 7 * 7, 7 * n_bins, 2, 7 * 2]
    offsets = np.cumsum([base_dim] + sizes[:-1]).astype(np.int64)
    new_dim = base_dim + int(sum(sizes))

    out = {}
    for sp, rows in splits.items():
        X, y, u = enc[sp]
        extra = np.zeros((len(rows), len(sizes)), dtype=np.int64)
        for i, r in enumerate(rows):
            dt = parse_date_int(r[0])
            raw_bin = int(np.floor((dt.toordinal() - lo) * n_bins / span))
            day_bin = max(0, min(n_bins - 1, raw_bin))
            dow = dt.weekday()
            wknd = 1 if dow >= 5 else 0
            tab = _tab_val(r[4])
            vals = [
                day_bin,
                dow,
                tab * 7 + dow,
                tab * n_bins + day_bin,
                wknd,
                tab * 2 + wknd,
            ]
            extra[i] = offsets + np.asarray(vals, dtype=np.int64)
        out[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, u)
    print(f"added categorical time fields: daybin,dow,tab*dow,tab*daybin,weekend,tab*weekend; dim {base_dim}->{new_dim}")
    return out, new_dim


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True):
    enc0, base_dim = encode(splits)
    enc, dim = add_time_cats(splits, enc0, base_dim)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    model = FM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.w], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
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
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
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
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+categorical_time")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== time_cat_fm_dev (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
