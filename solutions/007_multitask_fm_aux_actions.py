"""Multi-task FM using auxiliary engagement actions when available.

The prediction target remains the official encoded main label. During training,
we add BCE losses for available binary auxiliary columns such as is_click,
is_like, is_follow, is_comment, and is_forward from the raw training split. The
FM interaction embedding V is shared across tasks, with task-specific first-order
weights and biases. At inference we output only the main task head.
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


AUX_CANDIDATES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']


class MultiTaskFM(torch.nn.Module):
    """Shared FM interaction with task-specific linear terms.

    logits_t = b_t + sum(W_t[x]) + shared_interaction(x)
    """
    def __init__(self, dim, n_tasks=1, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(n_tasks, dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))  # (B,)
        linear = self.W[:, X].sum(2).transpose(0, 1)             # (B,T)
        return self.b.unsqueeze(0) + linear + inter.unsqueeze(1)

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb)[:, 0].cpu().numpy())
        return np.concatenate(out)


def has_columns(obj):
    return hasattr(obj, 'columns')


def extract_aux(raw_train, n_rows):
    """Return aux matrix and names. Empty matrix if raw split lacks columns."""
    cols = []
    vals = []
    if has_columns(raw_train):
        for c in AUX_CANDIDATES:
            if c in raw_train.columns:
                arr = np.asarray(raw_train[c]).astype(np.float32)
                if len(arr) == n_rows:
                    # Keep only genuine 0/1-ish binary columns.
                    finite = arr[np.isfinite(arr)]
                    if len(finite) and finite.min() >= 0 and finite.max() <= 1:
                        cols.append(c)
                        vals.append(np.nan_to_num(arr, nan=0.0))
    elif isinstance(raw_train, dict):
        for c in AUX_CANDIDATES:
            if c in raw_train:
                arr = np.asarray(raw_train[c]).astype(np.float32)
                if len(arr) == n_rows:
                    finite = arr[np.isfinite(arr)]
                    if len(finite) and finite.min() >= 0 and finite.max() <= 1:
                        cols.append(c)
                        vals.append(np.nan_to_num(arr, nan=0.0))
    if vals:
        return np.stack(vals, axis=1).astype(np.float32), cols
    return np.zeros((n_rows, 0), dtype=np.float32), []


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        aux_weight=0.10, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    aux, aux_names = extract_aux(splits['train'], len(ytr))
    n_tasks = 1 + aux.shape[1]

    model = MultiTaskFM(dim, n_tasks=n_tasks, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    lossfn = torch.nn.BCEWithLogitsLoss()

    Y = np.concatenate([ytr.astype(np.float32).reshape(-1, 1), aux], axis=1)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Y_t = torch.from_numpy(Y.astype(np.float32))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    if verbose:
        print(f"  aux tasks: {aux_names if aux_names else 'none'}")

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = Y_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            main_loss = lossfn(logits[:, 0], yb[:, 0])
            if n_tasks > 1:
                aux_loss = lossfn(logits[:, 1:], yb[:, 1:])
                loss = main_loss + aux_weight * aux_loss
            else:
                loss = main_loss
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
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, enc


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
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== multitask_fm_aux_actions (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
