"""FM in PyTorch — a straight port of the official baseline, model unchanged.

Purpose: replace hand-derived gradients with autograd. Nothing about the model,
data, splits or scoring changes, so this must reproduce the official numbers
(valid primary ~0.6015, test ~0.5953). If it doesn't, the port is wrong.

Why bother when the score is identical: every later experiment changes the loss
function. With hand-derived gradients that is a fresh calculus problem each
time, and a mistake does not crash - it trains toward the wrong thing and
reports a plausible number. Autograd makes the gradient correct by construction.

To make the comparison exact this reuses the baseline's initialisation and batch
order (same numpy RNG, same seed), so the only moving part is how gradients are
computed.

    python solutions/001_torch_fm.py --data_dir rec_datasets/KuaiRand-Pure/data
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# APPEND, not insert(0). The harness puts harness/ first on PYTHONPATH so that a
# non-Pure variant resolves `data` to harness/data.py, which knows how to find
# 1k and 27k filenames. Prepending the kit here overrode that, and the kit's
# loader hardcodes video_features_basic_pure.csv - so this file could not run on
# any dataset except Pure, which is why 1k never had a baseline control row.
# Appending keeps a standalone run working (the kit is still found, just later)
# while letting the harness's choice win. Pure is unaffected either way:
# harness/data.py delegates to the kit's own load() and returns row-identical
# splits on all three.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402  official, unmodified
from evaluate import evaluate                  # noqa: E402  official, unmodified


class TorchFM(torch.nn.Module):
    """Same arithmetic as baseline.py's FM class.

    logits = b + sum(W[x]) + 0.5 * ((sum E)^2 - sum(E^2))
    """

    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)       # same init as the baseline
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]                                    # (B, F, k)
        S = E.sum(1)                                     # (B, k)
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


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']       # early stopping only; test is never read here

    model = TorchFM(dim, k=k, seed=seed).to(device)
    # l2 in the baseline is applied to V and W but not to b, so b gets its own
    # group. (In practice b cannot affect the score at all: it is constant
    # across every row, and the metric ranks within a user.)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    lossfn = torch.nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr)

    rng = np.random.default_rng(seed)           # same batch order as the baseline
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
            loss = lossfn(model(xb), yb)        # <- the only line to change
            loss.backward()                     # <- autograd, no hand calculus
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
    ap.add_argument('--split', default='valid',
                    choices=['train', 'valid', 'test', 'dev'],
                    help='which split to write predictions for. "dev" is a '
                         'train-only holdout for screening; see the block below')
    ap.add_argument('--out', default=None,
                    help='write predictions here as .npy, one score per row')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        # Screening run: fit on the earlier training days, predict the later
        # ones. Same row format, so nothing downstream changes - but no test
        # data exists in this view at all, and the holdout stands in for
        # 'valid'. Costs no public-validation experiment.
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
        # Harness mode: emit predictions only. The harness owns the labels and
        # does the scoring, so a solution can neither grade itself nor pick
        # which split it is graded on.
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        # Standalone mode: report, for a human running this directly.
        print(f"\n=== torch_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
        print("\n  official baseline: valid primary 0.6016 | test primary 0.5946")
