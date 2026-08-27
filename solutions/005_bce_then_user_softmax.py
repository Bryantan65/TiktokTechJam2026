"""FM with BCE warmup followed by per-user listwise softmax fine-tuning.

For each training user with both positive and negative impressions, the listwise
loss is:
    logsumexp(scores over that user's impressions) - mean(score of positives)
This is equivalent to cross-entropy where probability mass is distributed over
that user's positive rows, and directly trains within-user ranking. The script
keeps the best validation checkpoint over BCE warmup and listwise fine-tuning.
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


def make_user_groups(users, y):
    """Groups for users with at least one positive and one negative row."""
    users = np.asarray(users)
    y = np.asarray(y)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    yy = y[order]
    bounds = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    groups = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b].astype(np.int64)
        lab = yy[a:b].astype(np.float32)
        npos = int((lab > 0.5).sum())
        if npos > 0 and npos < len(lab):
            groups.append((idx, lab > 0.5))
    return groups


def eval_and_maybe_save(model, Xva, yva, uva, best, best_state, bad, phase, ep,
                        losses, verbose, device, t0):
    va = evaluate(uva, yva, model.predict(Xva, device=device))
    if verbose:
        print(f"  {phase:4s} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
              f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
              f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
    if va['primary'] > best + 1e-5:
        best = va['primary']
        best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
    return best, best_state, bad


def listwise_batch_loss(model, Xtr_t, groups_batch, device):
    idx_parts = [g[0] for g in groups_batch]
    pos_masks = [g[1] for g in groups_batch]
    lengths = [len(x) for x in idx_parts]
    idx = np.concatenate(idx_parts)
    xb = Xtr_t[torch.from_numpy(idx)].to(device)
    scores = model(xb)

    losses = []
    start = 0
    for ln, pm_np in zip(lengths, pos_masks):
        s = scores[start:start + ln]
        pm = torch.from_numpy(pm_np).to(device)
        losses.append(torch.logsumexp(s, dim=0) - s[pm].mean())
        start += ln
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, bce_epochs=10, list_epochs=10,
        bs=8192, user_bs=512, patience=4, seed=0, device='cpu', verbose=True):
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
    best, best_state, bad = -1.0, None, 0

    # Warm up with the reliable baseline objective.
    for ep in range(1, bce_epochs + 1):
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
            losses.append(loss.item())
        best, best_state, bad = eval_and_maybe_save(
            model, Xva, yva, uva, best, best_state, bad, 'bce', ep,
            losses, verbose, device, t0)
        if bad >= patience:
            if verbose:
                print(f"  BCE early stop at epoch {ep}")
            break

    groups = make_user_groups(utr, ytr)
    if verbose:
        rows = sum(len(g[0]) for g in groups)
        print(f"  listwise groups: {len(groups):,d} users, {rows:,d} rows")

    # Fine-tune more gently than BCE. The constant global bias cancels within
    # a user in this loss but keeping it in the optimizer is harmless.
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr * 0.25, betas=(0.9, 0.999), eps=1e-8)
    bad = 0
    for ep in range(1, list_epochs + 1):
        order = rng.permutation(len(groups))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), user_bs):
            gb = [groups[j] for j in order[i:i + user_bs]]
            opt.zero_grad(set_to_none=True)
            loss = listwise_batch_loss(model, Xtr_t, gb, device)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        best, best_state, bad = eval_and_maybe_save(
            model, Xva, yva, uva, best, best_state, bad, 'list', ep,
            losses, verbose, device, t0)
        if bad >= patience:
            if verbose:
                print(f"  listwise early stop at epoch {ep}")
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
    ap.add_argument('--epochs', type=int, default=40, help='accepted for compatibility; unused')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, seed=a.seed,
                     device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== bce_then_user_softmax (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
