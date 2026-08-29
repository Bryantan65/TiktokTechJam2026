"""FM trained with same-user listwise softmax loss.

Keeps the baseline FM features/architecture but replaces pair sampling with a
per-user softmax objective: for each training user with both positive and
negative impressions, maximize the log-softmax probability assigned to that
user's positive rows among all of that user's rows in the batch.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

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


def build_user_groups(users, y):
    by_u = defaultdict(list)
    for i, u in enumerate(users):
        by_u[u].append(i)
    groups = []
    for idxs in by_u.values():
        yy = y[idxs]
        # all-positive/all-negative users do not define within-user ordering
        if yy.max() > 0.5 and yy.min() < 0.5:
            groups.append(np.asarray(idxs, dtype=np.int64))
    return groups


def iter_user_batches(groups, rng, max_rows=8192):
    order = rng.permutation(len(groups))
    cur, n = [], 0
    for gi in order:
        g = groups[gi]
        if cur and n + len(g) > max_rows:
            yield cur
            cur, n = [], 0
        cur.append(g)
        n += len(g)
    if cur:
        yield cur


def listwise_loss(scores, labels, lengths):
    losses = []
    start = 0
    for ln in lengths:
        end = start + ln
        s = scores[start:end]
        y = labels[start:end]
        pos = y > 0.5
        # average positive log probability; users with more positives still
        # contribute through having more target rows inside their list.
        losses.append(torch.logsumexp(s, dim=0) - s[pos].mean())
        start = end
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, bce_warmup=1):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bce = torch.nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    groups = build_user_groups(utr, ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        if ep <= bce_warmup:
            # Warm start only; do not early-stop/save on this pointwise phase.
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
                print(f"  epoch {ep:2d} bce-warm | loss {np.mean(losses):.4f} | {time.time() - t0:.1f}s")
            continue

        for batch_groups in iter_user_batches(groups, rng, max_rows=bs):
            idx = np.concatenate(batch_groups)
            lengths = [len(g) for g in batch_groups]
            xb = Xtr_t[torch.from_numpy(idx)].to(device)
            yb = ytr_t[torch.from_numpy(idx)].to(device)
            opt.zero_grad(set_to_none=True)
            loss = listwise_loss(model(xb), yb, lengths)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} list | loss {np.mean(losses):.4f} | valid "
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

    if best_state is None:
        best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
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
        print(f"\n=== listwise_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
