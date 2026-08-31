"""FM trained with a within-user listwise softmax loss.

This refines the BPR loss direction: instead of sampling one negative for each
positive, each training step ranks all impressions for a sampled user and applies
a softmax cross-entropy whose target mass is spread over that user's positives.
A small BCE term keeps the FM biases calibrated without dominating the ranking
objective.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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


def make_user_groups(y, users, max_rows_per_user=80):
    """Return arrays of row indices per user that have at least one pos/neg.

    Very long histories are capped per epoch by sampling inside the training loop;
    this prevents a few heavy users from dominating and keeps softmax groups small.
    """
    by_user = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, []).append(i)
    groups = []
    y = np.asarray(y)
    for idxs in by_user.values():
        arr = np.asarray(idxs, dtype=np.int64)
        s = float(y[arr].sum())
        if s > 0.0 and s < len(arr):
            groups.append(arr)
    return groups


def listwise_loss_for_batch(model, Xtr_t, ytr_t, groups, chosen, rng, device,
                            max_rows_per_user=80, bce_weight=0.03):
    rows = []
    sizes = []
    for gi in chosen:
        g = groups[int(gi)]
        if len(g) > max_rows_per_user:
            # Keep all positives when possible and sample negatives around them.
            yy = ytr_t[torch.from_numpy(g)].numpy()
            pos = g[yy > 0.5]
            neg = g[yy <= 0.5]
            room = max(1, max_rows_per_user - len(pos))
            if len(pos) >= max_rows_per_user:
                sel = rng.choice(pos, size=max_rows_per_user, replace=False)
            else:
                nsel = rng.choice(neg, size=min(room, len(neg)), replace=False)
                sel = np.concatenate([pos, nsel])
            rng.shuffle(sel)
            g = sel.astype(np.int64)
        rows.append(g)
        sizes.append(len(g))
    all_rows = np.concatenate(rows)
    xb = Xtr_t[torch.from_numpy(all_rows)].to(device)
    yb = ytr_t[torch.from_numpy(all_rows)].to(device)
    logits = model(xb)

    losses = []
    off = 0
    for n in sizes:
        s = logits[off:off + n]
        yy = yb[off:off + n]
        pos = yy.sum()
        # Target probability is uniform over positives.  This is equivalent to
        # -mean log P(positive | all impressions of this user segment).
        logp = torch.nn.functional.log_softmax(s, dim=0)
        losses.append(-(logp * (yy / pos)).sum())
        off += n
    loss = torch.stack(losses).mean()
    if bce_weight > 0:
        loss = loss + bce_weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
    return loss


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, user_bs=256,
        patience=5, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    groups = make_user_groups(ytr, utr)
    if not groups:
        raise RuntimeError('no users with both positive and negative rows')
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(groups))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(order), user_bs):
            chosen = order[i:i + user_bs]
            opt.zero_grad(set_to_none=True)
            loss = listwise_loss_for_batch(model, Xtr_t, ytr_t, groups, chosen,
                                           rng, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            bad = 0
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
    print({kk: len(vv) for kk, vv in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, seed=a.seed, device=a.device,
                     verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== listwise_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
