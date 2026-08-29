"""FM trained with a same-user listwise softmax loss.

For each sampled user, maximize the softmax probability mass assigned to that
user's positive impressions: logsumexp(scores_all) - logsumexp(scores_pos).
This is a direct within-user ranking objective and keeps the same FM features as
nodes 1/2.
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


def build_user_groups(y, users):
    buckets = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        if u not in buckets:
            buckets[u] = [[], []]
        buckets[u][1 if yy > 0.5 else 0].append(i)
    groups, weights = [], []
    for neg, pos in buckets.values():
        if pos and neg:
            idx = np.asarray(pos + neg, dtype=np.int64)
            lab = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.float32)
            # Shuffle within group once so positives are not always first.
            order = np.arange(len(idx))
            groups.append((idx, lab, len(pos), len(neg)))
            # Balance GAUC's positive-count weighting with nDCG's per-user nature.
            weights.append(min(len(pos), 20) * np.sqrt(len(idx)))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return groups, weights


def sample_user_batch(groups, weights, target_rows, rng):
    chosen = []
    nrows = 0
    # Replacement sampling gives a stochastic epoch and avoids huge user batches.
    while nrows < target_rows:
        gi = int(rng.choice(len(groups), p=weights))
        idx, lab, _, _ = groups[gi]
        # Very large users can dominate memory/time; subsample but keep both labels.
        if len(idx) > 256:
            pos_idx = idx[lab > 0.5]
            neg_idx = idx[lab < 0.5]
            npos = min(len(pos_idx), 64)
            nneg = min(len(neg_idx), 192)
            p = pos_idx[rng.integers(0, len(pos_idx), size=npos)]
            n = neg_idx[rng.integers(0, len(neg_idx), size=nneg)]
            ii = np.concatenate([p, n]).astype(np.int64)
            ll = np.concatenate([np.ones(npos, dtype=np.float32),
                                 np.zeros(nneg, dtype=np.float32)])
            perm = rng.permutation(len(ii))
            ii, ll = ii[perm], ll[perm]
        else:
            perm = rng.permutation(len(idx))
            ii, ll = idx[perm], lab[perm]
        chosen.append((ii, ll))
        nrows += len(ii)
    return chosen, nrows


def listwise_loss_for_batch(model, Xtr_t, user_batch, device):
    idx = np.concatenate([g[0] for g in user_batch])
    labels = np.concatenate([g[1] for g in user_batch])
    xb = Xtr_t[torch.from_numpy(idx)].to(device)
    scores = model(xb)
    labels_t = torch.from_numpy(labels).to(device)
    losses = []
    off = 0
    for ii, _ in user_batch:
        n = len(ii)
        s = scores[off:off + n]
        m = labels_t[off:off + n] > 0.5
        # -log(sum_pos exp(s) / sum_all exp(s)); zero only if all rows positive,
        # but such users are excluded when groups are built.
        losses.append(torch.logsumexp(s, dim=0) - torch.logsumexp(s[m], dim=0))
        off += n
    return torch.stack(losses).mean()


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    groups, weights = build_user_groups(ytr, utr)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, int(np.ceil(len(ytr) / bs)))
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            user_batch, _ = sample_user_batch(groups, weights, bs, rng)
            opt.zero_grad(set_to_none=True)
            loss = listwise_loss_for_batch(model, Xtr_t, user_batch, device)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | listwise {np.mean(losses):.4f} | valid "
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
        print(f"\n=== listwise_softmax_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
