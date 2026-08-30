"""FM trained with within-user BPR and a same-tab hard negative.

Improve node 2: two uniform negatives contain many easy cross-tab comparisons
because tab has very different positive rates.  Keep the same FM and BPR loss,
but make one of the two sampled negatives come from the same user AND same tab
when possible, so the model learns finer within-tab item ordering while the
second negative remains uniform over the user's negatives.
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


def make_pair_sampler(y, users, tabs):
    users = np.asarray(users)
    tabs = np.asarray(tabs)
    y = np.asarray(y)
    order = np.argsort(users, kind='mergesort')
    su = users[order]
    pos_rows = []
    neg_by_user = {}
    neg_by_user_tab = {}
    start = 0
    n = len(users)
    while start < n:
        end = start + 1
        while end < n and su[end] == su[start]:
            end += 1
        rows = order[start:end]
        pos = rows[y[rows] > 0.5]
        neg = rows[y[rows] <= 0.5]
        if len(pos) and len(neg):
            u = su[start]
            pos_rows.append(pos)
            neg_by_user[u] = neg.astype(np.int64)
            for t in np.unique(tabs[neg]):
                nt = neg[tabs[neg] == t]
                if len(nt):
                    neg_by_user_tab[(u, t)] = nt.astype(np.int64)
        start = end
    if not pos_rows:
        raise RuntimeError('No users with both positive and negative examples for BPR')
    pos_rows = np.concatenate(pos_rows).astype(np.int64)
    pos_users = users[pos_rows]
    pos_tabs = tabs[pos_rows]
    return pos_rows, pos_users, pos_tabs, neg_by_user, neg_by_user_tab


def sample_pairs(pos_rows, pos_users, pos_tabs, neg_by_user, neg_by_user_tab, rng):
    perm = rng.permutation(len(pos_rows))
    p = pos_rows[perm]
    pu = pos_users[perm]
    pt = pos_tabs[perm]
    negs = np.empty((len(p), 2), dtype=np.int64)
    for i, (u, t) in enumerate(zip(pu, pt)):
        pool_u = neg_by_user[u]
        pool_t = neg_by_user_tab.get((u, t), pool_u)
        negs[i, 0] = pool_t[rng.integers(len(pool_t))]
        negs[i, 1] = pool_u[rng.integers(len(pool_u))]
    return p, negs


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
    # FIELDS are [user_id, video_id, author_id, tab, dur_bucket]
    pos_rows, pos_users, pos_tabs, neg_by_user, neg_by_user_tab = make_pair_sampler(
        ytr, utr, Xtr[:, 3]
    )
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    n_neg = 2

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pidx, nidx = sample_pairs(pos_rows, pos_users, pos_tabs,
                                  neg_by_user, neg_by_user_tab, rng)
        model.train()
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs]).long()
            ns_np = nidx[i:i + bs].reshape(-1)
            ns = torch.from_numpy(ns_np).long()
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp).repeat_interleave(n_neg)
            sn = model(xn)
            loss = -torch.nn.functional.logsigmoid(sp - sn).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | bpr_same_tab {np.mean(losses):.4f} | valid "
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
        print(f"\n=== bpr_same_tab_neg (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
