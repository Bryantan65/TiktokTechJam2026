"""FM with semi-hard BPR+BCE and a light LambdaRank-style nDCG@5 weight.

Improves node 8 by keeping its conservative semi-hard negative sampling, but
weights each same-user pair by the approximate absolute change in nDCG@5 if the
positive and negative swapped under the current model ranking.  The weight is
normalised inside each batch so this should emphasise top-5 mistakes without
changing the overall learning-rate scale too much.
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


def build_user_groups(users, y):
    pos_by_u, neg_by_u = defaultdict(list), defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0.5:
            pos_by_u[u].append(i)
        else:
            neg_by_u[u].append(i)
    groups = []
    for u in pos_by_u.keys():
        if u in neg_by_u:
            groups.append((np.asarray(pos_by_u[u], dtype=np.int64),
                           np.asarray(neg_by_u[u], dtype=np.int64)))
    return groups


def make_pairs(groups, rng, train_scores=None, semi_k=4, semi_frac=0.25,
               lambda_alpha=2.5):
    """Sample same-user pairs and return per-pair LambdaRank-style weights."""
    left, right, wts = [], [], []
    order_groups = np.arange(len(groups))
    rng.shuffle(order_groups)
    disc5 = np.asarray([1.0 / np.log2(i + 2.0) for i in range(5)], dtype=np.float32)
    for gi in order_groups:
        ps, ns = groups[gi]
        m = len(ps)
        chosen = rng.choice(ns, size=m, replace=True)

        if train_scores is not None and semi_k > 1 and semi_frac > 0:
            mask = rng.random(m) < semi_frac
            hh = int(mask.sum())
            if hh > 0:
                psel = ps[mask]
                cand = rng.choice(ns, size=(hh, semi_k), replace=True)
                cs = train_scores[cand]
                pscores = train_scores[psel]
                ok = cs < pscores[:, None]
                masked = np.where(ok, cs, -np.inf)
                any_ok = ok.any(axis=1)
                if any_ok.any():
                    rows = np.where(any_ok)[0]
                    cols = np.argmax(masked[rows], axis=1)
                    tmp = chosen[mask]
                    tmp[rows] = cand[rows, cols]
                    chosen[mask] = tmp

        weights = np.ones(m, dtype=np.float32)
        if train_scores is not None and lambda_alpha > 0:
            # Current within-user ranks over this user's training impressions.
            all_idx = np.concatenate([ps, ns])
            order = np.argsort(-train_scores[all_idx], kind='mergesort')
            ranks = np.empty(len(all_idx), dtype=np.int32)
            ranks[order] = np.arange(1, len(all_idx) + 1, dtype=np.int32)
            local_rank = dict(zip(all_idx.tolist(), ranks.tolist()))
            rp = np.fromiter((local_rank[int(x)] for x in ps), dtype=np.int32, count=m)
            rn = np.fromiter((local_rank[int(x)] for x in chosen), dtype=np.int32, count=m)
            dp = np.where(rp <= 5, 1.0 / np.log2(rp.astype(np.float32) + 1.0), 0.0)
            dn = np.where(rn <= 5, 1.0 / np.log2(rn.astype(np.float32) + 1.0), 0.0)
            idcg = float(disc5[:min(len(ps), 5)].sum())
            if idcg > 0:
                delta = np.abs(dp - dn) / idcg
                weights = (1.0 + lambda_alpha * delta).astype(np.float32)

        left.append(ps)
        right.append(chosen.astype(np.int64, copy=False))
        wts.append(weights)
    p = np.concatenate(left)
    n = np.concatenate(right)
    w = np.concatenate(wts)
    perm = rng.permutation(len(p))
    return p[perm], n[perm], w[perm]


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, bce_warmup=1, bce_weight=0.15,
        semi_k=4, semi_frac=0.25, lambda_alpha=2.5):
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
        else:
            train_scores = model.predict(Xtr, device=device)
            pidx, nidx, pair_w = make_pairs(groups, rng, train_scores=train_scores,
                                            semi_k=semi_k, semi_frac=semi_frac,
                                            lambda_alpha=lambda_alpha)
            for i in range(0, len(pidx), bs):
                ps = torch.from_numpy(pidx[i:i + bs])
                ns = torch.from_numpy(nidx[i:i + bs])
                wt = torch.from_numpy(pair_w[i:i + bs]).to(device)
                xp = Xtr_t[ps].to(device)
                xn = Xtr_t[ns].to(device)
                opt.zero_grad(set_to_none=True)
                sp = model(xp)
                sn = model(xn)
                loss_vec = torch.nn.functional.softplus(-(sp - sn))
                bpr_loss = (loss_vec * wt).sum() / (wt.sum() + 1e-8)
                point_scores = torch.cat([sp, sn])
                point_labels = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
                loss = bpr_loss + bce_weight * bce(point_scores, point_labels)
                loss.backward()
                opt.step()
                losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            phase = 'bce' if ep <= bce_warmup else 'lambda_semihard'
            print(f"  epoch {ep:2d} {phase} | loss {np.mean(losses):.4f} | valid "
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
        print(f"\n=== lambda_semihard (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
