"""BPR-FM with a DIN-style attention term over prior positive video history.

The baseline FM is trained with same-user BPR.  This variant augments its score
with target-aware attention over a user's previous positive videos:

  score = FM(x) + alpha * <target_video_embedding, Attn(history_positive_videos)>

For training rows, history contains only positives that occurred earlier in the
training row order, avoiding self-label leakage.  For valid/test predictions,
history is built from all train positives only; valid/test labels are never used.
"""
import argparse
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class HistoryFM(torch.nn.Module):
    def __init__(self, dim, pad_id, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim + 1, k)).astype(np.float32)
        V0[pad_id] = 0.0
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(dim + 1, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.hist_scale = torch.nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.pad_id = int(pad_id)
        self.k = int(k)

    def fm_score(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    def hist_score(self, X, H, M):
        # Candidate video is field 1 in FIELDS=[user_id, video_id, author_id, tab, dur_bucket].
        target = self.V[X[:, 1]]                  # (B, k)
        hv = self.V[H]                            # (B, L, k)
        logits = (hv * target[:, None, :]).sum(2) / np.sqrt(float(self.k))
        logits = logits.masked_fill(M <= 0.0, -1.0e9)
        mx = logits.max(1, keepdim=True).values
        w = torch.exp(logits - mx) * M
        denom = w.sum(1, keepdim=True).clamp_min(1.0e-8)
        ctx = (hv * (w / denom)[:, :, None]).sum(1)
        return (ctx * target).sum(1)

    def forward(self, X, H=None, M=None):
        s = self.fm_score(X)
        if H is not None and M is not None:
            s = s + self.hist_scale * self.hist_score(X, H, M)
        return s

    @torch.no_grad()
    def predict(self, X, H, M, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            hb = torch.from_numpy(H[i:i + bs].astype(np.int64)).to(device)
            mb = torch.from_numpy(M[i:i + bs].astype(np.float32)).to(device)
            out.append(self(xb, hb, mb).cpu().numpy())
        return np.concatenate(out)


def make_histories(enc, max_len=20):
    """Build fixed-length prior-positive video histories for each split."""
    pad_id = max(int(enc[sp][0].max()) for sp in enc) + 1
    out = {}

    Xtr, ytr, utr = enc['train']
    hist_by_u = defaultdict(lambda: deque(maxlen=max_len))
    H = np.full((len(Xtr), max_len), pad_id, dtype=np.int32)
    M = np.zeros((len(Xtr), max_len), dtype=np.float32)
    for i in range(len(Xtr)):
        dq = hist_by_u[int(utr[i])]
        if dq:
            vals = list(dq)
            H[i, :len(vals)] = vals
            M[i, :len(vals)] = 1.0
        if ytr[i] > 0.5:
            dq.appendleft(int(Xtr[i, 1]))
    out['train'] = (H, M)

    # Freeze histories after train.  This avoids using validation/test labels.
    frozen = {u: list(dq) for u, dq in hist_by_u.items()}
    for sp in ('valid', 'test'):
        X, y, uarr = enc[sp]
        H = np.full((len(X), max_len), pad_id, dtype=np.int32)
        M = np.zeros((len(X), max_len), dtype=np.float32)
        for i in range(len(X)):
            vals = frozen.get(int(uarr[i]), [])
            if vals:
                n = min(len(vals), max_len)
                H[i, :n] = vals[:n]
                M[i, :n] = 1.0
        out[sp] = (H, M)
    return out, pad_id


def make_pos_user_negpools(y, users):
    pos_by_u = defaultdict(list)
    neg_by_u = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos_by_u[int(uu)].append(i)
        else:
            neg_by_u[int(uu)].append(i)

    pos_idx = []
    pos_user = []
    neg_pools = {}
    for u, ps in pos_by_u.items():
        ns = neg_by_u.get(u)
        if ns:
            neg_pools[u] = np.asarray(ns, dtype=np.int64)
            pos_idx.extend(ps)
            pos_user.extend([u] * len(ps))
    return (np.asarray(pos_idx, dtype=np.int64),
            np.asarray(pos_user, dtype=np.int64),
            neg_pools)


def sample_pairs(pos_idx, pos_user, neg_pools, rng, multiplier=2):
    n_base = len(pos_idx)
    order = rng.permutation(n_base)
    if multiplier > 1:
        order = np.tile(order, multiplier)
        rng.shuffle(order)
    p = pos_idx[order]
    n = np.empty_like(p)
    for j, u in enumerate(pos_user[order]):
        pool = neg_pools[int(u)]
        n[j] = pool[rng.integers(len(pool))]
    return p, n


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        hist_len=20, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    histories, pad_id = make_histories(enc, max_len=hist_len)
    Htr, Mtr = histories['train']
    Hva, Mva = histories['valid']

    model = HistoryFM(dim, pad_id=pad_id, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W, model.hist_scale], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Htr_t = torch.from_numpy(Htr.astype(np.int64))
    Mtr_t = torch.from_numpy(Mtr.astype(np.float32))
    pos_idx, pos_user, neg_pools = make_pos_user_negpools(ytr, utr)
    if len(pos_idx) == 0:
        raise RuntimeError('No same-user positive/negative pairs found for BPR')

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        pidx, nidx = sample_pairs(pos_idx, pos_user, neg_pools, rng, multiplier=2)
        losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i + bs])
            ns = torch.from_numpy(nidx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            hp = Htr_t[ps].to(device)
            hn = Htr_t[ns].to(device)
            mp = Mtr_t[ps].to(device)
            mn = Mtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            diff = model(xp, hp, mp) - model(xn, hn, mn)
            loss = torch.nn.functional.softplus(-diff).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va_scores = model.predict(Xva, Hva, Mva, device=device)
        va = evaluate(uva, yva, va_scores)
        if verbose:
            print(f"  epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | scale {float(model.hist_scale.detach().cpu()):.3f} "
                  f"| pairs {len(pidx):,d} | {time.time() - t0:.1f}s")

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
    return model, enc, histories


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

    model, enc, histories = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                                seed=a.seed, device=a.device,
                                verbose=a.out is None)

    X, y, users = enc[a.split]
    H, M = histories[a.split]
    scores = model.predict(X, H, M, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== din_history_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            Hs, Ms = histories[sp]
            r = evaluate(us, ys, model.predict(Xs, Hs, Ms, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
