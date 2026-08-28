"""FM plus online user-history features.

A lightweight DIN-inspired sequence draft: instead of representing a user only by
an id embedding, add target-aware summaries of the user's previous positive
behaviour for the current author/video/tab/duration bucket.  Training rows use
strictly previous rows; validation/test rows use histories accumulated from the
training split only.
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


class HistState:
    def __init__(self):
        self.ui = defaultdict(int)
        self.up = defaultdict(int)
        self.uai = defaultdict(int)
        self.uap = defaultdict(int)
        self.uvi = defaultdict(int)
        self.uvp = defaultdict(int)
        self.uti = defaultdict(int)
        self.utp = defaultdict(int)
        self.udi = defaultdict(int)
        self.udp = defaultdict(int)
        self.ai = defaultdict(int)
        self.ap = defaultdict(int)
        self.vi = defaultdict(int)
        self.vp = defaultdict(int)

    def copy(self):
        other = HistState()
        for name in ('ui', 'up', 'uai', 'uap', 'uvi', 'uvp', 'uti', 'utp',
                     'udi', 'udp', 'ai', 'ap', 'vi', 'vp'):
            setattr(other, name, defaultdict(int, getattr(self, name).copy()))
        return other

    def features_one(self, row):
        u = row[1]
        v = row[2]
        a = row[3]
        tab = row[4]
        dur = int(row[5]) // 10000
        ui = self.ui[u]
        up = self.up[u]
        uai = self.uai[(u, a)]
        uap = self.uap[(u, a)]
        uvi = self.uvi[(u, v)]
        uvp = self.uvp[(u, v)]
        uti = self.uti[(u, tab)]
        utp = self.utp[(u, tab)]
        udi = self.udi[(u, dur)]
        udp = self.udp[(u, dur)]
        ai = self.ai[a]
        ap = self.ap[a]
        vi = self.vi[v]
        vp = self.vp[v]
        return [
            np.log1p(ui), (up + 1.0) / (ui + 2.0),
            np.log1p(uai), (uap + 0.5) / (uai + 2.0), uap / (up + 1.0),
            np.log1p(uvi), (uvp + 0.5) / (uvi + 2.0),
            np.log1p(uti), (utp + 0.5) / (uti + 2.0),
            np.log1p(udi), (udp + 0.5) / (udi + 2.0),
            np.log1p(ai), (ap + 1.0) / (ai + 2.0),
            np.log1p(vi), (vp + 0.5) / (vi + 2.0),
        ]

    def update(self, row):
        u = row[1]
        v = row[2]
        a = row[3]
        tab = row[4]
        dur = int(row[5]) // 10000
        y = 1 if row[6] > 0 else 0
        self.ui[u] += 1
        self.up[u] += y
        self.uai[(u, a)] += 1
        self.uap[(u, a)] += y
        self.uvi[(u, v)] += 1
        self.uvp[(u, v)] += y
        self.uti[(u, tab)] += 1
        self.utp[(u, tab)] += y
        self.udi[(u, dur)] += 1
        self.udp[(u, dur)] += y
        self.ai[a] += 1
        self.ap[a] += y
        self.vi[v] += 1
        self.vp[v] += y


def make_history_features(splits):
    state = HistState()
    feats = {}
    tr = np.empty((len(splits['train']), 15), dtype=np.float32)
    for i, row in enumerate(splits['train']):
        tr[i] = state.features_one(row)
        state.update(row)
    feats['train'] = tr
    train_state = state.copy()
    for sp in ('valid', 'test'):
        st = train_state.copy()
        arr = np.empty((len(splits[sp]), 15), dtype=np.float32)
        for i, row in enumerate(splits[sp]):
            arr[i] = st.features_one(row)
            # Only training labels are part of the available behaviour history.
        feats[sp] = arr
    mu = feats['train'].mean(axis=0)
    sd = feats['train'].std(axis=0) + 1e-6
    for sp in feats:
        feats[sp] = (feats[sp] - mu) / sd
    return feats


class HistFM(torch.nn.Module):
    def __init__(self, dim, hist_dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.H = torch.nn.Linear(hist_dim, 1)
        torch.nn.init.zeros_(self.H.weight)
        torch.nn.init.zeros_(self.H.bias)

    def forward(self, X, H):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm = self.b + self.W[X].sum(1) + inter
        return fm + self.H(H).squeeze(1)

    @torch.no_grad()
    def predict(self, X, H, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            hb = torch.from_numpy(H[i:i + bs].astype(np.float32)).to(device)
            out.append(self(xb, hb).cpu().numpy())
        return np.concatenate(out)


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    hist = make_history_features(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Htr = hist['train']
    Hva = hist['valid']

    model = HistFM(dim, Htr.shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0},
                            {'params': model.H.parameters(), 'weight_decay': 1e-5}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    lossfn = torch.nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Htr_t = torch.from_numpy(Htr.astype(np.float32))
    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32))

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
            hb = Htr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb, hb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, Hva, device=device))
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
    return model, enc, hist


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

    model, enc, hist = run(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                           device=a.device, verbose=a.out is None)

    X, y, users = enc[a.split]
    scores = model.predict(X, hist[a.split], device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== histfm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, hist[sp], device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
