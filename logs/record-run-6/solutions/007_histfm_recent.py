"""FM plus online user-history features, with recent positive sequence summaries.

Refines 006: the aggregate history features helped nDCG a little, but DIN-style
interest should be strongest for recent behaviour.  Add target-aware counts in a
user's last positive impressions (author/video/tab/duration) while still using
strictly previous training rows and training-only history for valid/test.
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
        self.ra = defaultdict(lambda: deque(maxlen=50))
        self.rv = defaultdict(lambda: deque(maxlen=50))
        self.rt = defaultdict(lambda: deque(maxlen=50))
        self.rd = defaultdict(lambda: deque(maxlen=50))
        self.ia = defaultdict(lambda: deque(maxlen=20))
        self.iv = defaultdict(lambda: deque(maxlen=20))

    def copy(self):
        other = HistState()
        for name in ('ui', 'up', 'uai', 'uap', 'uvi', 'uvp', 'uti', 'utp',
                     'udi', 'udp', 'ai', 'ap', 'vi', 'vp'):
            setattr(other, name, defaultdict(int, getattr(self, name).copy()))
        for name in ('ra', 'rv', 'rt', 'rd'):
            src = getattr(self, name)
            dst = defaultdict(lambda: deque(maxlen=50))
            for k, v in src.items():
                dst[k] = deque(v, maxlen=50)
            setattr(other, name, dst)
        for name in ('ia', 'iv'):
            src = getattr(self, name)
            dst = defaultdict(lambda: deque(maxlen=20))
            for k, v in src.items():
                dst[k] = deque(v, maxlen=20)
            setattr(other, name, dst)
        return other

    @staticmethod
    def _cnt(seq, val, n):
        if not seq:
            return 0.0
        c = 0
        m = min(len(seq), n)
        # right side is most recent
        for x in list(seq)[-m:]:
            if x == val:
                c += 1
        return float(c)

    @staticmethod
    def _recency(seq, val):
        if not seq:
            return 0.0
        dist = 1
        for x in reversed(seq):
            if x == val:
                return 1.0 / dist
            dist += 1
        return 0.0

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
        ra = self.ra[u]
        rv = self.rv[u]
        rt = self.rt[u]
        rd = self.rd[u]
        ia = self.ia[u]
        iv = self.iv[u]
        return [
            np.log1p(ui), (up + 1.0) / (ui + 2.0),
            np.log1p(uai), (uap + 0.5) / (uai + 2.0), uap / (up + 1.0),
            np.log1p(uvi), (uvp + 0.5) / (uvi + 2.0),
            np.log1p(uti), (utp + 0.5) / (uti + 2.0),
            np.log1p(udi), (udp + 0.5) / (udi + 2.0),
            np.log1p(ai), (ap + 1.0) / (ai + 2.0),
            np.log1p(vi), (vp + 0.5) / (vi + 2.0),
            # recent positive sequence, target-aware
            self._cnt(ra, a, 5), self._cnt(ra, a, 20), self._recency(ra, a),
            self._cnt(rv, v, 20), self._recency(rv, v),
            self._cnt(rt, tab, 5), self._cnt(rt, tab, 20),
            self._cnt(rd, dur, 5), self._cnt(rd, dur, 20),
            np.log1p(len(ra)),
            # recent impressions regardless of label; useful for repeated exposure fatigue/interest
            self._cnt(ia, a, 5), self._cnt(ia, a, 20),
            self._cnt(iv, v, 20), self._recency(iv, v),
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
        self.ia[u].append(a)
        self.iv[u].append(v)
        if y:
            self.ra[u].append(a)
            self.rv[u].append(v)
            self.rt[u].append(tab)
            self.rd[u].append(dur)


def make_history_features(splits):
    state = HistState()
    feats = {}
    sample = state.features_one(splits['train'][0])
    tr = np.empty((len(splits['train']), len(sample)), dtype=np.float32)
    for i, row in enumerate(splits['train']):
        tr[i] = state.features_one(row)
        state.update(row)
    feats['train'] = tr
    train_state = state.copy()
    for sp in ('valid', 'test'):
        st = train_state.copy()
        arr = np.empty((len(splits[sp]), tr.shape[1]), dtype=np.float32)
        for i, row in enumerate(splits[sp]):
            arr[i] = st.features_one(row)
            # Do not use validation/test labels as behaviour history.
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
        self.H = torch.nn.Sequential(
            torch.nn.Linear(hist_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        # Start as the parent FM so early stopping sees only learned improvements.
        torch.nn.init.normal_(self.H[0].weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.H[0].bias)
        torch.nn.init.zeros_(self.H[2].weight)
        torch.nn.init.zeros_(self.H[2].bias)

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
        print(f"\n=== histfm_recent (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, hist[sp], device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
