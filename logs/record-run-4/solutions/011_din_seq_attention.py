"""Time-aware sampled-softmax FM with DIN-style positive-history attention.

Adds a user's recent positive train-history (video, author) sequence.  The main
long_view score is FM + a target-attention score between the current item and
that user's recent positive items, trained with the same within-user sampled
softmax objective as the best time/multitask FM.
"""
import argparse
import csv
import datetime as _dt
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


AUX_NAMES = ('is_click', 'is_like', 'is_follow', 'is_comment',
             'is_forward', 'is_profile_enter', 'is_hate')


class DINSeqFM(torch.nn.Module):
    def __init__(self, dim, k=16, n_aux=0, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.k = k
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.seq_scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
        self.n_aux = int(n_aux)
        if self.n_aux:
            self.W_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dim, dtype=torch.float32))
            self.b_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dtype=torch.float32))

    def _inter(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def _seq_score(self, X, Hv, Ha, Hm):
        # Query = current target video + author; history = previous positive video + author.
        q = self.V[X[:, 1]] + self.V[X[:, 2]]                         # [B,k]
        h = self.V[Hv] + self.V[Ha]                                    # [B,L,k]
        m = Hm.float()
        att = (h * q.unsqueeze(1)).sum(-1) / (self.k ** 0.5)            # [B,L]
        att = att.masked_fill(~Hm, -1.0e4)
        w = torch.softmax(att, dim=1) * m
        w = w / w.sum(1, keepdim=True).clamp_min(1.0e-6)
        ctx = (h * w.unsqueeze(-1)).sum(1)
        return (q * ctx).sum(1)

    def forward(self, X, Hv=None, Ha=None, Hm=None):
        base = self.b + self.W[X].sum(1) + self._inter(X)
        if Hv is None:
            return base
        return base + self.seq_scale * self._seq_score(X, Hv, Ha, Hm)

    def aux_forward(self, X):
        inter = self._inter(X).view(-1, 1)
        lin = self.W_aux[:, X].sum(2).t()
        return self.b_aux.view(1, -1) + lin + inter

    @torch.no_grad()
    def predict(self, X, Hv, Ha, Hm, bs=100_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            hv = torch.from_numpy(Hv[i:i + bs].astype(np.int64)).to(device)
            ha = torch.from_numpy(Ha[i:i + bs].astype(np.int64)).to(device)
            hm = torch.from_numpy(Hm[i:i + bs].astype(bool)).to(device)
            out.append(self(xb, hv, ha, hm).cpu().numpy())
        return np.concatenate(out)


def _to_int(x, default=0):
    try:
        if x is None or x == '':
            return default
        return int(float(x))
    except Exception:
        return default


def _get(row, *names):
    for n in names:
        if n in row:
            return row[n]
    return None


def _dow_from_date(d):
    try:
        return _dt.datetime.strptime(str(int(d)), '%Y%m%d').weekday()
    except Exception:
        return 0


def _raw_maps(data_dir):
    names = ['log_standard_4_08_to_4_21_pure.csv',
             'log_standard_4_22_to_5_08_pure.csv']
    full = defaultdict(deque)
    nodur = defaultdict(deque)
    for name in names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                date = _to_int(_get(r, 'date'))
                user = _to_int(_get(r, 'user_id', 'user'))
                video = _to_int(_get(r, 'video_id', 'item_id', 'photo_id'))
                author = _to_int(_get(r, 'author_id'))
                tab = _to_int(_get(r, 'tab'))
                tab_clip = max(0, min(tab, 7))
                dur = _to_int(_get(r, 'duration_ms', 'duration'))
                lab = _to_int(_get(r, 'long_view', 'label'))
                hm = _to_int(_get(r, 'hourmin', 'hour_min', 'time'))
                hour = (hm // 100) % 24 if hm >= 100 else hm % 24
                dow = _dow_from_date(date)
                htab = hour * 8 + tab_clip
                extra = (hour, dow, htab)
                aux = tuple(float(_to_int(_get(r, n), 0)) for n in AUX_NAMES)
                val = (extra, aux)
                k_full = (date, user, video, author, tab, dur, lab)
                k_nodur = (date, user, video, author, tab, lab)
                full[k_full].append(val)
                nodur[k_nodur].append(val)
    return full, nodur


def _build_histories(enc, L=10):
    seq = {}
    train_hist = defaultdict(list)

    # Training rows get only previous train positives; update after creating features.
    X, y, users = enc['train']
    Hv = np.zeros((len(X), L), dtype=np.int64)
    Ha = np.zeros((len(X), L), dtype=np.int64)
    Hm = np.zeros((len(X), L), dtype=bool)
    for i in range(len(X)):
        u = int(users[i])
        h = train_hist[u]
        if h:
            recent = h[-L:]
            st = L - len(recent)
            Hv[i, st:] = [p[0] for p in recent]
            Ha[i, st:] = [p[1] for p in recent]
            Hm[i, st:] = True
        if y[i] > 0.5:
            h.append((int(X[i, 1]), int(X[i, 2])))
    seq['train'] = (Hv, Ha, Hm)

    # Validation/test do not consume their own labels: both use the frozen train history.
    frozen = train_hist
    for sp in ('valid', 'test'):
        X, y, users = enc[sp]
        Hv = np.zeros((len(X), L), dtype=np.int64)
        Ha = np.zeros((len(X), L), dtype=np.int64)
        Hm = np.zeros((len(X), L), dtype=bool)
        for i in range(len(X)):
            h = frozen.get(int(users[i]), [])
            if h:
                recent = h[-L:]
                st = L - len(recent)
                Hv[i, st:] = [p[0] for p in recent]
                Ha[i, st:] = [p[1] for p in recent]
                Hm[i, st:] = True
        seq[sp] = (Hv, Ha, Hm)
    return seq


def encode_with_time_aux_seq(splits, data_dir, hist_len=10):
    enc0, dim = encode(splits)
    full, nodur = _raw_maps(data_dir)
    sizes = np.array([24, 7, 24 * 8], dtype=np.int64)
    offsets = dim + np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    out = {}
    aux_out = {}
    matched = {}
    for sp, rows in splits.items():
        Xb, y, users = enc0[sp]
        extra = np.zeros((len(rows), len(sizes)), dtype=np.int64)
        aux = np.zeros((len(rows), len(AUX_NAMES)), dtype=np.float32)
        ok = np.zeros(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            date, user, video, author, tab, dur, lab = row
            k_full = (_to_int(date), _to_int(user), _to_int(video),
                      _to_int(author), _to_int(tab), _to_int(dur), _to_int(lab))
            q = full.get(k_full)
            if q:
                e, a = q.popleft()
                extra[i] = e
                aux[i] = a
                ok[i] = True
                continue
            k_nodur = (_to_int(date), _to_int(user), _to_int(video),
                       _to_int(author), _to_int(tab), _to_int(lab))
            q = nodur.get(k_nodur)
            if q:
                e, a = q.popleft()
                extra[i] = e
                aux[i] = a
                ok[i] = True
        Xt = np.concatenate([Xb.astype(np.int64), extra + offsets], axis=1)
        out[sp] = (Xt, y, users)
        aux_out[sp] = aux
        matched[sp] = ok
    seq = _build_histories(out, L=hist_len)
    return out, aux_out, matched, seq, int(dim + sizes.sum())


def make_pair_sampler(y, users):
    y = np.asarray(y)
    users = np.asarray(users)
    order = np.argsort(users, kind='mergesort')
    us = users[order]
    pos_chunks, neg_pools = [], []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and us[j] == us[i]:
            j += 1
        idx = order[i:j]
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_chunks.append(pos.astype(np.int64, copy=False))
            neg = neg.astype(np.int64, copy=False)
            neg_pools.extend([neg] * len(pos))
        i = j
    if not pos_chunks:
        raise RuntimeError('no users with both positive and negative rows')
    return np.concatenate(pos_chunks), np.asarray(neg_pools, dtype=object)


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=4096,
        patience=4, neg_k=8, aux_weight=0.05, hist_len=10, seed=0,
        device='cpu', verbose=True):
    enc, aux, matched, seq, dim = encode_with_time_aux_seq(splits, data_dir, hist_len=hist_len)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Hvtr, Hatr, Hmtr = seq['train']
    Hvva, Hava, Hmva = seq['valid']

    valid_rows = matched['train']
    prev = aux['train'][valid_rows].mean(0) if valid_rows.any() else np.zeros(len(AUX_NAMES))
    keep_aux = np.where((prev > 0.001) & (prev < 0.999))[0]
    if len(keep_aux) == 0:
        aux_weight = 0.0
    aux_tr = aux['train'][:, keep_aux] if len(keep_aux) else np.zeros((len(Xtr), 0), dtype=np.float32)

    model = DINSeqFM(dim, k=k, n_aux=len(keep_aux), seed=seed).to(device)
    params = [{'params': [model.V, model.W, model.seq_scale], 'weight_decay': l2},
              {'params': [model.b], 'weight_decay': 0.0}]
    if len(keep_aux):
        params.append({'params': [model.W_aux], 'weight_decay': l2})
        params.append({'params': [model.b_aux], 'weight_decay': 0.0})
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    Hvtr_t = torch.from_numpy(Hvtr.astype(np.int64))
    Hatr_t = torch.from_numpy(Hatr.astype(np.int64))
    Hmtr_t = torch.from_numpy(Hmtr.astype(bool))
    aux_t = torch.from_numpy(aux_tr.astype(np.float32))
    pos_idx, neg_pools = make_pair_sampler(ytr, utr)
    rng = np.random.default_rng(seed)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(pos_idx))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(perm), bs):
            psel = perm[i:i + bs]
            bsz = len(psel)
            pidx = pos_idx[psel]
            nidx = np.empty((bsz, neg_k), dtype=np.int64)
            for t, pool in enumerate(neg_pools[psel]):
                nidx[t] = pool[rng.integers(len(pool), size=neg_k)]
            cand_idx = np.concatenate([pidx.reshape(-1, 1), nidx], axis=1)

            pidx_t = torch.from_numpy(pidx)
            nflat = nidx.reshape(-1)
            nidx_t = torch.from_numpy(nflat)
            xp = Xtr_t[pidx_t].to(device)
            xpn_hv = Hvtr_t[pidx_t].to(device)
            xpn_ha = Hatr_t[pidx_t].to(device)
            xpn_hm = Hmtr_t[pidx_t].to(device)
            xn = Xtr_t[nidx_t].to(device)
            xnn_hv = Hvtr_t[nidx_t].to(device)
            xnn_ha = Hatr_t[nidx_t].to(device)
            xnn_hm = Hmtr_t[nidx_t].to(device)

            opt.zero_grad(set_to_none=True)
            sp = model(xp, xpn_hv, xpn_ha, xpn_hm).view(bsz, 1)
            sn = model(xn, xnn_hv, xnn_ha, xnn_hm).view(bsz, neg_k)
            logits = torch.cat([sp, sn], dim=1)
            target = torch.zeros(bsz, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)

            if aux_weight > 0.0 and len(keep_aux):
                flat_idx = cand_idx.reshape(-1)
                xc = Xtr_t[torch.from_numpy(flat_idx)].to(device)
                ya = aux_t[torch.from_numpy(flat_idx)].to(device)
                aux_logits = model.aux_forward(xc)
                aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux_logits, ya)
                loss = loss + aux_weight * aux_loss

            loss.backward()
            opt.step()
            losses.append(loss.item())

        va_pred = model.predict(Xva, Hvva, Hava, Hmva, device=device)
        va = evaluate(uva, yva, va_pred)
        if verbose:
            kept = ','.join(AUX_NAMES[i] for i in keep_aux)
            cover = float(Hmva.any(1).mean())
            print(f"  epoch {ep:2d} | din-softmax {np.mean(losses):.4f} scale={model.seq_scale.item():.3f} "
                  f"hist_cover={cover:.3f} aux=[{kept}] | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
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
    return model, enc, seq


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
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}+hour,dow,hour_tab+DIN_hist aux={AUX_NAMES}")

    model, enc, seq = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                          seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[a.split]
    Hv, Ha, Hm = seq[a.split]
    scores = model.predict(X, Hv, Ha, Hm, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== din_seq_attention (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            Hvs, Has, Hms = seq[sp]
            r = evaluate(us, ys, model.predict(Xs, Hvs, Has, Hms, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                  f"| primary {r['primary']:.4f}")
