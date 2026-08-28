"""Sampled-softmax multitask FM plus user-history affinities, selecting epochs on the final adjusted score."""
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


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, k=16, n_aux=0, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.n_aux = int(n_aux)
        if self.n_aux:
            self.W_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dim, dtype=torch.float32))
            self.b_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dtype=torch.float32))

    def _inter(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def forward(self, X):
        return self.b + self.W[X].sum(1) + self._inter(X)

    def aux_forward(self, X):
        inter = self._inter(X).view(-1, 1)
        lin = self.W_aux[:, X].sum(2).t()
        return self.b_aux.view(1, -1) + lin + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
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
                extra = (hour, dow, hour * 8 + tab_clip)
                aux = tuple(float(_to_int(_get(r, n), 0)) for n in AUX_NAMES)
                val = (extra, aux)
                full[(date, user, video, author, tab, dur, lab)].append(val)
                nodur[(date, user, video, author, tab, lab)].append(val)
    return full, nodur


def encode_with_time_aux(splits, data_dir):
    enc, dim = encode(splits)
    full, nodur = _raw_maps(data_dir)
    sizes = np.array([24, 7, 24 * 8], dtype=np.int64)
    offsets = dim + np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    out = {}
    aux_out = {}
    matched = {}
    for sp, rows in splits.items():
        Xb, y, users = enc[sp]
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
        out[sp] = (np.concatenate([Xb.astype(np.int64), extra + offsets], axis=1), y, users)
        aux_out[sp] = aux
        matched[sp] = ok
    return out, aux_out, matched, int(dim + sizes.sum())


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


def _clip01(x):
    return min(max(float(x), 1e-4), 1.0 - 1e-4)


def _logit(p):
    p = _clip01(p)
    return np.log(p / (1.0 - p))


def build_history_stats(train_rows):
    up = defaultdict(float); uc = defaultdict(float)
    uap = defaultdict(float); uac = defaultdict(float)
    uvp = defaultdict(float); uvc = defaultdict(float)
    utp = defaultdict(float); utc = defaultdict(float)
    gp = 0.0; gc = 0.0
    for r in train_rows:
        user = _to_int(r[1]); video = _to_int(r[2]); author = _to_int(r[3])
        tab = _to_int(r[4]); lab = float(_to_int(r[6]))
        gp += lab; gc += 1.0
        up[user] += lab; uc[user] += 1.0
        ka = (user, author); kv = (user, video); kt = (user, tab)
        uap[ka] += lab; uac[ka] += 1.0
        uvp[kv] += lab; uvc[kv] += 1.0
        utp[kt] += lab; utc[kt] += 1.0
    return {'global': gp / max(gc, 1.0), 'up': up, 'uc': uc,
            'uap': uap, 'uac': uac, 'uvp': uvp, 'uvc': uvc, 'utp': utp, 'utc': utc}


def history_adjust(rows, stats):
    g = stats['global']
    out = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        user = _to_int(r[1]); video = _to_int(r[2]); author = _to_int(r[3]); tab = _to_int(r[4])
        uc = stats['uc'].get(user, 0.0)
        ur = (stats['up'].get(user, 0.0) + 8.0 * g) / (uc + 8.0)
        base = _logit(ur)
        adj = 0.0
        ka = (user, author)
        ca = stats['uac'].get(ka, 0.0)
        if ca > 0.0:
            ra = (stats['uap'].get(ka, 0.0) + 2.5 * ur) / (ca + 2.5)
            adj += 0.45 * np.sqrt(ca / (ca + 4.0)) * (_logit(ra) - base)
        kv = (user, video)
        cv = stats['uvc'].get(kv, 0.0)
        if cv > 0.0:
            rv = (stats['uvp'].get(kv, 0.0) + 1.5 * ur) / (cv + 1.5)
            adj += 0.30 * np.sqrt(cv / (cv + 3.0)) * (_logit(rv) - base)
        kt = (user, tab)
        ct = stats['utc'].get(kt, 0.0)
        if ct > 0.0:
            rt = (stats['utp'].get(kt, 0.0) + 4.0 * ur) / (ct + 4.0)
            adj += 0.20 * np.sqrt(ct / (ct + 8.0)) * (_logit(rt) - base)
        out[i] = np.float32(np.clip(adj, -1.0, 1.0))
    return out


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=4096,
        patience=4, neg_k=8, aux_weight=0.05, seed=0, device='cpu', verbose=True):
    enc, aux, matched, dim = encode_with_time_aux(splits, data_dir)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    hist = build_history_stats(splits['train'])
    va_hist = history_adjust(splits['valid'], hist)

    valid_rows = matched['train']
    prev = aux['train'][valid_rows].mean(0) if valid_rows.any() else np.zeros(len(AUX_NAMES))
    keep_aux = np.where((prev > 0.001) & (prev < 0.999))[0]
    if len(keep_aux) == 0:
        aux_weight = 0.0
    aux_tr = aux['train'][:, keep_aux] if len(keep_aux) else np.zeros((len(Xtr), 0), dtype=np.float32)

    model = MultiTaskFM(dim, k=k, n_aux=len(keep_aux), seed=seed).to(device)
    params = [{'params': [model.V, model.W], 'weight_decay': l2},
              {'params': [model.b], 'weight_decay': 0.0}]
    if len(keep_aux):
        params.append({'params': [model.W_aux], 'weight_decay': l2})
        params.append({'params': [model.b_aux], 'weight_decay': 0.0})
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
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

            xp = Xtr_t[torch.from_numpy(pidx)].to(device)
            xn = Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True)
            logits = torch.cat([model(xp).view(bsz, 1), model(xn).view(bsz, neg_k)], dim=1)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(bsz, dtype=torch.long, device=device))

            if aux_weight > 0.0 and len(keep_aux):
                xc = Xtr_t[torch.from_numpy(cand_idx.reshape(-1))].to(device)
                ya = aux_t[torch.from_numpy(cand_idx.reshape(-1))].to(device)
                loss = loss + aux_weight * torch.nn.functional.binary_cross_entropy_with_logits(model.aux_forward(xc), ya)

            loss.backward()
            opt.step()
            losses.append(loss.item())

        pred_va = model.predict(Xva, device=device)
        va_base = evaluate(uva, yva, pred_va)
        va = evaluate(uva, yva, pred_va + va_hist)
        if verbose:
            kept = ','.join(AUX_NAMES[i] for i in keep_aux)
            print(f"  epoch {ep:2d} | mt-softmax {np.mean(losses):.4f} aux=[{kept}] | valid+hist "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
                  f"(base {va_base['primary']:.4f}) | {time.time() - t0:.1f}s")
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
    print({k_: len(v) for k_, v in splits.items()},
          f"fields={FIELDS}+hour,dow,hour_tab aux={AUX_NAMES} history_checkpoint")

    model, enc, hist = run(splits, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs,
                           seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[a.split]
    scores = model.predict(X, device=a.device) + history_adjust(splits[a.split], hist)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== history_checkpoint (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            pred = model.predict(Xs, device=a.device) + history_adjust(splits[sp], hist)
            r = evaluate(us, ys, pred)
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
