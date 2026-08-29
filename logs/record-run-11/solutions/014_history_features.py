"""FM with ranking loss, time/multitask signals, plus causal user-history features.

Adds DIN-inspired candidate-specific user behaviour summaries: whether this user
has previously watched/liked this author/video/tab in the training stream.  These
are categorical fields used by the same FM scorer; auxiliary heads remain only a
training regularizer.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

AUX_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
PLAY_COL = 'play_time_ms'


class TorchFMTower(torch.nn.Module):
    def __init__(self, dim, k=16, n_aux=0, use_play=False, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.n_aux = int(n_aux)
        self.use_play = bool(use_play)
        if self.n_aux > 0:
            self.aux_A = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (k, self.n_aux)).astype(np.float32)))
            self.aux_W = torch.nn.Parameter(torch.zeros((dim, self.n_aux), dtype=torch.float32))
            self.aux_b = torch.nn.Parameter(torch.zeros(self.n_aux, dtype=torch.float32))
        if self.use_play:
            self.play_A = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (k, 1)).astype(np.float32)))
            self.play_W = torch.nn.Parameter(torch.zeros((dim, 1), dtype=torch.float32))
            self.play_b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def shared_sum(self, X):
        E = self.V[X]
        return E, E.sum(1)

    def forward(self, X):
        E, S = self.shared_sum(X)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    def aux_logits(self, X):
        if self.n_aux <= 0:
            return None
        _E, S = self.shared_sum(X)
        return S @ self.aux_A + self.aux_W[X].sum(1) + self.aux_b

    def play_pred(self, X):
        if not self.use_play:
            return None
        _E, S = self.shared_sum(X)
        return (S @ self.play_A + self.play_W[X].sum(1) + self.play_b).squeeze(1)

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def _weekday(yyyymmdd):
    try:
        return datetime.strptime(str(int(yyyymmdd)), '%Y%m%d').weekday()
    except Exception:
        return 7


def _bucket(c):
    if c <= 0:
        return 0
    if c == 1:
        return 1
    if c <= 3:
        return 2
    return 3


def _read_raw_queues(data_dir):
    files = [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
             os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]
    q = defaultdict(deque)
    present = set()
    has_play = False
    n = 0
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            cols = set(rd.fieldnames or [])
            if not {'date', 'user_id', 'video_id', 'tab', 'hourmin'}.issubset(cols):
                continue
            present.update([c for c in AUX_COLS if c in cols])
            has_play = has_play or (PLAY_COL in cols)
            for r in rd:
                try:
                    key = (int(r['date']), str(r['user_id']), str(r['video_id']), int(r['tab']))
                    hm = int(float(r['hourmin']))
                    aux = []
                    for c in AUX_COLS:
                        aux.append(1.0 if c in cols and float(r.get(c, 0.0) or 0.0) > 0 else (np.nan if c not in cols else 0.0))
                    play = np.log1p(max(float(r.get(PLAY_COL, '') or 0.0), 0.0)) / 10.0 if PLAY_COL in cols else np.nan
                except Exception:
                    continue
                q[key].append((hm, tuple(aux), play))
                n += 1
    active = [c for c in AUX_COLS if c in present]
    active_idx = [AUX_COLS.index(c) for c in active]
    print(f"raw rows loaded: {n:,d} keys={len(q):,d} aux={active} play={has_play}")
    return q, active, active_idx, has_play


def _make_hist_features(splits, offsets):
    # Six categorical fields, four buckets each:
    # previous positive/negative count for same author, same video, same tab.
    apos = defaultdict(lambda: defaultdict(int)); aneg = defaultdict(lambda: defaultdict(int))
    vpos = defaultdict(lambda: defaultdict(int)); vneg = defaultdict(lambda: defaultdict(int))
    tpos = defaultdict(lambda: defaultdict(int)); tneg = defaultdict(lambda: defaultdict(int))

    def compute(rows, update):
        H = np.empty((len(rows), 6), dtype=np.int64)
        for i, row in enumerate(rows):
            _date, uid, vid, aid, tab, _dur, lab = row
            u, v, a, t = str(uid), str(vid), str(aid), int(tab)
            vals = [_bucket(apos[u][a]), _bucket(aneg[u][a]),
                    _bucket(vpos[u][v]), _bucket(vneg[u][v]),
                    _bucket(tpos[u][t]), _bucket(tneg[u][t])]
            for j, val in enumerate(vals):
                H[i, j] = offsets[j] + val
            if update:
                if lab > 0.5:
                    apos[u][a] += 1; vpos[u][v] += 1; tpos[u][t] += 1
                else:
                    aneg[u][a] += 1; vneg[u][v] += 1; tneg[u][t] += 1
        return H

    out = {}
    if 'train' in splits:
        out['train'] = compute(splits['train'], update=True)
    for sp in ('valid', 'test'):
        if sp in splits:
            out[sp] = compute(splits[sp], update=False)
    return out


def augment_features_and_aux(splits, enc, dim, data_dir):
    raw_q, active, active_idx, has_play = _read_raw_queues(data_dir)
    off_week = dim
    off_hour = off_week + 8
    off_part = off_hour + 25
    hist_offsets = [off_part + 7 + 4 * j for j in range(6)]
    new_dim = off_part + 7 + 24
    hist = _make_hist_features(splits, hist_offsets)
    new_enc, aux_out, play_out = {}, {}, {}
    total = matched = 0
    nz_hist = 0
    for sp, (X, y, users) in enc.items():
        rows = splits[sp]
        extra = np.empty((len(rows), 3), dtype=np.int64)
        aux = np.zeros((len(rows), len(active)), dtype=np.float32)
        mask = np.zeros((len(rows), len(active)), dtype=np.float32)
        play = np.zeros(len(rows), dtype=np.float32)
        pmask = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            date, uid, vid, _aid, tab, _dur, _lab = row
            wd = _weekday(date)
            key = (int(date), str(uid), str(vid), int(tab))
            hm, avals, pval = None, None, np.nan
            if key in raw_q and raw_q[key]:
                hm, avals, pval = raw_q[key].popleft(); matched += 1
            total += 1
            if hm is None:
                hour, part = 24, 6
            else:
                hour = int(hm) // 100
                if hour < 0 or hour > 23:
                    hour, part = 24, 6
                else:
                    part = hour // 4
            if wd < 0 or wd > 6:
                wd = 7
            extra[i, 0] = off_week + wd
            extra[i, 1] = off_hour + hour
            extra[i, 2] = off_part + part
            if avals is not None and active_idx:
                for j, ai in enumerate(active_idx):
                    v = avals[ai]
                    if not np.isnan(v):
                        aux[i, j] = v; mask[i, j] = 1.0
            if not np.isnan(pval):
                play[i] = pval; pmask[i] = 1.0
        H = hist[sp]
        nz_hist += int(((H - np.asarray(hist_offsets, dtype=np.int64)[None, :]) > 0).any(axis=1).sum())
        new_enc[sp] = (np.concatenate([X.astype(np.int64), extra, H], axis=1), y, users)
        aux_out[sp] = (aux, mask)
        play_out[sp] = (play, pmask)
    print(f"raw matched {matched:,d}/{total:,d} ({matched / max(total,1):.3%}); history_nonzero_rows={nz_hist:,d}; dim {dim}->{new_dim}")
    return new_enc, new_dim, aux_out, play_out, active, has_play


def build_user_groups(users, y):
    pos_by_u, neg_by_u = defaultdict(list), defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        (pos_by_u if yy > 0.5 else neg_by_u)[u].append(i)
    groups = []
    for u in pos_by_u.keys():
        if u in neg_by_u:
            groups.append((np.asarray(pos_by_u[u], dtype=np.int64), np.asarray(neg_by_u[u], dtype=np.int64)))
    return groups


def make_pairs(groups, rng, train_scores=None, semi_k=4, semi_frac=0.25, lambda_alpha=2.5):
    left, right, wts = [], [], []
    order_groups = np.arange(len(groups)); rng.shuffle(order_groups)
    disc5 = np.asarray([1.0 / np.log2(i + 2.0) for i in range(5)], dtype=np.float32)
    for gi in order_groups:
        ps, ns = groups[gi]
        m = len(ps)
        chosen = rng.choice(ns, size=m, replace=True)
        if train_scores is not None and semi_k > 1 and semi_frac > 0:
            mask = rng.random(m) < semi_frac; hh = int(mask.sum())
            if hh > 0:
                psel = ps[mask]
                cand = rng.choice(ns, size=(hh, semi_k), replace=True)
                cs, pscores = train_scores[cand], train_scores[psel]
                ok = cs < pscores[:, None]
                any_ok = ok.any(axis=1)
                if any_ok.any():
                    masked = np.where(ok, cs, -np.inf)
                    rows = np.where(any_ok)[0]; cols = np.argmax(masked[rows], axis=1)
                    tmp = chosen[mask]; tmp[rows] = cand[rows, cols]; chosen[mask] = tmp
        weights = np.ones(m, dtype=np.float32)
        if train_scores is not None and lambda_alpha > 0:
            all_idx = np.concatenate([ps, ns])
            order = np.argsort(-train_scores[all_idx], kind='mergesort')
            ranks = np.empty(len(all_idx), dtype=np.int32); ranks[order] = np.arange(1, len(all_idx) + 1, dtype=np.int32)
            local_rank = dict(zip(all_idx.tolist(), ranks.tolist()))
            rp = np.fromiter((local_rank[int(x)] for x in ps), dtype=np.int32, count=m)
            rn = np.fromiter((local_rank[int(x)] for x in chosen), dtype=np.int32, count=m)
            dp = np.where(rp <= 5, 1.0 / np.log2(rp.astype(np.float32) + 1.0), 0.0)
            dn = np.where(rn <= 5, 1.0 / np.log2(rn.astype(np.float32) + 1.0), 0.0)
            idcg = float(disc5[:min(len(ps), 5)].sum())
            if idcg > 0:
                weights = (1.0 + lambda_alpha * np.abs(dp - dn) / idcg).astype(np.float32)
        left.append(ps); right.append(chosen.astype(np.int64, copy=False)); wts.append(weights)
    p, n, w = np.concatenate(left), np.concatenate(right), np.concatenate(wts)
    perm = rng.permutation(len(p))
    return p[perm], n[perm], w[perm]


def extra_loss_fn(model, xb, ab, mb, pb, pmb):
    losses = []
    if model.n_aux > 0 and ab.shape[1] > 0:
        logits = model.aux_logits(xb)
        loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(logits, ab, reduction='none')
        denom = mb.sum()
        if denom.item() > 0:
            losses.append((loss_mat * mb).sum() / denom)
    if model.use_play:
        denom = pmb.sum()
        if denom.item() > 0:
            pred = model.play_pred(xb)
            losses.append(((pred - pb) ** 2 * pmb).sum() / denom)
    if not losses:
        return None
    return sum(losses) / len(losses)


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4,
        seed=0, device='cpu', verbose=True, bce_warmup=1, bce_weight=0.15,
        aux_weight=0.05, semi_k=4, semi_frac=0.25, lambda_alpha=2.5):
    enc0, dim0 = encode(splits)
    enc, dim, aux, play, active_aux, has_play = augment_features_and_aux(splits, enc0, dim0, data_dir)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    atr, mtr = aux['train']; ptr, pmtr = play['train']
    model = TorchFMTower(dim, k=k, n_aux=len(active_aux), use_play=has_play, seed=seed).to(device)
    params_decay = [model.V, model.W]; params_nodecay = [model.b]
    if len(active_aux) > 0:
        params_decay += [model.aux_A, model.aux_W]; params_nodecay += [model.aux_b]
    if has_play:
        params_decay += [model.play_A, model.play_W]; params_nodecay += [model.play_b]
    opt = torch.optim.Adam([{'params': params_decay, 'weight_decay': l2}, {'params': params_nodecay, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    bce = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    atr_t = torch.from_numpy(atr.astype(np.float32)); mtr_t = torch.from_numpy(mtr.astype(np.float32))
    ptr_t = torch.from_numpy(ptr.astype(np.float32)); pmtr_t = torch.from_numpy(pmtr.astype(np.float32))
    groups = build_user_groups(utr, ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); losses = []
        if ep <= bce_warmup:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), bs):
                sel = torch.from_numpy(idx[i:i + bs])
                xb = Xtr_t[sel].to(device); yb = ytr_t[sel].to(device)
                opt.zero_grad(set_to_none=True)
                loss = bce(model(xb), yb)
                if aux_weight > 0:
                    al = extra_loss_fn(model, xb, atr_t[sel].to(device), mtr_t[sel].to(device), ptr_t[sel].to(device), pmtr_t[sel].to(device))
                    if al is not None: loss = loss + aux_weight * al
                loss.backward(); opt.step(); losses.append(loss.item())
        else:
            train_scores = model.predict(Xtr, device=device)
            pidx, nidx, pair_w = make_pairs(groups, rng, train_scores=train_scores, semi_k=semi_k, semi_frac=semi_frac, lambda_alpha=lambda_alpha)
            for i in range(0, len(pidx), bs):
                ps_np, ns_np = pidx[i:i + bs], nidx[i:i + bs]
                ps = torch.from_numpy(ps_np); ns = torch.from_numpy(ns_np)
                wt = torch.from_numpy(pair_w[i:i + bs]).to(device)
                xp = Xtr_t[ps].to(device); xn = Xtr_t[ns].to(device)
                opt.zero_grad(set_to_none=True)
                sp, sn = model(xp), model(xn)
                bpr_loss = (torch.nn.functional.softplus(-(sp - sn)) * wt).sum() / (wt.sum() + 1e-8)
                loss = bpr_loss + bce_weight * bce(torch.cat([sp, sn]), torch.cat([torch.ones_like(sp), torch.zeros_like(sn)]))
                if aux_weight > 0:
                    both = torch.from_numpy(np.concatenate([ps_np, ns_np]))
                    xb_aux = Xtr_t[both].to(device)
                    al = extra_loss_fn(model, xb_aux, atr_t[both].to(device), mtr_t[both].to(device), ptr_t[both].to(device), pmtr_t[both].to(device))
                    if al is not None: loss = loss + aux_weight * al
                loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} hist_seq | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
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
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+weekday+hour+daypart+hist6 aux={AUX_COLS}+{PLAY_COL}")
    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== history_features (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            Xs, ys, us = enc[sp]
            r = evaluate(us, ys, model.predict(Xs, device=a.device))
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
