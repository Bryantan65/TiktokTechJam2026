"""Blend best FM seed bag with user-history preference scores.

Direction: user behaviour sequences.  This keeps the current best node-13
BCE/BPR/rank seed bag unchanged and adds a deterministic per-user history score
computed from train labels only: repeated user-video, user-author, user-tab and
user-duration preferences with Bayesian log-odds.  It is a cheap DIN/SIM-style
probe: the target item is scored by relevance to the user's historical positive
and negative interactions, without touching valid/test labels.
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
from evaluate import evaluate                  # noqa: E402  early stopping only


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
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


def fit_bce(splits, enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, seed=0, device='cpu', verbose=True):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    lossfn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train(); losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BCE(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def make_user_pair_sources(y, users):
    pos = defaultdict(list); neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)
    pos_idx = []; neg_pools = []
    for uu, ps in pos.items():
        ns = neg.get(uu)
        if ns:
            arr = np.asarray(ns, dtype=np.int64)
            for p in ps:
                pos_idx.append(p); neg_pools.append(arr)
    return np.asarray(pos_idx, dtype=np.int64), neg_pools


def sample_uniform_pairs(pos_idx, neg_pools, rng, neg_per_pos=3):
    total = len(pos_idx) * neg_per_pos
    p_out = np.empty(total, dtype=np.int64)
    n_out = np.empty(total, dtype=np.int64)
    k = 0
    for p, pool in zip(pos_idx, neg_pools):
        m = len(pool)
        for _ in range(neg_per_pos):
            p_out[k] = p
            n_out[k] = pool[rng.integers(0, m)]
            k += 1
    order = rng.permutation(total)
    return p_out[order], n_out[order]


def fit_bpr(splits, enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
            patience=4, neg_per_pos=3, seed=0, device='cpu', verbose=True):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_idx, neg_pools = make_user_pair_sources(ytr, utr)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        p_idx, n_idx = sample_uniform_pairs(pos_idx, neg_pools, rng, neg_per_pos=neg_per_pos)
        model.train(); losses = []
        for i in range(0, len(p_idx), bs):
            ps = torch.from_numpy(p_idx[i:i + bs])
            ns = torch.from_numpy(n_idx[i:i + bs])
            xp = Xtr_t[ps].to(device)
            xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(xp) - model(xn))).mean()
            loss.backward(); opt.step()
            losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  BPR(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | pairs {len(p_idx):,d} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def user_groups(users):
    groups = defaultdict(list)
    for i, u in enumerate(users):
        groups[u].append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def per_user_zscore(scores, groups):
    scores = scores.astype(np.float64, copy=False)
    out = np.empty_like(scores, dtype=np.float64)
    for idx in groups:
        s = scores[idx]
        sd = s.std()
        if sd > 1e-12:
            out[idx] = (s - s.mean()) / sd
        else:
            out[idx] = s - s.mean()
    return out


def per_user_rank01(scores, groups):
    scores = scores.astype(np.float64, copy=False)
    out = np.empty_like(scores, dtype=np.float64)
    for idx in groups:
        n = len(idx)
        if n <= 1:
            out[idx] = 0.0
            continue
        order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64) / (n - 1.0)
        out[idx] = ranks
    return out


def cached_predict(member_name, train_fn, enc, target, member_seed, device, use_cache=True):
    os.makedirs('pred_cache', exist_ok=True)
    X, _, _ = enc[target]
    cache_path = os.path.join('pred_cache', f'006_{member_name}_{target}_seed{member_seed}.npy')
    if use_cache and os.path.isfile(cache_path):
        preds = np.load(cache_path)
        if len(preds) == len(X):
            return preds.astype(np.float64, copy=False)
    model = train_fn()
    preds = model.predict(X, device=device).astype(np.float64)
    if use_cache:
        np.save(cache_path, preds)
    return preds


def node8_seed_score(bce_preds, bpr_preds, groups):
    zblend = 0.35 * per_user_zscore(bce_preds, groups) + 0.65 * per_user_zscore(bpr_preds, groups)
    rblend = 0.35 * per_user_rank01(bce_preds, groups) + 0.65 * per_user_rank01(bpr_preds, groups)
    return 0.70 * zblend + 0.30 * rblend


def _logodds(pos, neg, a_pos=1.0, a_neg=1.0):
    return np.log((pos + a_pos) / (neg + a_neg))


def _dur_bucket_ms(x):
    try:
        v = int(float(x))
    except Exception:
        return 'durUNK'
    # Coarse enough to generalize but still captures short-vs-long preference.
    if v < 5_000: return 'dur00'
    if v < 10_000: return 'dur01'
    if v < 20_000: return 'dur02'
    if v < 30_000: return 'dur03'
    if v < 60_000: return 'dur04'
    if v < 120_000: return 'dur05'
    if v < 300_000: return 'dur06'
    return 'dur07'


def history_preference_score(splits, target):
    """Train-label-only user history relevance for each row in target split."""
    # stats[(kind, user, value)] = [pos, neg]
    stats = defaultdict(lambda: [0.0, 0.0])
    user_tot = defaultdict(lambda: [0.0, 0.0])
    author_glob = defaultdict(lambda: [0.0, 0.0])
    video_glob = defaultdict(lambda: [0.0, 0.0])
    for r in splits['train']:
        u, v, a, tab, dur, y = r[1], r[2], r[3], r[4], _dur_bucket_ms(r[5]), float(r[6])
        p = y > 0.5
        pi = 0 if p else 1
        user_tot[u][pi] += 1.0
        for key in (('uv', u, v), ('ua', u, a), ('ut', u, tab), ('ud', u, dur)):
            stats[key][pi] += 1.0
        author_glob[a][pi] += 1.0
        video_glob[v][pi] += 1.0

    rows = splits[target]
    out = np.zeros(len(rows), dtype=np.float64)
    nonzero = 0
    for i, r in enumerate(rows):
        u, v, a, tab, dur = r[1], r[2], r[3], r[4], _dur_bucket_ms(r[5])
        up, un = user_tot.get(u, (0.0, 0.0))
        # Within-user ranking: subtract the user's overall tendency from every
        # user-conditioned component, so known dislikes are negative and known
        # likes are positive for that user.
        ubase = _logodds(up, un, 2.0, 2.0)
        s = 0.0
        touched = False
        for w, key, ap, an in [
            (2.5, ('uv', u, v), 0.5, 0.5),
            (1.2, ('ua', u, a), 1.0, 1.0),
            (0.35, ('ut', u, tab), 2.0, 2.0),
            (0.25, ('ud', u, dur), 2.0, 2.0),
        ]:
            pp, nn = stats.get(key, (0.0, 0.0))
            if pp + nn > 0:
                touched = True
                # shrink rare exact-video repeats less aggressively; they are
                # sparse but highly item-specific.
                shrink = (pp + nn) / (pp + nn + (1.0 if key[0] == 'uv' else 4.0))
                s += w * shrink * (_logodds(pp, nn, ap, an) - ubase)
        # Small global item quality backoff helps cold user-item pairs but is
        # downweighted so this remains primarily a user-history signal.
        gp, gn = video_glob.get(v, (0.0, 0.0))
        if gp + gn >= 3:
            touched = True
            s += 0.20 * ((gp + gn) / (gp + gn + 20.0)) * _logodds(gp, gn, 2.0, 2.0)
        gp, gn = author_glob.get(a, (0.0, 0.0))
        if gp + gn >= 5:
            touched = True
            s += 0.15 * ((gp + gn) / (gp + gn + 30.0)) * _logodds(gp, gn, 2.0, 2.0)
        if touched:
            nonzero += 1
        out[i] = s
    print(f"history score touched rows: {nonzero}/{len(rows)}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--neg_per_pos', type=int, default=3)
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

    enc, dim = encode(splits)
    verbose = a.out is None
    use_cache = a.out is not None and a.split != 'dev'
    X, y, users = enc[target]
    groups = user_groups(users)

    member_seeds = [0, 1, 2, 3, 4]
    weights = np.asarray([0.12, 0.12, 0.12, 0.32, 0.32], dtype=np.float64)
    seed_scores = []
    for ms in member_seeds:
        bce_preds = cached_predict(
            'bce',
            lambda ms=ms: fit_bce(splits, enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                  seed=ms, device=a.device, verbose=verbose),
            enc, target, ms, a.device, use_cache=use_cache)
        bpr_preds = cached_predict(
            'bpr_uniform_np3',
            lambda ms=ms: fit_bpr(splits, enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                  neg_per_pos=a.neg_per_pos, seed=ms,
                                  device=a.device, verbose=verbose),
            enc, target, ms, a.device, use_cache=use_cache)
        s = node8_seed_score(bce_preds, bpr_preds, groups)
        seed_scores.append(per_user_zscore(s, groups))

    score_mat = np.vstack(seed_scores)
    bag = weights @ score_mat
    cur_idx = member_seeds.index(a.seed) if a.seed in member_seeds else (a.seed % len(member_seeds))
    base_scores = 0.995 * bag + 0.005 * seed_scores[cur_idx]

    hist = history_preference_score(splits, target)
    hist_mix = 0.70 * per_user_zscore(hist, groups) + 0.30 * per_user_rank01(hist, groups)
    # New mechanism gets 30% of the vote so its signal is readable.
    scores = 0.70 * per_user_zscore(base_scores, groups) + 0.30 * per_user_zscore(hist_mix, groups)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== history_blend (seed={a.seed}, device={a.device}) ===")
        r = evaluate(users, y, scores)
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
