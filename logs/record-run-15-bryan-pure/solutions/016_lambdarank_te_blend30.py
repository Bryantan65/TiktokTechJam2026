"""Blend best FM ranking ensemble with a LightGBM LambdaRank target-encoding member.

The added member is deliberately different from the FM members: leakage-safe
(leave-one-out on train) historical CTR/count features for item/author and
user-item/user-author/user-tab preferences, trained with LambdaRank/NDCG@5.
"""
import argparse, csv, os, sys, time
from collections import defaultdict, deque
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate


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
        self.eval(); out = []
        for i in range(0, len(X), bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)


def make_pair_sampler(y, users):
    pos, neg = defaultdict(list), defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        (pos if yy > 0.5 else neg)[uu].append(i)
    pidx, pusers, neg_arrays = [], [], {}
    for u, ps in pos.items():
        ns = neg.get(u)
        if ns:
            neg_arrays[u] = np.asarray(ns, dtype=np.int64)
            pidx.extend(ps); pusers.extend([u] * len(ps))
    return np.asarray(pidx, dtype=np.int64), np.asarray(pusers, dtype=object), neg_arrays


def train_member(enc, dim, pair_weights=None, loss_type='bpr', k=16, lr=0.001, l2=1e-6, epochs=50, bs=8192, patience=5, seed=0, device='cpu', verbose=True, nneg=3):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    torch.manual_seed(seed)
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params':[model.V, model.W], 'weight_decay':l2}, {'params':[model.b], 'weight_decay':0.0}], lr=lr, betas=(0.9,0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    w_t = None if pair_weights is None else torch.from_numpy(pair_weights.astype(np.float32))
    pos_idx, pos_users, neg_by_u = make_pair_sampler(ytr, utr)
    if len(pos_idx) == 0: raise RuntimeError('no same-user positive/negative training pairs found')
    if verbose:
        kind = loss_type if pair_weights is None else ('gentle-watch-' + loss_type)
        print(f"member seed={seed}: {kind} positives {len(pos_idx):,d}; nneg={nneg}")
    rng = np.random.default_rng(seed); best = -1.0; best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(pos_idx)); t0 = time.time(); model.train(); losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i+bs]; pidx = pos_idx[sel]
            nidx = np.empty((len(sel), nneg), dtype=np.int64)
            for j, u in enumerate(pos_users[sel]):
                ns = neg_by_u[u]; nidx[j] = ns[rng.integers(len(ns), size=nneg)]
            xp = Xtr_t[torch.from_numpy(pidx)].to(device)
            xn = Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model(xp)
            sn = model(xn).reshape(len(sel), nneg)
            if loss_type == 'softmax':
                logits = torch.cat([sp.reshape(-1, 1), sn], dim=1)
                loss_vec = torch.nn.functional.cross_entropy(logits, torch.zeros(len(sel), dtype=torch.long, device=device), reduction='none')
                loss = loss_vec.mean() if w_t is None else (loss_vec * w_t[torch.from_numpy(pidx)].to(device)).mean()
            else:
                loss_vec = torch.nn.functional.softplus(-(sp.repeat_interleave(nneg) - sn.reshape(-1)))
                loss = loss_vec.mean() if w_t is None else (loss_vec * w_t[torch.from_numpy(pidx)].to(device).repeat_interleave(nneg)).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f"  member {seed} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model


def row_get(row, names, default=''):
    for n in names:
        if n in row and row[n] not in ('', 'None', 'nan'): return row[n]
    return default

def row_float(row, names, default=0.0):
    try: return float(row_get(row, names, ''))
    except Exception: return default

def find_log_files(data_dir):
    names = ['log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv']
    for base in [data_dir, os.path.join(data_dir, 'data'), os.path.dirname(data_dir)]:
        paths = [os.path.join(base, n) for n in names]
        if all(os.path.isfile(p) for p in paths): return paths
    return None

def make_key_from_tuple(r, with_author=True):
    return (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])) if with_author else (str(r[0]), str(r[1]), str(r[2]), str(r[4]))

def make_keys_from_csv(row):
    date = str(row_get(row, ['date','day'], '')); user = str(row_get(row, ['user_id','user'], ''))
    video = str(row_get(row, ['video_id','item_id','photo_id'], '')); author = str(row_get(row, ['author_id','author'], ''))
    tab = str(row_get(row, ['tab'], ''))
    return ((date, user, video, author, tab) if author != '' else None), (date, user, video, tab)

def load_gentle_watch_weights(data_dir, splits, verbose=False):
    ntr = len(splits['train']); paths = find_log_files(data_dir)
    if paths is None: return np.ones(ntr, dtype=np.float32)
    full_q, short_q = defaultdict(deque), defaultdict(deque); rid = 0
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                play = row_float(row, ['play_time_ms','play_time','watch_time_ms'], 0.0)
                full, short = make_keys_from_csv(row); rec = (rid, play)
                if full is not None: full_q[full].append(rec)
                short_q[short].append(rec); rid += 1
    vals = np.zeros(ntr, dtype=np.float32); miss = 0
    for i, r in enumerate(splits['train']):
        rec = None; fq = full_q.get(make_key_from_tuple(r, True))
        if fq: rec = fq.popleft()
        else:
            sq = short_q.get(make_key_from_tuple(r, False))
            if sq: rec = sq.popleft()
        if rec is None: miss += 1; play = 0.0
        else: play = rec[1]
        dur = float(r[5]) if float(r[5]) > 0 else 0.0
        vals[i] = np.log1p(max(0.0, min(play / dur, 5.0))) if dur > 0 else np.log1p(max(0.0, play) / 1000.0)
    y = np.asarray([r[6] for r in splits['train']], dtype=np.float32); pos = y > 0.5
    w = np.ones(ntr, dtype=np.float32); pv = vals[pos]
    if pos.any() and np.isfinite(pv).all() and pv.std() > 1e-8 and miss < 0.2 * ntr:
        z = (pv - pv.mean()) / pv.std(); w[pos] = np.clip(1.0 + 0.15 * z, 0.7, 1.3); w[pos] /= max(float(w[pos].mean()), 1e-8)
    if verbose:
        pp = w[pos] if pos.any() else w; print(f'gentle watch weights: missing={miss}/{ntr} mean={pp.mean():.3f} min={pp.min():.3f} max={pp.max():.3f}')
    return w.astype(np.float32)


def cached_fm_predictions(enc, dim, target, split_name, seed, device, verbose, name, loss_type='bpr', pair_weights=None, nneg=3, k=16, lr=0.001, epochs=50):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'{name}_seed{seed}_{split_name}_{target}.npy')
    if os.path.isfile(path):
        if verbose: print(f'loading cached member: {path}')
        return np.load(path)
    model = train_member(enc, dim, pair_weights=pair_weights, loss_type=loss_type, k=k, lr=lr, epochs=epochs, seed=seed, device=device, verbose=verbose, nneg=nneg)
    preds = model.predict(enc[target][0], device=device).astype(np.float64); np.save(path, preds); return preds


def per_user_zscore(scores, users):
    scores = scores.astype(np.float64, copy=True); groups = defaultdict(list)
    for i, u in enumerate(users): groups[u].append(i)
    for idxs in groups.values():
        idx = np.asarray(idxs, dtype=np.int64); vals = scores[idx]; sd = vals.std()
        scores[idx] = (vals - vals.mean()) / sd if sd > 1e-12 else 0.0
    return scores


def dur_bucket(ms):
    try: x = float(ms)
    except Exception: x = 0.0
    if x < 5000: return 0
    if x < 10000: return 1
    if x < 20000: return 2
    if x < 30000: return 3
    if x < 60000: return 4
    if x < 120000: return 5
    return 6


def build_stat_tables(train_rows):
    specs = [
        ('video', lambda r: ('v', str(r[2])), 30.0),
        ('author', lambda r: ('a', str(r[3])), 50.0),
        ('tab', lambda r: ('t', str(r[4])), 200.0),
        ('dur', lambda r: ('d', dur_bucket(r[5])), 200.0),
        ('author_tab', lambda r: ('at', str(r[3]), str(r[4])), 40.0),
        ('video_tab', lambda r: ('vt', str(r[2]), str(r[4])), 20.0),
        ('user_author', lambda r: ('ua', str(r[1]), str(r[3])), 15.0),
        ('user_video', lambda r: ('uv', str(r[1]), str(r[2])), 10.0),
        ('user_tab', lambda r: ('ut', str(r[1]), str(r[4])), 30.0),
        ('user_dur', lambda r: ('ud', str(r[1]), dur_bucket(r[5])), 30.0),
    ]
    y = np.asarray([r[6] for r in train_rows], dtype=np.float32); g = float(y.mean())
    tables = []
    for name, fn, alpha in specs:
        d = defaultdict(lambda: [0.0, 0])
        for r, yy in zip(train_rows, y):
            e = d[fn(r)]; e[0] += float(yy); e[1] += 1
        tables.append((name, fn, alpha, d))
    return y, g, tables


def make_te_features(rows, train_y, global_mean, tables, train_mode=False):
    n = len(rows); feats = []
    # Cheap raw/context columns first.
    raw = np.zeros((n, 5), dtype=np.float32)
    for i, r in enumerate(rows):
        try: date = int(r[0])
        except Exception: date = 0
        try: dur = float(r[5])
        except Exception: dur = 0.0
        raw[i, 0] = float(r[4]) if str(r[4]).lstrip('-').isdigit() else 0.0
        raw[i, 1] = float(dur_bucket(dur))
        raw[i, 2] = np.log1p(max(dur, 0.0)) / 12.0
        raw[i, 3] = (date % 100) / 31.0
        raw[i, 4] = 1.0
    feats.append(raw)
    for name, fn, alpha, table in tables:
        arr = np.zeros((n, 2), dtype=np.float32)
        for i, r in enumerate(rows):
            s, c = table.get(fn(r), (0.0, 0))
            if train_mode:
                yy = float(train_y[i]); s -= yy; c -= 1
            if c < 0: c = 0; s = 0.0
            ctr = (s + alpha * global_mean) / (c + alpha)
            arr[i, 0] = float(ctr)
            arr[i, 1] = np.log1p(float(c)) / 8.0
        feats.append(arr)
    return np.hstack(feats).astype(np.float32)


def order_by_user(users):
    d = defaultdict(list)
    for i, u in enumerate(users): d[u].append(i)
    order = [] ; groups = []
    for idxs in d.values():
        groups.append(len(idxs)); order.extend(idxs)
    return np.asarray(order, dtype=np.int64), groups


def train_lgbm_ranker_predict(splits, target, split_name, seed=0, verbose=True):
    os.makedirs('pred_cache', exist_ok=True)
    path = os.path.join('pred_cache', f'016_lgbm_lambdarank_te_seed{seed}_{split_name}_{target}.npy')
    if os.path.isfile(path):
        if verbose: print(f'loading cached member: {path}')
        return np.load(path)
    import lightgbm as lgb
    ytr, global_mean, tables = build_stat_tables(splits['train'])
    Xtr = make_te_features(splits['train'], ytr, global_mean, tables, train_mode=True)
    yva = np.asarray([r[6] for r in splits['valid']], dtype=np.int32)
    Xva = make_te_features(splits['valid'], ytr, global_mean, tables, train_mode=False)
    utr = [str(r[1]) for r in splits['train']]; uva = [str(r[1]) for r in splits['valid']]
    otr, gtr = order_by_user(utr); ova, gva = order_by_user(uva)
    Xtr_s = Xtr[otr]; ytr_s = ytr[otr].astype(np.int32)
    Xva_s = Xva[ova]; yva_s = yva[ova]
    ranker = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5], boosting_type='gbdt',
        n_estimators=1200, learning_rate=0.035, num_leaves=63, max_depth=-1,
        min_child_samples=80, subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=-1,
        label_gain=[0, 1]
    )
    callbacks = [lgb.early_stopping(60, verbose=verbose), lgb.log_evaluation(50 if verbose else 0)]
    ranker.fit(Xtr_s, ytr_s, group=gtr, eval_set=[(Xva_s, yva_s)], eval_group=[gva], eval_at=[5], callbacks=callbacks)
    Xt = make_te_features(splits[target], ytr, global_mean, tables, train_mode=False)
    preds = ranker.predict(Xt, num_iteration=ranker.best_iteration_).astype(np.float64)
    np.save(path, preds)
    return preds


def run_solution(splits, target, split_name, data_dir, k=16, lr=0.001, epochs=50, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits); users = enc[target][2]; member_seeds = [seed + 1000 * m for m in range(3)]
    plain = [per_user_zscore(cached_fm_predictions(enc, dim, target, split_name, ms, device, verbose, '006_bpr3neg_member', 'bpr', None, 3, k, lr, epochs), users) for ms in member_seeds]
    plain_fused = per_user_zscore(np.mean(np.vstack(plain), axis=0), users)
    weights = load_gentle_watch_weights(data_dir, splits, verbose=verbose)
    watch = [per_user_zscore(cached_fm_predictions(enc, dim, target, split_name, ms, device, verbose, '010_gentlewatch_bpr_member', 'bpr', weights, 3, k, lr, epochs), users) for ms in member_seeds]
    watch_fused = per_user_zscore(np.mean(np.vstack(watch), axis=0), users)
    incumbent = per_user_zscore(0.50 * plain_fused + 0.50 * watch_fused, users)
    listwise = [per_user_zscore(cached_fm_predictions(enc, dim, target, split_name, ms, device, verbose, '014_softmax7neg_member', 'softmax', None, 7, k, lr, epochs), users) for ms in member_seeds]
    listwise_fused = per_user_zscore(np.mean(np.vstack(listwise), axis=0), users)
    base14 = per_user_zscore(0.70 * incumbent + 0.30 * listwise_fused, users)
    lgb_preds = [per_user_zscore(train_lgbm_ranker_predict(splits, target, split_name, seed=ms, verbose=verbose), users) for ms in member_seeds]
    lgb_fused = per_user_zscore(np.mean(np.vstack(lgb_preds), axis=0), users)
    return 0.70 * base14 + 0.30 * lgb_fused


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None); ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args(); torch.manual_seed(a.seed); print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f'fields={FIELDS}')
    scores = run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}')
