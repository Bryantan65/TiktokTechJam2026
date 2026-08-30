"""Improve node 19: fuse coarse and rich time members.

Node 18's coarse hour4+dow time member and node 19's richer time member landed
at the same primary with opposite metric tilt.  This keeps all member training
unchanged (and uses their caches), averages the two time views, and gives the
combined time signal a slightly larger but still readable 40% total weight.
"""
import argparse, csv, os, sys, time
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32)); self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    def forward(self, X):
        E = self.V[X]; S = E.sum(1); inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter
    @torch.no_grad()
    def predict(self, X, bs=200000, device='cpu'):
        self.eval(); out=[]
        for i in range(0, len(X), bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def make_pair_sampler(y, users):
    pos, neg = defaultdict(list), defaultdict(list)
    for i, (yy, u) in enumerate(zip(y, users)): (pos if yy > 0.5 else neg)[u].append(i)
    pidx=[]; pusers=[]; neg_by={}
    for u, ps in pos.items():
        ns = neg.get(u)
        if ns: neg_by[u] = np.asarray(ns, dtype=np.int64); pidx.extend(ps); pusers.extend([u] * len(ps))
    return np.asarray(pidx, dtype=np.int64), np.asarray(pusers, dtype=object), neg_by

def train_member(enc, dim, pair_weights=None, loss_type='bpr', k=16, lr=0.001, l2=1e-6, epochs=50, bs=8192, patience=5, seed=0, device='cpu', verbose=True, nneg=3):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; torch.manual_seed(seed)
    model = TorchFM(dim, k, seed).to(device)
    opt = torch.optim.Adam([{'params':[model.V, model.W], 'weight_decay':l2}, {'params':[model.b], 'weight_decay':0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); w_t = None if pair_weights is None else torch.from_numpy(pair_weights.astype(np.float32))
    pos_idx, pos_users, neg_by = make_pair_sampler(ytr, utr)
    if verbose: print(f"member seed={seed}: {loss_type if pair_weights is None else 'gentle-watch-'+loss_type} positives {len(pos_idx):,d}; nneg={nneg}")
    rng = np.random.default_rng(seed); best = -1.0; best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(pos_idx)); model.train(); losses=[]; t0=time.time()
        for i in range(0, len(order), bs):
            sel = order[i:i+bs]; pidx = pos_idx[sel]; nidx = np.empty((len(sel), nneg), dtype=np.int64)
            for j, u in enumerate(pos_users[sel]):
                ns = neg_by[u]; nidx[j] = ns[rng.integers(len(ns), size=nneg)]
            xp = Xtr_t[torch.from_numpy(pidx)].to(device); xn = Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True); sp = model(xp); sn = model(xn).reshape(len(sel), nneg)
            if loss_type == 'softmax':
                logits = torch.cat([sp.reshape(-1, 1), sn], dim=1)
                lv = torch.nn.functional.cross_entropy(logits, torch.zeros(len(sel), dtype=torch.long, device=device), reduction='none')
                loss = lv.mean() if w_t is None else (lv * w_t[torch.from_numpy(pidx)].to(device)).mean()
            else:
                lv = torch.nn.functional.softplus(-(sp.repeat_interleave(nneg) - sn.reshape(-1)))
                loss = lv.mean() if w_t is None else (lv * w_t[torch.from_numpy(pidx)].to(device).repeat_interleave(nneg)).mean()
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
    video = str(row_get(row, ['video_id','item_id','photo_id'], '')); author = str(row_get(row, ['author_id','author'], '')); tab = str(row_get(row, ['tab'], ''))
    return ((date, user, video, author, tab) if author != '' else None), (date, user, video, tab)

def parse_hour(v):
    try:
        x = int(float(v))
        if x > 10000: x = (x // 10000) % 24
        elif x >= 100: x = (x // 100) % 24
        else: x = x % 24
        return int(max(0, min(23, x)))
    except Exception: return 24

def day_of_week(date_int):
    try: return datetime.strptime(str(int(date_int)), '%Y%m%d').weekday()
    except Exception: return 7

def load_hours(data_dir, splits, verbose=False):
    paths = find_log_files(data_dir); out = {k: np.full(len(v), 24, dtype=np.int64) for k, v in splits.items()}
    if paths is None:
        if verbose: print('time features: raw log csvs not found')
        return out
    full_q, short_q = defaultdict(deque), defaultdict(deque)
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                h = parse_hour(row_get(row, ['hourmin','hour_min','hour','time','timestamp'], '')); full, short = make_keys_from_csv(row)
                if full is not None: full_q[full].append(h)
                short_q[short].append(h)
    for sp, rows in splits.items():
        miss = 0
        for i, r in enumerate(rows):
            h = None; fq = full_q.get(make_key_from_tuple(r, True))
            if fq: h = fq.popleft()
            else:
                sq = short_q.get(make_key_from_tuple(r, False))
                if sq: h = sq.popleft()
            if h is None: h = 24; miss += 1
            out[sp][i] = h
        if verbose: print(f'time join {sp}: missing {miss}/{len(rows)}')
    return out

def load_gentle_watch_weights(data_dir, splits, verbose=False):
    ntr = len(splits['train']); paths = find_log_files(data_dir)
    if paths is None: return np.ones(ntr, dtype=np.float32)
    full_q, short_q = defaultdict(deque), defaultdict(deque)
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                play = row_float(row, ['play_time_ms','play_time','watch_time_ms'], 0.0); full, short = make_keys_from_csv(row)
                if full is not None: full_q[full].append(play)
                short_q[short].append(play)
    vals = np.zeros(ntr, dtype=np.float32); miss = 0
    for i, r in enumerate(splits['train']):
        rec = None; fq = full_q.get(make_key_from_tuple(r, True))
        if fq: rec = fq.popleft()
        else:
            sq = short_q.get(make_key_from_tuple(r, False))
            if sq: rec = sq.popleft()
        if rec is None: play = 0.0; miss += 1
        else: play = rec
        dur = float(r[5]) if float(r[5]) > 0 else 0.0
        vals[i] = np.log1p(max(0.0, min(play / dur, 5.0))) if dur > 0 else np.log1p(max(0.0, play) / 1000.0)
    y = np.asarray([r[6] for r in splits['train']], dtype=np.float32); pos = y > 0.5; w = np.ones(ntr, dtype=np.float32); pv = vals[pos]
    if pos.any() and np.isfinite(pv).all() and pv.std() > 1e-8 and miss < 0.2 * ntr:
        z = (pv - pv.mean()) / pv.std(); w[pos] = np.clip(1.0 + 0.15 * z, 0.7, 1.3); w[pos] /= max(float(w[pos].mean()), 1e-8)
    if verbose:
        pp = w[pos] if pos.any() else w; print(f'gentle watch weights: missing={miss}/{ntr} mean={pp.mean():.3f} min={pp.min():.3f} max={pp.max():.3f}')
    return w.astype(np.float32)

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

def encode_coarse_time(splits, hours):
    maps = [dict() for _ in range(7)]; vals_by = {}
    for sp, rows in splits.items():
        vals = []
        hs = hours.get(sp, np.full(len(rows), 24, dtype=np.int64))
        for i, r in enumerate(rows):
            h = int(hs[i]); h4 = 6 if h >= 24 else h // 4
            vals.append((str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(dur_bucket(r[5])), str(h4), str(day_of_week(r[0]))))
        vals_by[sp] = vals
        for vr in vals:
            for j, v in enumerate(vr):
                if v not in maps[j]: maps[j][v] = len(maps[j])
    offsets=[]; off=0
    for m in maps: offsets.append(off); off += len(m)
    enc={}
    for sp, vals in vals_by.items():
        X = np.zeros((len(vals), len(maps)), dtype=np.int64)
        for i, vr in enumerate(vals):
            for j, v in enumerate(vr): X[i, j] = offsets[j] + maps[j][v]
        enc[sp] = (X, np.asarray([r[6] for r in splits[sp]], dtype=np.float32), np.asarray([r[1] for r in splits[sp]], dtype=object))
    return enc, off

def encode_rich_time(splits, hours):
    maps = [dict() for _ in range(10)]; vals_by = {}
    for sp, rows in splits.items():
        vals=[]; hs = hours.get(sp, np.full(len(rows), 24, dtype=np.int64))
        for i, r in enumerate(rows):
            h = int(hs[i]); h4 = 6 if h >= 24 else h // 4; dow = day_of_week(r[0]); date = str(r[0]); tab = str(r[4])
            vals.append((str(r[1]), str(r[2]), str(r[3]), tab, str(dur_bucket(r[5])), str(h), str(h4), str(dow), date, tab + '_' + str(h4)))
        vals_by[sp] = vals
        for vr in vals:
            for j, v in enumerate(vr):
                if v not in maps[j]: maps[j][v] = len(maps[j])
    offsets=[]; off=0
    for m in maps: offsets.append(off); off += len(m)
    enc={}
    for sp, vals in vals_by.items():
        X = np.zeros((len(vals), len(maps)), dtype=np.int64)
        for i, vr in enumerate(vals):
            for j, v in enumerate(vr): X[i, j] = offsets[j] + maps[j][v]
        enc[sp] = (X, np.asarray([r[6] for r in splits[sp]], dtype=np.float32), np.asarray([r[1] for r in splits[sp]], dtype=object))
    return enc, off

def cached_predictions(enc, dim, target, split_name, seed, device, verbose, name, loss_type='bpr', pair_weights=None, nneg=3, k=16, lr=0.001, epochs=50):
    os.makedirs('pred_cache', exist_ok=True); path = os.path.join('pred_cache', f'{name}_seed{seed}_{split_name}_{target}.npy')
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

def run_solution(splits, target, split_name, data_dir, k=16, lr=0.001, epochs=50, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits); users = enc[target][2]; member_seeds = [seed + 1000 * m for m in range(3)]
    plain = [per_user_zscore(cached_predictions(enc, dim, target, split_name, ms, device, verbose, '006_bpr3neg_member', 'bpr', None, 3, k, lr, epochs), users) for ms in member_seeds]
    plain_fused = per_user_zscore(np.mean(np.vstack(plain), axis=0), users)
    weights = load_gentle_watch_weights(data_dir, splits, verbose)
    watch = [per_user_zscore(cached_predictions(enc, dim, target, split_name, ms, device, verbose, '010_gentlewatch_bpr_member', 'bpr', weights, 3, k, lr, epochs), users) for ms in member_seeds]
    watch_fused = per_user_zscore(np.mean(np.vstack(watch), axis=0), users)
    incumbent = per_user_zscore(0.50 * plain_fused + 0.50 * watch_fused, users)
    listwise = [per_user_zscore(cached_predictions(enc, dim, target, split_name, ms, device, verbose, '014_softmax7neg_member', 'softmax', None, 7, k, lr, epochs), users) for ms in member_seeds]
    listwise_fused = per_user_zscore(np.mean(np.vstack(listwise), axis=0), users)
    base14 = per_user_zscore(0.70 * incumbent + 0.30 * listwise_fused, users)

    hours = load_hours(data_dir, splits, verbose)
    cenc, cdim = encode_coarse_time(splits, hours); cusers = cenc[target][2]
    coarse = [per_user_zscore(cached_predictions(cenc, cdim, target, split_name, ms, device, verbose, '018_time_hour_dow_bpr_member', 'bpr', None, 3, k, lr, epochs), cusers) for ms in member_seeds]
    coarse_fused = per_user_zscore(np.mean(np.vstack(coarse), axis=0), users)
    renc, rdim = encode_rich_time(splits, hours); rusers = renc[target][2]
    rich = [per_user_zscore(cached_predictions(renc, rdim, target, split_name, ms, device, verbose, '019_rich_time_bpr_member', 'bpr', None, 3, k, lr, epochs), rusers) for ms in member_seeds]
    rich_fused = per_user_zscore(np.mean(np.vstack(rich), axis=0), users)
    return 0.60 * base14 + 0.20 * coarse_fused + 0.20 * rich_fused

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='./KuaiRand-Pure/data'); ap.add_argument('--split', default='valid', choices=['train','valid','test','dev']); ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args(); torch.manual_seed(a.seed); print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f'fields={FIELDS}+coarse/rich_time_blend')
    scores = run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else: print(f'produced {len(scores):,d} predictions for split={a.split}')
