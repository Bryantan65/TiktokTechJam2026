import argparse, csv, glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, FIELDS
from evaluate import evaluate

DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)

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
        inter = 0.5 * ((S * S).sum(1) - (E * E).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter
    @torch.no_grad()
    def predict(self, X, bs=200000, device='cpu'):
        self.eval(); out = []
        for i in range(0, len(X), bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def norm(x):
    try: return str(int(float(x)))
    except Exception: return str(x)

def row_key(r): return (norm(r[DATE]), norm(r[USER]), norm(r[VIDEO]), norm(r[TAB]))
def raw_key(rec): return (norm(rec.get('date','')), norm(rec.get('user_id','')), norm(rec.get('video_id','')), norm(rec.get('tab','')))

def find_logs(data_dir):
    names = ['log_standard_4_08_to_4_21_1k.csv', 'log_standard_4_22_to_5_08_1k.csv']
    files = [os.path.join(data_dir, n) for n in names]
    if all(os.path.isfile(p) for p in files): return files
    out = []
    for pat in ['log_standard_4_08_to_4_21*.csv', 'log_standard_4_22_to_5_08*.csv']:
        g = sorted(glob.glob(os.path.join(data_dir, pat)))
        if g: out.append(g[0])
    return out

def parse_hourmin(x):
    try: hm = int(float(x))
    except Exception: return -1, -1, -1
    h, m = hm // 100, hm % 100
    if h < 0 or h > 23 or m < 0 or m > 59: return -1, -1, -1
    return h, h * 6 + (m // 10), h // 4

def read_time_ordered(data_dir, rows, name):
    n = len(rows)
    hour = np.full(n, -1, np.int16); ten = np.full(n, -1, np.int16); block = np.full(n, -1, np.int16)
    files = find_logs(data_dir)
    if not files or n == 0:
        print('warning: raw logs not found for', name, flush=True); return hour, ten, block
    i = 0; cur = row_key(rows[0]); seen = 0; t0 = time.time()
    for path in files:
        if i >= n: break
        with open(path, 'r', encoding='utf-8', newline='') as f:
            for rec in csv.DictReader(f):
                seen += 1
                if raw_key(rec) == cur:
                    h, te, bl = parse_hourmin(rec.get('hourmin', ''))
                    hour[i], ten[i], block[i] = h, te, bl
                    i += 1
                    if i >= n: break
                    cur = row_key(rows[i])
    print(f'aligned hourmin for {name}: {i:,d}/{n:,d} rows after scanning {seen:,d} raw rows in {time.time()-t0:.1f}s', flush=True)
    if i < n: print(f'warning: {n-i:,d} {name} rows missing time features', flush=True)
    return hour, ten, block

def dur_bucket(x): return int(np.log1p(float(x)) // 1)

def make_encoded(splits, aux, names):
    maps = [{} for _ in range(14)]
    feats = {}; ys = {}; raw_users = {}
    for sp in names:
        rows = splits[sp]; h, ten, block = aux[sp]
        fs = []; y = np.empty(len(rows), np.float32); ru = []
        for i, r in enumerate(rows):
            u = r[USER]; t = r[TAB]; hh = int(h[i]); te = int(ten[i]); bl = int(block[i])
            vals = [u, r[VIDEO], r[AUTHOR], t, dur_bucket(r[DUR]), r[DATE],
                    hh, te, bl, (t, hh), (t, te), (u, t), (u, hh), (u, bl)]
            fs.append(vals); y[i] = float(r[LABEL]); ru.append(u)
            for j, v in enumerate(vals):
                if v not in maps[j]: maps[j][v] = len(maps[j])
        feats[sp] = fs; ys[sp] = y; raw_users[sp] = ru
    offsets = np.cumsum([0] + [len(m) for m in maps[:-1]]).astype(np.int64)
    dim = int(sum(len(m) for m in maps))
    enc = {}; user_map = {}
    for sp in names:
        fs = feats[sp]; X = np.empty((len(fs), len(maps)), np.int64); u = np.empty(len(fs), np.int64)
        for i, vals in enumerate(fs):
            for j, v in enumerate(vals): X[i, j] = maps[j][v] + offsets[j]
        for i, raw_u in enumerate(raw_users[sp]):
            if raw_u not in user_map: user_map[raw_u] = len(user_map)
            u[i] = user_map[raw_u]
        enc[sp] = (X, ys[sp], u)
    return enc, dim

def make_sampler(y, users, tabs):
    users = np.asarray(users); tabs = np.asarray(tabs); y = np.asarray(y)
    order = np.argsort(users, kind='mergesort'); su = users[order]
    pos_list = []; neg_by_user = {}; neg_by_ut = {}; s = 0; n = len(users)
    while s < n:
        e = s + 1
        while e < n and su[e] == su[s]: e += 1
        rows = order[s:e]; pos = rows[y[rows] > 0.5]; neg = rows[y[rows] <= 0.5]
        if len(pos) and len(neg):
            u = su[s]; pos_list.append(pos); neg_by_user[u] = neg.astype(np.int64)
            nt = tabs[neg]
            for t in np.unique(nt): neg_by_ut[(u, t)] = neg[nt == t].astype(np.int64)
        s = e
    pos_rows = np.concatenate(pos_list).astype(np.int64)
    return pos_rows, users[pos_rows], tabs[pos_rows], neg_by_user, neg_by_ut

def sample_pairs(pos_rows, pos_users, pos_tabs, neg_by_user, neg_by_ut, rng):
    perm = rng.permutation(len(pos_rows)); p = pos_rows[perm]; pu = pos_users[perm]; pt = pos_tabs[perm]
    negs = np.empty((len(p), 3), np.int64)
    for i, (u, t) in enumerate(zip(pu, pt)):
        pool = neg_by_user[u]
        negs[i, :2] = pool[rng.integers(len(pool), size=2)]
        hp = neg_by_ut.get((u, t), pool)
        negs[i, 2] = hp[rng.integers(len(hp))]
    return p, negs

def prepare(splits, data_dir, target):
    names = ['train', 'valid']
    if target not in names: names.append(target)
    aux = {sp: read_time_ordered(data_dir, splits[sp], sp) for sp in names}
    return make_encoded(splits, aux, names)

def train_predict_member(enc, dim, target, seed, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params':[model.V, model.W], 'weight_decay':l2}, {'params':[model.b], 'weight_decay':0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    pos_rows, pos_users, pos_tabs, neg_by_user, neg_by_ut = make_sampler(ytr, utr, Xtr[:, 3])
    rng = np.random.default_rng(seed); best = -1.; best_state = None; bad = 0; n_neg = 3
    for ep in range(1, epochs + 1):
        t0 = time.time(); pidx, nidx = sample_pairs(pos_rows, pos_users, pos_tabs, neg_by_user, neg_by_ut, rng)
        model.train(); losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i+bs]).long(); ns = torch.from_numpy(nidx[i:i+bs].reshape(-1)).long()
            xp = Xtr_t[ps].to(device); xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            loss = -torch.nn.functional.logsigmoid(model(xp).repeat_interleave(n_neg) - model(xn)).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f"  seed {seed} epoch {ep:2d} | rrf_member {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print('  early stop member seed', seed, 'at epoch', ep)
                break
    model.load_state_dict(best_state)
    Xtar, _, _ = enc[target]
    return model.predict(Xtar, device=device).astype(np.float32)

def user_zscore(scores, users):
    out = np.empty_like(scores, dtype=np.float32)
    order = np.argsort(users, kind='mergesort'); su = users[order]
    s = 0; n = len(users)
    while s < n:
        e = s + 1
        while e < n and su[e] == su[s]: e += 1
        idx = order[s:e]; x = scores[idx].astype(np.float32)
        sd = float(x.std())
        out[idx] = (x - float(x.mean())) / (sd + 1e-6)
        s = e
    return out

def user_rrf_zscore(scores, users, k=60.0):
    out = np.empty_like(scores, dtype=np.float32)
    order = np.argsort(users, kind='mergesort'); su = users[order]
    s = 0; n = len(users)
    while s < n:
        e = s + 1
        while e < n and su[e] == su[s]: e += 1
        idx = order[s:e]
        # rank 1 is the highest-scored row for that user.
        local = np.argsort(-scores[idx], kind='mergesort')
        ranks = np.empty(e - s, dtype=np.float32)
        ranks[local] = np.arange(1, e - s + 1, dtype=np.float32)
        rrf = 1.0 / (k + ranks)
        sd = float(rrf.std())
        out[idx] = (rrf - float(rrf.mean())) / (sd + 1e-6)
        s = e
    return out

def target_users_only(splits, target):
    umap = {}; users = np.empty(len(splits[target]), np.int64)
    for i, r in enumerate(splits[target]):
        u = r[USER]
        if u not in umap: umap[u] = len(umap)
        users[i] = umap[u]
    return users

def get_preds(splits, data_dir, target, split_name, base_seed, k=16, lr=0.001, epochs=40, device='cpu', verbose=False):
    os.makedirs('pred_cache', exist_ok=True)
    member_seeds = [base_seed, base_seed + 100]
    # Reuse the unchanged node-25 members; only the fusion rule changes here.
    paths = [os.path.join('pred_cache', f'025_v19_seedens_{split_name}_member_seed{ms}.npy') for ms in member_seeds]
    preds = [np.load(p).astype(np.float32) if os.path.isfile(p) else None for p in paths]
    enc = None
    if any(p is None for p in preds):
        enc, dim = prepare(splits, data_dir, target)
        for j, ms in enumerate(member_seeds):
            if preds[j] is None:
                preds[j] = train_predict_member(enc, dim, target, ms, k=k, lr=lr, epochs=epochs, device=device, verbose=verbose)
                np.save(paths[j], preds[j])
        users = enc[target][2]
    else:
        users = target_users_only(splits, target)
    z = [user_zscore(p, users) for p in preds]
    r = [user_rrf_zscore(p, users, k=60.0) for p in preds]
    zavg = 0.5 * z[0] + 0.5 * z[1]
    ravg = 0.5 * r[0] + 0.5 * r[1]
    return (0.70 * zavg + 0.30 * ravg).astype(np.float32)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data'); ap.add_argument('--split', default='valid', choices=['train','valid','test','dev']); ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args(); torch.manual_seed(a.seed)
    print(f'loading {a.data_dir} ...', flush=True)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, f'fields={FIELDS}', flush=True)
    scores = get_preds(splits, a.data_dir, target, a.split, a.seed, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
