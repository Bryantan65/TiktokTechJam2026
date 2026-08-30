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

def parse_float(rec, names, default=0.0):
    for nm in names:
        if nm in rec and rec[nm] not in ('', 'NULL', 'nan', 'NaN'):
            try: return float(rec[nm])
            except Exception: pass
    return default

def read_aux_ordered(data_dir, rows, name):
    n = len(rows)
    hour = np.full(n, -1, np.int16); ten = np.full(n, -1, np.int16); block = np.full(n, -1, np.int16)
    play = np.zeros(n, np.float32)
    files = find_logs(data_dir)
    if not files or n == 0:
        print('warning: raw logs not found for', name, flush=True); return hour, ten, block, play
    i = 0; cur = row_key(rows[0]); seen = 0; t0 = time.time()
    for path in files:
        if i >= n: break
        with open(path, 'r', encoding='utf-8', newline='') as f:
            for rec in csv.DictReader(f):
                seen += 1
                if raw_key(rec) == cur:
                    h, te, bl = parse_hourmin(rec.get('hourmin', ''))
                    hour[i], ten[i], block[i] = h, te, bl
                    play[i] = parse_float(rec, ['play_time_ms', 'play_time', 'playtime_ms', 'duration_play_ms'], 0.0)
                    i += 1
                    if i >= n: break
                    cur = row_key(rows[i])
    print(f'aligned raw aux for {name}: {i:,d}/{n:,d} rows after scanning {seen:,d} raw rows in {time.time()-t0:.1f}s; play_mean={float(play[:max(i,1)].mean()):.1f}', flush=True)
    if i < n: print(f'warning: {n-i:,d} {name} rows missing raw aux', flush=True)
    return hour, ten, block, play

def dur_bucket(x): return int(np.log1p(float(x)) // 1)

def make_encoded(splits, aux, names):
    maps = [{} for _ in range(14)]
    feats = {}; ys = {}; raw_users = {}; plays = {}
    for sp in names:
        rows = splits[sp]; h, ten, block, play = aux[sp]
        fs = []; y = np.empty(len(rows), np.float32); ru = []
        for i, r in enumerate(rows):
            u = r[USER]; t = r[TAB]; hh = int(h[i]); te = int(ten[i]); bl = int(block[i])
            vals = [
                u, r[VIDEO], r[AUTHOR], t, dur_bucket(r[DUR]), r[DATE],
                hh, te, bl, (t, hh), (t, te),
                (u, t), (u, hh), (u, bl)
            ]
            fs.append(vals); y[i] = float(r[LABEL]); ru.append(u)
            for j, v in enumerate(vals):
                if v not in maps[j]: maps[j][v] = len(maps[j])
        feats[sp] = fs; ys[sp] = y; raw_users[sp] = ru; plays[sp] = play.astype(np.float32)
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
        enc[sp] = (X, ys[sp], u, plays[sp])
    return enc, dim

def make_sampler(y, users, tabs, play, dur_ms):
    users = np.asarray(users); tabs = np.asarray(tabs); y = np.asarray(y); play = np.asarray(play, dtype=np.float32)
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
    # Confidence weight only positives. Use watch duration beyond the binary label signal, clipped to avoid a few very long sessions dominating.
    dur = np.asarray(dur_ms, dtype=np.float32)
    ratio = play[pos_rows] / np.maximum(dur[pos_rows], 1.0)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    w = 1.0 + 0.35 * np.clip(np.log1p(np.maximum(ratio, 0.0)), 0.0, 2.5)
    w = w.astype(np.float32)
    return pos_rows, users[pos_rows], tabs[pos_rows], w, neg_by_user, neg_by_ut

def sample_pairs(pos_rows, pos_users, pos_tabs, pos_w, neg_by_user, neg_by_ut, rng):
    perm = rng.permutation(len(pos_rows)); p = pos_rows[perm]; pu = pos_users[perm]; pt = pos_tabs[perm]; w = pos_w[perm]
    negs = np.empty((len(p), 3), np.int64)
    for i, (u, t) in enumerate(zip(pu, pt)):
        pool = neg_by_user[u]
        negs[i, :2] = pool[rng.integers(len(pool), size=2)]
        hp = neg_by_ut.get((u, t), pool)
        negs[i, 2] = hp[rng.integers(len(hp))]
    return p, negs, w

def run(splits, data_dir, target, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0, device='cpu', verbose=True):
    names = ['train', 'valid']
    if target not in names: names.append(target)
    aux = {sp: read_aux_ordered(data_dir, splits[sp], sp) for sp in names}
    enc, dim = make_encoded(splits, aux, names)
    Xtr, ytr, utr, playtr = enc['train']; Xva, yva, uva, _ = enc['valid']
    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params':[model.V, model.W], 'weight_decay':l2}, {'params':[model.b], 'weight_decay':0.0}], lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    dur_ms = np.array([float(r[DUR]) for r in splits['train']], dtype=np.float32)
    pos_rows, pos_users, pos_tabs, pos_w, neg_by_user, neg_by_ut = make_sampler(ytr, utr, Xtr[:, 3], playtr, dur_ms)
    print(f'positive pair weights: mean={float(pos_w.mean()):.4f} min={float(pos_w.min()):.4f} max={float(pos_w.max()):.4f}', flush=True)
    rng = np.random.default_rng(seed); best = -1.; best_state = None; bad = 0; n_neg = 3
    for ep in range(1, epochs + 1):
        t0 = time.time(); pidx, nidx, pw = sample_pairs(pos_rows, pos_users, pos_tabs, pos_w, neg_by_user, neg_by_ut, rng)
        model.train(); losses = []
        for i in range(0, len(pidx), bs):
            ps = torch.from_numpy(pidx[i:i+bs]).long(); ns = torch.from_numpy(nidx[i:i+bs].reshape(-1)).long()
            wt = torch.from_numpy(pw[i:i+bs]).to(device).repeat_interleave(n_neg)
            xp = Xtr_t[ps].to(device); xn = Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True)
            raw = -torch.nn.functional.logsigmoid(model(xp).repeat_interleave(n_neg) - model(xn))
            loss = (raw * wt).mean() / wt.mean().clamp_min(1e-6)
            loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f"  epoch {ep:2d} | bpr_time_userctx_playweight {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print('  early stop at epoch', ep)
                break
    model.load_state_dict(best_state)
    return model, enc

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
    model, enc = run(splits, a.data_dir, target, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    X, y, users, _ = enc[target]; scores = model.predict(X, device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
