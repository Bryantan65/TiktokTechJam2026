import argparse, csv, os, sys, time
from collections import defaultdict, deque
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode
from evaluate import evaluate

AUX_NAMES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
RANK_GAMMA = 2.0
MARGIN_WEIGHT = 0.30
BPR_EPOCHS = 2
BPR_LR = 3e-4
BPR_BCE_ANCHOR = 0.15

def to_int(x, default=0):
    try: return int(float(x))
    except Exception: return default

def raw_paths(data_dir):
    p = [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'), os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]
    if all(os.path.exists(x) for x in p): return p
    return [os.path.join(data_dir, 'data', 'log_standard_4_08_to_4_21_pure.csv'), os.path.join(data_dir, 'data', 'log_standard_4_22_to_5_08_pure.csv')]

def raw_key(r):
    return (to_int(r.get('date')), to_int(r.get('user_id')), to_int(r.get('video_id')), to_int(r.get('author_id')), to_int(r.get('tab')), to_int(r.get('duration_ms')))

def row_key(row):
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))

def read_time_features(data_dir, splits):
    by = defaultdict(deque)
    for path in raw_paths(data_dir):
        if not os.path.exists(path): continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                hm = to_int(r.get('hourmin'), 0); hour = max(0, min(23, hm // 100))
                by[raw_key(r)].append((hour, hour // 4))
    out = {}
    for sp, rows in splits.items():
        a = np.zeros((len(rows), 2), dtype=np.int64)
        for i, row in enumerate(rows):
            q = by.get(row_key(row))
            if q: a[i] = q.popleft()
        out[sp] = a
    return out

def add_time_fields(enc, splits, data_dir):
    times = read_time_features(data_dir, splits)
    offset = int(max(v[0].max() for v in enc.values())) + 1
    maps = []
    for j in range(times['train'].shape[1]):
        vals = np.unique(times['train'][:, j]); mp = {int(v): offset + i + 1 for i, v in enumerate(vals)}
        maps.append((mp, offset)); offset += len(vals) + 1
    out = {}
    for sp, (X, y, u) in enc.items():
        extra = np.zeros((len(X), len(maps)), dtype=np.int64)
        for j, (mp, unk) in enumerate(maps):
            extra[:, j] = [mp.get(int(v), unk) for v in times[sp][:, j]]
        out[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, u)
    return out, offset

def read_aux_targets(data_dir, splits):
    by = defaultdict(deque)
    for path in raw_paths(data_dir):
        if not os.path.exists(path): continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                dur = max(to_int(r.get('duration_ms')), 1); play = max(to_int(r.get('play_time_ms')), 0)
                vals = [float(to_int(r.get(n), 0) > 0) for n in AUX_NAMES]
                ratio = play / float(dur); vals += [float(ratio >= .25), float(ratio >= .50), float(ratio >= 1.0)]
                by[raw_key(r)].append(np.asarray(vals, dtype=np.float32))
    out = {}; dim = len(AUX_NAMES) + 3
    for sp, rows in splits.items():
        a = np.zeros((len(rows), dim), dtype=np.float32)
        for i, row in enumerate(rows):
            q = by.get(row_key(row))
            if q: a[i] = q.popleft()
        out[sp] = a
    return out

class HistState:
    names = ('ui','up','uai','uap','uvi','uvp','uti','utp','udi','udp','ai','ap','vi','vp')
    def __init__(self):
        for n in self.names: setattr(self, n, defaultdict(int))
    def copy(self):
        o = HistState()
        for n in self.names: setattr(o, n, defaultdict(int, getattr(self, n).copy()))
        return o
    def features_one(self, row):
        u,v,a,tab = row[1],row[2],row[3],row[4]; dur = int(row[5]) // 10000
        ui,up = self.ui[u],self.up[u]; uai,uap = self.uai[(u,a)],self.uap[(u,a)]
        uvi,uvp = self.uvi[(u,v)],self.uvp[(u,v)]; uti,utp = self.uti[(u,tab)],self.utp[(u,tab)]
        udi,udp = self.udi[(u,dur)],self.udp[(u,dur)]; ai,ap = self.ai[a],self.ap[a]; vi,vp = self.vi[v],self.vp[v]
        return [np.log1p(ui),(up+1)/(ui+2),np.log1p(uai),(uap+.5)/(uai+2),uap/(up+1),np.log1p(uvi),(uvp+.5)/(uvi+2),np.log1p(uti),(utp+.5)/(uti+2),np.log1p(udi),(udp+.5)/(udi+2),np.log1p(ai),(ap+1)/(ai+2),np.log1p(vi),(vp+.5)/(vi+2)]
    def update(self, row):
        u,v,a,tab = row[1],row[2],row[3],row[4]; dur = int(row[5]) // 10000; y = 1 if row[6] > 0 else 0
        self.ui[u]+=1; self.up[u]+=y; self.uai[(u,a)]+=1; self.uap[(u,a)]+=y; self.uvi[(u,v)]+=1; self.uvp[(u,v)]+=y
        self.uti[(u,tab)]+=1; self.utp[(u,tab)]+=y; self.udi[(u,dur)]+=1; self.udp[(u,dur)]+=y; self.ai[a]+=1; self.ap[a]+=y; self.vi[v]+=1; self.vp[v]+=y

def make_history_features(splits):
    st = HistState(); feats = {}; tr = np.empty((len(splits['train']), 15), dtype=np.float32)
    for i, row in enumerate(splits['train']): tr[i] = st.features_one(row); st.update(row)
    feats['train'] = tr; train_st = st.copy()
    for sp in ('valid', 'test'):
        s = train_st.copy(); a = np.empty((len(splits[sp]), 15), dtype=np.float32)
        for i, row in enumerate(splits[sp]): a[i] = s.features_one(row)
        feats[sp] = a
    mu = feats['train'].mean(0); sd = feats['train'].std(0) + 1e-6
    for sp in feats: feats[sp] = (feats[sp] - mu) / sd
    return feats

class Model(torch.nn.Module):
    def __init__(self, dim, hdim, adim, k=16, seed=0):
        super().__init__(); rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, .01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim)); self.b = torch.nn.Parameter(torch.zeros(()))
        self.H = torch.nn.Linear(hdim, 1); torch.nn.init.zeros_(self.H.weight); torch.nn.init.zeros_(self.H.bias)
        self.aux = torch.nn.Sequential(torch.nn.Linear(k + hdim, 32), torch.nn.ReLU(), torch.nn.Linear(32, adim))
    def forward(self, X, H, aux=False):
        E = self.V[X]; S = E.sum(1); inter = .5 * ((S*S).sum(1) - (E*E).sum((1,2)))
        main = self.b + self.W[X].sum(1) + inter + self.H(H).squeeze(1)
        if aux: return main, self.aux(torch.cat([S, H], 1))
        return main
    @torch.no_grad()
    def predict(self, X, H, bs=200000, device='cpu'):
        self.eval(); out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device); hb = torch.from_numpy(H[i:i+bs].astype(np.float32)).to(device)
            out.append(self(xb, hb).cpu().numpy())
        return np.concatenate(out)

def prepare(splits, data_dir):
    enc0, _ = encode(splits); enc, dim = add_time_fields(enc0, splits, data_dir)
    return enc, dim, make_history_features(splits), read_aux_targets(data_dir, splits)

def train_bce(enc, dim, hist, aux_t, seed=0, k=16, lr=.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    torch.manual_seed(seed); Xtr,ytr,_ = enc['train']; Xva,yva,uva = enc['valid']; Htr,Hva = hist['train'],hist['valid']; Atr = aux_t['train']
    m = Model(dim, Htr.shape[1], Atr.shape[1], k, seed).to(device)
    opt = torch.optim.Adam([{'params':[m.V,m.W],'weight_decay':l2},{'params':[m.b],'weight_decay':0.0},{'params':m.H.parameters(),'weight_decay':1e-5},{'params':m.aux.parameters(),'weight_decay':1e-5}], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(); Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); Htr_t = torch.from_numpy(Htr.astype(np.float32)); ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32)); Atr_t = torch.from_numpy(Atr.astype(np.float32))
    rng = np.random.default_rng(seed); best = -1.; state = None; bad = 0
    for ep in range(1, epochs+1):
        idx = rng.permutation(len(ytr)); m.train(); losses = []; t0 = time.time()
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i+bs]); xb = Xtr_t[sel].to(device); hb = Htr_t[sel].to(device); yb = ytr_t[sel].to(device); ab = Atr_t[sel].to(device)
            opt.zero_grad(set_to_none=True); main, aux = m(xb, hb, aux=True); loss = bce(main, yb) + .08 * bce(aux, ab); loss.backward(); opt.step(); losses.append(loss.item())
        va = evaluate(uva, yva, m.predict(Xva, Hva, device=device))
        if verbose: print(f'seed {seed} ep {ep} loss {np.mean(losses):.4f} valid {va["primary"]:.4f} {time.time()-t0:.1f}s')
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; state = {kk: vv.detach().clone() for kk, vv in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m

def make_pair_arrays(y, users, seed=0):
    y = np.asarray(y); users = np.asarray(users); pos = defaultdict(list); neg = defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        if yy > 0: pos[u].append(i)
        else: neg[u].append(i)
    rng = np.random.default_rng(seed + 12345); pp = []; nn = []
    for u in pos:
        if len(neg.get(u, [])) == 0: continue
        p = np.asarray(pos[u], dtype=np.int64); n = np.asarray(neg[u], dtype=np.int64)
        reps = min(len(p) * 3, max(len(p), len(n)))
        pp.append(rng.choice(p, size=reps, replace=True)); nn.append(rng.choice(n, size=reps, replace=True))
    return np.concatenate(pp), np.concatenate(nn)

def bpr_finetune(m, enc, hist, seed=0, lr=BPR_LR, epochs=BPR_EPOCHS, bs=8192, device='cpu'):
    Xtr,ytr,utr = enc['train']; Htr = hist['train']
    pos, neg = make_pair_arrays(ytr, utr, seed)
    X = torch.from_numpy(Xtr.astype(np.int64)); H = torch.from_numpy(Htr.astype(np.float32)); Y = torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6); bce = torch.nn.BCEWithLogitsLoss(); rng = np.random.default_rng(seed + 777)
    for ep in range(epochs):
        ordp = rng.permutation(len(pos)); m.train()
        for i in range(0, len(ordp), bs):
            ids = ordp[i:i+bs]; pi = torch.from_numpy(pos[ids]); ni = torch.from_numpy(neg[ids]); both = torch.cat([pi, ni])
            xp = X[pi].to(device); hp = H[pi].to(device); xn = X[ni].to(device); hn = H[ni].to(device)
            xb = X[both].to(device); hb = H[both].to(device); yb = Y[both].to(device)
            opt.zero_grad(set_to_none=True)
            sp = m(xp, hp); sn = m(xn, hn)
            loss = -torch.nn.functional.logsigmoid(sp - sn).mean() + BPR_BCE_ANCHOR * bce(m(xb, hb), yb)
            loss.backward(); opt.step()
    return m

def per_user_rank_margin_blend(preds, users, gamma=RANK_GAMMA, margin_weight=MARGIN_WEIGHT):
    P = np.vstack(preds).astype(np.float64); meanp = P.mean(axis=0); users = np.asarray(users); out = np.zeros(P.shape[1], dtype=np.float64)
    order = np.argsort(users, kind='mergesort'); su = users[order]; start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and su[end] == su[start]: end += 1
        idx = order[start:end]; n = end - start
        if n <= 1: out[idx] = meanp[idx]
        else:
            acc = np.zeros(n, dtype=np.float64)
            for mi in range(P.shape[0]):
                vals = P[mi, idx]; r = np.empty(n, dtype=np.float64); r[np.argsort(vals, kind='mergesort')] = np.arange(n, dtype=np.float64); acc += (r / (n - 1.0)) ** gamma
            rank_score = acc / P.shape[0]; z = meanp[idx]; z = (z - z.mean()) / (z.std() + 1e-6); margin_score = 1.0 / (1.0 + np.exp(-z))
            out[idx] = (1.0 - margin_weight) * rank_score + margin_weight * margin_score
        start = end
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='./KuaiRand-Pure/data'); ap.add_argument('--split', default='valid', choices=['train','valid','test']); ap.add_argument('--out', default=None); ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=.001); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu', choices=['cpu','cuda']); a = ap.parse_args()
    print(f'loading {a.data_dir} ...'); splits = load(a.data_dir); print({k: len(v) for k, v in splits.items()}, '8x BCE+BPR hybrid rank+margin blend')
    enc, dim, hist, aux_t = prepare(splits, a.data_dir); X, y, users = enc[a.split]; preds = []
    for s in range(4):
        m = train_bce(enc, dim, hist, aux_t, seed=s, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=(a.out is None))
        preds.append(m.predict(X, hist[a.split], device=a.device))
        mb = Model(dim, hist['train'].shape[1], aux_t['train'].shape[1], a.k, s).to(a.device); mb.load_state_dict({kk: vv.detach().clone() for kk, vv in m.state_dict().items()})
        mb = bpr_finetune(mb, enc, hist, seed=s, device=a.device)
        preds.append(mb.predict(X, hist[a.split], device=a.device))
    scores = per_user_rank_margin_blend(preds, users)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else: print(scores[:10])
