"""Debug node 20: content FM rank ensemble plus sequentially aligned raw hourmin prior.

Node 20 joined raw CSV hourmin by a fragile tuple key and missed every row.  The
starter data preserves raw log row order, so this version attaches time features
by reading the two log_standard files in order and slicing them to the loaded
train/valid/test splits.  Content FM member code/cache names are unchanged from
node 17, so cached member predictions are reused when present and retrained when
absent.
"""
import argparse, csv, os, sys, time, math
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate

CONTENT_FIELDS = ['music_id', 'tag', 'video_type', 'upload_type', 'server_width', 'server_height', 'music_type']

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32)); self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        return self.b + self.W[X].sum(1) + 0.5 * ((S*S).sum(1) - (E*E).sum((1,2)))
    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval(); out=[]
        for i in range(0, len(X), bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def norm_val(v):
    if v is None or v == '': return '__MISS__'
    s = str(v).strip(); return s if s else '__MISS__'

def load_video_content(data_dir):
    path = os.path.join(data_dir, 'video_features_basic_pure.csv'); mp={}
    if not os.path.isfile(path): print('WARNING: no video_features_basic_pure.csv'); return mp
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f); vid_col = 'video_id' if 'video_id' in (rdr.fieldnames or []) else ('item_id' if 'item_id' in (rdr.fieldnames or []) else None)
        if vid_col is None: print('WARNING: video feature file has no video_id/item_id'); return mp
        for rec in rdr: mp[norm_val(rec.get(vid_col))] = tuple(norm_val(rec.get(c)) for c in CONTENT_FIELDS)
    print('loaded video content rows', len(mp)); return mp

def append_content_features(splits, enc, base_dim, data_dir):
    content = load_video_content(data_dir); rows_all=[]
    for sp in ['train','valid','test']:
        if sp in splits: rows_all.extend(splits[sp])
    maps=[{'__MISS__':0} for _ in CONTENT_FIELDS]
    for r in rows_all:
        vals = content.get(norm_val(r[2]), ('__MISS__',)*len(CONTENT_FIELDS))
        for j,v in enumerate(vals):
            if v not in maps[j]: maps[j][v]=len(maps[j])
    offsets=[]; off=base_dim
    for m in maps: offsets.append(off); off += len(m)
    print('content cardinalities', {CONTENT_FIELDS[i]:len(maps[i]) for i in range(len(CONTENT_FIELDS))}, 'new_dim', off)
    out={}
    for sp,(X,y,u) in enc.items():
        C=np.empty((len(X),len(CONTENT_FIELDS)), dtype=np.int64)
        for i,r in enumerate(splits[sp]):
            vals = content.get(norm_val(r[2]), ('__MISS__',)*len(CONTENT_FIELDS))
            for j,v in enumerate(vals): C[i,j] = offsets[j] + maps[j].get(v,0)
        out[sp]=(np.concatenate([X.astype(np.int64), C], axis=1), y, u)
    return out, off

def train_bce_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}], lr=lr)
    X_t=torch.from_numpy(Xtr.astype(np.int64)); y_t=torch.from_numpy(ytr.astype(np.float32)); rng=np.random.default_rng(seed)
    best=-1; best_state=None; bad=0; n=len(Xtr)
    for ep in range(1, epochs+1):
        t0=time.time(); model.train(); losses=[]; perm=rng.permutation(n)
        for i in range(0,n,bs):
            idx=torch.from_numpy(perm[i:i+bs]); xb=X_t[idx].to(device); yb=y_t[idx].to(device)
            opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(model(xb), yb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  bce seed {seed} epoch {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5: best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model

def make_positive_index_pairs(y, users):
    y=np.asarray(y); users=np.asarray(users); order=np.argsort(users, kind='mergesort'); us=users[order]
    pos_indices=[]; pos_gids=[]; neg_by_gid=[]; start=0; gid=0; n=len(order)
    while start<n:
        end=start+1
        while end<n and us[end]==us[start]: end+=1
        idx=order[start:end]; yy=y[idx]; pos=idx[yy>0.5]; neg=idx[yy<=0.5]
        if len(pos)>0 and len(neg)>0:
            pos_indices.append(pos.astype(np.int64)); pos_gids.append(np.full(len(pos),gid,dtype=np.int32)); neg_by_gid.append(neg.astype(np.int64)); gid+=1
        start=end
    return np.concatenate(pos_indices), np.concatenate(pos_gids), neg_by_gid

def sample_negatives_for_batch(gids, neg_by_gid, rng):
    neg=np.empty(len(gids),dtype=np.int64)
    for g in np.unique(gids):
        m=(gids==g); pool=neg_by_gid[int(g)]; neg[m]=pool[rng.integers(0,len(pool),size=int(m.sum()))]
    return neg

def train_bpr_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5, repeats=2, bce_weight=0.10, device='cpu', verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}], lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); pos_base,pos_gid_base,neg_by_gid=make_positive_index_pairs(ytr,utr); rng=np.random.default_rng(seed)
    best=-1; best_state=None; bad=0
    for ep in range(1, epochs+1):
        t0=time.time(); model.train(); losses=[]
        for _ in range(repeats):
            perm=rng.permutation(len(pos_base))
            for i in range(0,len(perm),bs):
                psel=perm[i:i+bs]; pos_idx=pos_base[psel]; neg_idx=sample_negatives_for_batch(pos_gid_base[psel],neg_by_gid,rng)
                xb=torch.cat([Xtr_t[torch.from_numpy(pos_idx)], Xtr_t[torch.from_numpy(neg_idx)]], dim=0).to(device)
                opt.zero_grad(set_to_none=True); logits=model(xb); m=len(pos_idx); loss=F.softplus(-(logits[:m]-logits[m:])).mean()
                if bce_weight>0:
                    labels=torch.cat([torch.ones(m,device=device), torch.zeros(m,device=device)])
                    loss = loss + bce_weight * F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  bpr seed {seed} epoch {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5: best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model

def percentile_rank_by_user(scores, users):
    scores=np.asarray(scores,dtype=np.float64); users=np.asarray(users); out=np.empty_like(scores,dtype=np.float64)
    order=np.argsort(users,kind='mergesort'); us=users[order]; start=0; n=len(order)
    while start<n:
        end=start+1
        while end<n and us[end]==us[start]: end+=1
        idx=order[start:end]; ord2=idx[np.argsort(scores[idx],kind='mergesort')]; m=len(ord2)
        out[ord2]=0.0 if m<=1 else np.arange(m,dtype=np.float64)/(m-1.0); start=end
    return out

def get_member_preds(name, train_fn, enc, dim, Xtar, seed, device, verbose):
    os.makedirs('pred_cache', exist_ok=True); path=os.path.join('pred_cache', f'018_{name}_seed{seed}_n{len(Xtar)}.npy')
    if os.path.isfile(path): print(f'loading cached {name} seed {seed} predictions {path}'); return np.load(path).astype(np.float64)
    print(f'training {name} seed {seed} member'); model=train_fn(enc,dim,seed=seed,device=device,verbose=verbose); preds=model.predict(Xtar,device=device).astype(np.float64); np.save(path,preds); return preds

def parse_hourmin(v):
    try: x=int(float(str(v).strip()))
    except Exception: return -1,-1
    h=x//100; m=x%100
    return (h,m) if (0<=h<=23 and 0<=m<=59) else (-1,-1)

def day_of_week(date_val):
    try: return datetime.strptime(str(int(date_val)), '%Y%m%d').weekday()
    except Exception: return -1

def load_raw_time_sequence(data_dir):
    seq=[]; files=['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']
    for fn in files:
        path=os.path.join(data_dir,fn)
        if not os.path.isfile(path): print('WARNING: missing raw log', path); continue
        with open(path,newline='',encoding='utf-8') as f:
            rdr=csv.DictReader(f)
            for rec in rdr: seq.append(parse_hourmin(rec.get('hourmin','')))
    print('loaded raw hour sequence rows', len(seq)); return seq

def attach_times_sequential(splits, data_dir):
    seq=load_raw_time_sequence(data_dir); out={}; off=0; miss=0; seen=0
    # data.load preserves the same row order as the concatenated raw logs.  Devdata is also a prefix split of train.
    for sp in ['train','valid','test']:
        if sp not in splits: continue
        n=len(splits[sp]); arr=np.empty((n,5),dtype=np.int16)
        for i,r in enumerate(splits[sp]):
            if off+i < len(seq): h,m=seq[off+i]
            else: h,m=-1,-1; miss += 1
            dow=day_of_week(r[0]); coarse=(h//4) if h>=0 else -1; quarter=(h*4+m//15) if h>=0 else -1; weekend=1 if dow in (5,6) else (0 if dow>=0 else -1)
            arr[i]=(h,coarse,quarter,dow,weekend); seen += 1
        out[sp]=arr; off += n
    bad=int(sum((out[sp][:,0] < 0).sum() for sp in out)); print('time sequential missing/out-of-range', bad, 'of', seen, 'raw_short', miss)
    return out

def fit_time_prior(train_rows, y, train_time):
    y=np.asarray(y,dtype=np.float64); mu=float(y.mean()); eps=1e-6
    def logit(p): p=min(max(float(p),eps),1-eps); return math.log(p/(1-p))
    glob=logit(mu); sums=defaultdict(lambda:[0.0,0.0])
    def add(key,val): sums[key][0]+=float(val); sums[key][1]+=1.0
    for r,yy,tt in zip(train_rows,y,train_time):
        user=str(r[1]); tab=str(r[4]); durb=str(r[5]); h,coarse,quarter,dow,weekend=[int(x) for x in tt]
        add(('h',h),yy); add(('q',quarter),yy); add(('dow',dow),yy); add(('htab',h,tab),yy); add(('qtab',quarter,tab),yy); add(('cdur',coarse,durb),yy); add(('u_c',user,coarse),yy); add(('u_h',user,h),yy)
    smooth={'h':200.0,'q':150.0,'dow':500.0,'htab':120.0,'qtab':80.0,'cdur':120.0,'u_c':8.0,'u_h':5.0}; table={}
    for key,(s,n) in sums.items():
        lam=smooth.get(key[0],100.0); table[key]=logit((s+lam*mu)/(n+lam))-glob
    return mu,glob,table

def predict_time_prior(rows, tarr, prior):
    mu,glob,table=prior; scores=np.empty(len(rows),dtype=np.float64)
    for i,(r,tt) in enumerate(zip(rows,tarr)):
        user=str(r[1]); tab=str(r[4]); durb=str(r[5]); h,coarse,quarter,dow,weekend=[int(x) for x in tt]
        s=glob + 0.22*table.get(('h',h),0.0) + 0.14*table.get(('q',quarter),0.0) + 0.12*table.get(('dow',dow),0.0)
        s += 0.20*table.get(('htab',h,tab),0.0) + 0.12*table.get(('qtab',quarter,tab),0.0) + 0.08*table.get(('cdur',coarse,durb),0.0)
        s += 0.08*table.get(('u_c',user,coarse),0.0) + 0.04*table.get(('u_h',user,h),0.0); scores[i]=s
    return scores

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(0); print(f'loading {a.data_dir} ...')
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}+{CONTENT_FIELDS}+seq_time_prior')
    enc0,dim0=encode(splits); enc,dim=append_content_features(splits,enc0,dim0,a.data_dir); Xtar,_,utar=enc[target]; verbose=(a.out is None)
    blended=[]
    for member_seed in (0,1,2):
        bpr=get_member_preds('content_bpr_anchor_v1',train_bpr_member,enc,dim,Xtar,member_seed,a.device,verbose)
        bce=get_member_preds('content_bce_v1',train_bce_member,enc,dim,Xtar,member_seed,a.device,verbose)
        blended.append(0.70*percentile_rank_by_user(bpr,utar)+0.30*percentile_rank_by_user(bce,utar))
    content_score=np.mean(blended,axis=0)
    times=attach_times_sequential(splits,a.data_dir); prior=fit_time_prior(splits['train'],enc['train'][1],times['train'])
    time_score=percentile_rank_by_user(predict_time_prior(splits[target],times[target],prior), utar)
    scores=0.80*content_score + 0.20*time_score
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else: print('done')
