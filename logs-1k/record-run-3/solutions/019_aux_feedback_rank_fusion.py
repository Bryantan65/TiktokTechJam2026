"""Auxiliary-feedback historical rank member blended with node-10 rank fusion.

Keeps the two cached node-10 members unchanged: listwise softmax FM and target
historical-rate stats.  Adds a readable 30% rank member built only from TRAIN
raw auxiliary feedback columns (click/like/follow/comment/forward and watch-time
magnitudes when present).  It never reads raw long_view or uses target-row raw
feedback as a feature; target predictions use only item/user/tab/duration keys
and TRAIN aggregates.
"""
import argparse, csv, glob, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate          # noqa  (early stopping only)

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32)); self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float32))
    def forward(self,X):
        E=self.V[X]; S=E.sum(1); inter=0.5*((S**2).sum(1)-(E**2).sum((1,2)))
        return self.b+self.W[X].sum(1)+inter
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def make_user_groups(users,y):
    users=np.asarray(users); order=np.argsort(users,kind='stable'); su=users[order]
    bounds=np.r_[0,np.flatnonzero(su[1:]!=su[:-1])+1,len(su)]; groups=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        idx=order[a:b]; npos=int(y[idx].sum())
        if npos>0 and npos<len(idx): groups.append(idx.astype(np.int64))
    return groups

def listwise_user_loss(model,Xtr_t,ytr_t,groups,device,max_rows=24000):
    losses=[]; rows=0
    for g in groups:
        if rows>=max_rows and losses: break
        s=model(Xtr_t[g].to(device)); yb=ytr_t[g].to(device); pos=yb>0.5
        losses.append(torch.logsumexp(s,0)-torch.logsumexp(s[pos],0)); rows+=len(g)
    return torch.stack(losses).mean()

def train_softmax_member(splits,target,seed,device='cpu'):
    enc,dim=encode(splits); Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    model=TorchFM(dim,16,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=0.003)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); groups=make_user_groups(utr,ytr)
    rng=np.random.default_rng(seed); best=-1.; best_state=None; bad=0
    for ep in range(1,81):
        perm=rng.permutation(len(groups)); losses=[]; model.train(); t0=time.time()
        for i in range(0,len(perm),48):
            bg=[groups[j] for j in perm[i:i+48]]; opt.zero_grad(set_to_none=True)
            loss=listwise_user_loss(model,Xtr_t,ytr_t,bg,device); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device)); print(f"softmax ep {ep:02d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=6: break
    model.load_state_dict(best_state); Xtg,_,users=enc[target]
    return model.predict(Xtg,device=device).astype(np.float64), users

def logit(p): p=np.clip(p,1e-5,1-1e-5); return np.log(p/(1-p))
def one_rate(ktr,ytr,ktg,gm,alpha,train=False):
    n=int(max(ktr.max(initial=0),ktg.max(initial=0)))+1; cnt=np.bincount(ktr,minlength=n).astype(np.float32); sm=np.bincount(ktr,weights=ytr,minlength=n).astype(np.float32)
    if train: c=cnt[ktr]-1.; s=sm[ktr]-ytr
    else: c=cnt[ktg]; s=sm[ktg]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha)
def one_mean(ktr,val,ktg,gm,alpha,train=False):
    n=int(max(ktr.max(initial=0),ktg.max(initial=0)))+1; cnt=np.bincount(ktr,minlength=n).astype(np.float32); sm=np.bincount(ktr,weights=val,minlength=n).astype(np.float32)
    if train: c=cnt[ktr]-1.; s=sm[ktr]-val
    else: c=cnt[ktg]; s=sm[ktg]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha)
def pair_keys(a,b,nb): return a.astype(np.int64)*np.int64(nb)+b.astype(np.int64)
def pair_rate(atr,btr,ytr,atg,btg,gm,nb,alpha,train=False):
    ktr=pair_keys(atr,btr,nb); uniq,inv=np.unique(ktr,return_inverse=True); cnt=np.bincount(inv).astype(np.float32); sm=np.bincount(inv,weights=ytr).astype(np.float32)
    if train: c=cnt[inv]-1.; s=sm[inv]-ytr
    else:
        ktg=pair_keys(atg,btg,nb); pos=np.searchsorted(uniq,ktg); ok=(pos<len(uniq))&(uniq[np.minimum(pos,len(uniq)-1)]==ktg)
        c=np.zeros(len(ktg),dtype=np.float32); s=np.zeros(len(ktg),dtype=np.float32)
        if ok.any(): c[ok]=cnt[pos[ok]]; s[ok]=sm[pos[ok]]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha)
def pair_mean(atr,btr,val,atg,btg,gm,nb,alpha,train=False):
    ktr=pair_keys(atr,btr,nb); uniq,inv=np.unique(ktr,return_inverse=True); cnt=np.bincount(inv).astype(np.float32); sm=np.bincount(inv,weights=val).astype(np.float32)
    if train: c=cnt[inv]-1.; s=sm[inv]-val
    else:
        ktg=pair_keys(atg,btg,nb); pos=np.searchsorted(uniq,ktg); ok=(pos<len(uniq))&(uniq[np.minimum(pos,len(uniq)-1)]==ktg)
        c=np.zeros(len(ktg),dtype=np.float32); s=np.zeros(len(ktg),dtype=np.float32)
        if ok.any(): c[ok]=cnt[pos[ok]]; s[ok]=sm[pos[ok]]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha)

def stats_member(splits,target,seed):
    enc,dim=encode(splits); Xtr,ytr,_=enc['train']; Xtg,_,users=enc[target]; ytr=ytr.astype(np.float32); gm=float(ytr.mean()); is_train=target=='train'; rng=np.random.default_rng(seed)
    score=np.zeros(len(Xtg),dtype=np.float64); wsum=0.
    def jw(w): return float(w*(1.+rng.normal(0.,0.015)))
    def add(w,r):
        nonlocal score,wsum; w=jw(w); score+=w*logit(r).astype(np.float64); wsum+=w
    for col,w,a in [(1,1.4,30.),(2,1.1,30.),(3,0.9,80.),(4,0.5,80.)]: add(w,one_rate(Xtr[:,col].astype(np.int64),ytr,Xtg[:,col].astype(np.int64),gm,a,train=is_train))
    maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1; maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    specs=[(Xtr[:,0],Xtr[:,1],Xtg[:,0],Xtg[:,1],maxv,1.6,15.),(Xtr[:,0],Xtr[:,2],Xtg[:,0],Xtg[:,2],maxa,1.3,25.),(Xtr[:,0],Xtr[:,3],Xtg[:,0],Xtg[:,3],maxt,1.0,40.),(Xtr[:,1],Xtr[:,3],Xtg[:,1],Xtg[:,3],maxt,0.6,40.),(Xtr[:,2],Xtr[:,3],Xtg[:,2],Xtg[:,3],maxt,0.5,40.),(Xtr[:,1],Xtr[:,4],Xtg[:,1],Xtg[:,4],maxd,0.4,40.),(Xtr[:,2],Xtr[:,4],Xtg[:,2],Xtg[:,4],maxd,0.3,40.)]
    for atr,btr,atg,btg,nb,w,a in specs: add(w,pair_rate(atr.astype(np.int64),btr.astype(np.int64),ytr,atg.astype(np.int64),btg.astype(np.int64),gm,nb,a,train=is_train))
    print(f"stats dim={dim} gm={gm:.6f}"); return (score/wsum).astype(np.float64), users

def find_train_log(data_dir):
    pats=[os.path.join(data_dir,'log_standard_4_08_to_4_21_1k.csv'), os.path.join(data_dir,'log_standard_4_08_to_4_21*.csv')]
    for pat in pats:
        fs=glob.glob(pat); fs=[f for f in fs if '_pure' not in os.path.basename(f)] or fs
        if fs: return sorted(fs)[0]
    raise FileNotFoundError('train log csv not found')

def to_float(x):
    try:
        if x is None or x=='': return 0.0
        return float(x)
    except Exception:
        return 0.0

def read_aux_train(data_dir,n):
    path=find_train_log(data_dir); print('reading aux from',path)
    with open(path,newline='') as f:
        rdr=csv.DictReader(f); names=rdr.fieldnames or []
        lower={c.lower():c for c in names}
        forbidden={'long_view','is_long_view','label','duration_ms'}
        bin_wanted=['is_click','is_like','is_follow','is_comment','is_forward','is_hate','is_profile_enter','is_profile_stay']
        cont_wanted=['play_time_ms','profile_stay_time','comment_stay_time']
        bin_cols=[lower[c] for c in bin_wanted if c in lower and c not in forbidden]
        cont_cols=[lower[c] for c in cont_wanted if c in lower and c not in forbidden]
        vals={c:np.zeros(n,dtype=np.float32) for c in bin_cols+cont_cols}
        for i,row in enumerate(rdr):
            if i>=n: break
            for c in bin_cols:
                vals[c][i]=1.0 if to_float(row.get(c,''))>0 else 0.0
            for c in cont_cols:
                # Magnitude only; no division by duration and no target-row usage.
                vals[c][i]=np.log1p(max(0.0,to_float(row.get(c,''))))
    print('aux columns', list(vals.keys()))
    return vals, bin_cols, cont_cols

def standardize(v):
    v=v.astype(np.float64); sd=float(np.std(v))
    if sd<1e-12: return v*0.0
    return (v-float(np.mean(v)))/sd

def aux_feedback_member(splits,target,seed,data_dir):
    enc,dim=encode(splits); Xtr,ytr,_=enc['train']; Xtg,_,users=enc[target]
    vals,bin_cols,cont_cols=read_aux_train(data_dir,len(Xtr)); is_train=target=='train'
    if not vals:
        print('no aux columns found; falling back to zero aux member')
        return np.zeros(len(Xtg),dtype=np.float64), users
    maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1; maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    rng=np.random.default_rng(seed+33031); score=np.zeros(len(Xtg),dtype=np.float64); wsum=0.0
    def jw(w): return float(w*(1.+rng.normal(0.,0.015)))
    def add_bin(w,r):
        nonlocal score,wsum; w=jw(w); score += w*logit(r).astype(np.float64); wsum += abs(w)
    def add_cont(w,r):
        nonlocal score,wsum; w=jw(w); score += w*standardize(r); wsum += abs(w)
    # Binary auxiliary events: sparse actions are high-precision preference signals.
    bin_weights={'is_click':1.00,'is_like':0.80,'is_follow':0.65,'is_comment':0.55,'is_forward':0.45,'is_hate':-0.45,'is_profile_enter':0.35,'is_profile_stay':0.35}
    key1=[(1,1.00,80.),(2,0.85,100.),(3,0.35,500.),(4,0.20,500.)]
    key2=[(0,1,maxv,1.30,40.),(0,2,maxa,1.05,60.),(0,3,maxt,0.80,100.),(1,3,maxt,0.40,120.),(2,3,maxt,0.35,150.),(1,4,maxd,0.25,150.)]
    for c in bin_cols:
        v=vals[c].astype(np.float32); gm=float(np.clip(v.mean(),1e-5,1-1e-5)); bw=bin_weights.get(c.lower(),0.4)
        for col,w,a in key1:
            add_bin(bw*w, one_rate(Xtr[:,col].astype(np.int64),v,Xtg[:,col].astype(np.int64),gm,a,train=is_train))
        for ca,cb,nb,w,a in key2:
            add_bin(bw*w, pair_rate(Xtr[:,ca].astype(np.int64),Xtr[:,cb].astype(np.int64),v,Xtg[:,ca].astype(np.int64),Xtg[:,cb].astype(np.int64),gm,nb,a,train=is_train))
    # Continuous dwell/play magnitudes as preference strength, not as a reconstructed label.
    cont_weights={'play_time_ms':0.90,'profile_stay_time':0.25,'comment_stay_time':0.20}
    for c in cont_cols:
        v=vals[c].astype(np.float32); gm=float(v.mean()); cw=cont_weights.get(c.lower(),0.25)
        for col,w,a in [(1,1.0,80.),(2,0.8,100.),(3,0.25,500.)]:
            add_cont(cw*w, one_mean(Xtr[:,col].astype(np.int64),v,Xtg[:,col].astype(np.int64),gm,a,train=is_train))
        for ca,cb,nb,w,a in [(0,1,maxv,1.25,40.),(0,2,maxa,1.0,60.),(0,3,maxt,0.70,100.),(1,3,maxt,0.35,150.),(2,3,maxt,0.30,150.)]:
            add_cont(cw*w, pair_mean(Xtr[:,ca].astype(np.int64),Xtr[:,cb].astype(np.int64),v,Xtg[:,ca].astype(np.int64),Xtg[:,cb].astype(np.int64),gm,nb,a,train=is_train))
    print(f"aux_feedback dim={dim} members={len(bin_cols)}bin/{len(cont_cols)}cont wsum={wsum:.3f}")
    return (score/max(wsum,1e-9)).astype(np.float64), users

def cached(name,seed,split,expected_len,fn,prefix='009'):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'{prefix}_{name}_{split}_seed{seed}.npy'); upath=os.path.join('pred_cache',f'{prefix}_{name}_{split}_seed{seed}_users.npy')
    if os.path.isfile(path) and os.path.isfile(upath):
        p=np.load(path); u=np.load(upath,allow_pickle=True)
        if len(p)==expected_len: print(f"loaded cache {path}"); return p.astype(np.float64),u
    p,u=fn(); np.save(path,p); np.save(upath,np.asarray(u)); return p,u

def per_user_percentile(p,users):
    p=p.astype(np.float64); users=np.asarray(users); order=np.argsort(users,kind='stable'); su=users[order]
    bounds=np.r_[0,np.flatnonzero(su[1:]!=su[:-1])+1,len(su)]; out=np.empty_like(p)
    for a,b in zip(bounds[:-1],bounds[1:]):
        idx=order[a:b]; vals=p[idx]; sidx=idx[np.argsort(vals,kind='stable')]; n=len(sidx)
        if n<=1: out[sidx]=0.0
        else: out[sidx]=np.arange(n,dtype=np.float64)/(n-1.0)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args(); torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; cache_split='dev'
    else:
        splits=load(a.data_dir); target=a.split; cache_split=a.split
    print({k:len(v) for k,v in splits.items()},f"fields={FIELDS}"); n=len(splits[target])
    ps,users=cached('softmax',a.seed,cache_split,n,lambda:train_softmax_member(splits,target,a.seed,a.device),prefix='009')
    ph,_=cached('stats',a.seed,cache_split,n,lambda:stats_member(splits,target,a.seed),prefix='009')
    pa,_=cached('aux_feedback',a.seed,cache_split,n,lambda:aux_feedback_member(splits,target,a.seed,a.data_dir),prefix='019')
    base=0.50*per_user_percentile(ps,users)+0.50*per_user_percentile(ph,users)
    preds=0.70*base + 0.30*per_user_percentile(pa,users)
    if a.out: np.save(a.out,preds.astype(np.float64)); print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else: print(preds[:10])
