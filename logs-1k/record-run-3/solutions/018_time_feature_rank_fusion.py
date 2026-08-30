"""Add raw timestamp/hour features as a readable rank-fusion member.

Node 10 fused a listwise FM with historical ID/tab/duration rates.  This script
keeps those cached members unchanged and adds a 30% member built from TRAIN-only
smoothed rates involving raw CSV hourmin and tuple dates (user-hour, tab-hour,
item-hour, author-hour, day-of-week crosses).  No target feedback columns or raw
long_view labels are read.
"""
import argparse, csv, glob, os, sys, time, datetime
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
def one_rate_count(ktr,ytr,ktg,gm,alpha,train=False):
    n=int(max(ktr.max(initial=0),ktg.max(initial=0)))+1; cnt=np.bincount(ktr,minlength=n).astype(np.float32); sm=np.bincount(ktr,weights=ytr,minlength=n).astype(np.float32)
    if train: c=cnt[ktr]-1.; s=sm[ktr]-ytr
    else: c=cnt[ktg]; s=sm[ktg]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha), c
def pair_keys(a,b,nb): return a.astype(np.int64)*np.int64(nb)+b.astype(np.int64)
def pair_rate(atr,btr,ytr,atg,btg,gm,nb,alpha,train=False):
    ktr=pair_keys(atr,btr,nb); uniq,inv=np.unique(ktr,return_inverse=True); cnt=np.bincount(inv).astype(np.float32); sm=np.bincount(inv,weights=ytr).astype(np.float32)
    if train: c=cnt[inv]-1.; s=sm[inv]-ytr
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

def find_log_file(data_dir, marker):
    pats=[os.path.join(data_dir, f'log_standard_{marker}_1k.csv'), os.path.join(data_dir, f'log_standard_{marker}*.csv')]
    for pat in pats:
        fs=glob.glob(pat)
        fs=[f for f in fs if '_pure' not in os.path.basename(f)] or fs
        if fs: return sorted(fs)[0]
    return None

def hm_to_bin(hm):
    try: v=int(float(hm))
    except Exception: return 0,0,0
    h=max(0,min(23,v//100)); m=max(0,min(59,v%100))
    return h, h*2+(m>=30), h*4+min(3,m//15)

def read_hour_bins(path, n=None, skip=0):
    h24=[]; h48=[]; h96=[]
    with open(path, newline='') as f:
        rdr=csv.DictReader(f)
        cols={c.lower():c for c in (rdr.fieldnames or [])}
        hc=cols.get('hourmin') or cols.get('time') or cols.get('hour_min')
        for i,row in enumerate(rdr):
            if i<skip: continue
            if n is not None and len(h24)>=n: break
            a,b,c=hm_to_bin(row.get(hc,'0') if hc else '0')
            h24.append(a); h48.append(b); h96.append(c)
    return np.asarray(h24,dtype=np.int64),np.asarray(h48,dtype=np.int64),np.asarray(h96,dtype=np.int64)

def dates_to_dow(rows):
    vals=np.asarray([int(r[0]) for r in rows],dtype=np.int64); out=np.empty(len(vals),dtype=np.int64); cache={}
    for d in np.unique(vals):
        try: cache[int(d)]=datetime.datetime.strptime(str(int(d)),'%Y%m%d').weekday()
        except Exception: cache[int(d)]=0
    for i,d in enumerate(vals): out[i]=cache[int(d)]
    return out

def load_time_arrays(data_dir,splits,target,cache_split):
    ntr=len(splits['train']); ntg=len(splits[target])
    f1=find_log_file(data_dir,'4_08_to_4_21'); f2=find_log_file(data_dir,'4_22_to_5_08')
    if cache_split=='dev' or f2 is None:
        # devdata is a date cut from the first chronological log; this preserves order.
        tr=read_hour_bins(f1,ntr,0); tg=read_hour_bins(f1,ntg,ntr)
    else:
        tr=read_hour_bins(f1,ntr,0)
        skip=0 if target=='valid' else len(splits.get('valid',[])) if target=='test' else 0
        tg=read_hour_bins(f2,ntg,skip)
    if len(tr[0])!=ntr or len(tg[0])!=ntg:
        print('WARNING hour CSV alignment failed; falling back to zeros', len(tr[0]), ntr, len(tg[0]), ntg)
        tr=(np.zeros(ntr,dtype=np.int64),np.zeros(ntr,dtype=np.int64),np.zeros(ntr,dtype=np.int64))
        tg=(np.zeros(ntg,dtype=np.int64),np.zeros(ntg,dtype=np.int64),np.zeros(ntg,dtype=np.int64))
    return tr,tg

def time_stats_member(splits,target,seed,data_dir,cache_split):
    enc,dim=encode(splits); Xtr,ytr,_=enc['train']; Xtg,_,users=enc[target]; ytr=ytr.astype(np.float32); gm=float(ytr.mean()); is_train=target=='train'
    (h24,h48,h96),(th24,th48,th96)=load_time_arrays(data_dir,splits,target,cache_split)
    dow=dates_to_dow(splits['train']); tdow=dates_to_dow(splits[target])
    rng=np.random.default_rng(seed+91017); score=np.zeros(len(Xtg),dtype=np.float64); wsum=0.0
    def jw(w): return float(w*(1.+rng.normal(0.,0.015)))
    def add(w,r):
        nonlocal score,wsum; w=jw(w); score += w*logit(r).astype(np.float64); wsum += w
    # global temporal priors
    for tr,tg,nb,w,a in [(h24,th24,24,0.25,500.),(h48,th48,48,0.35,500.),(h96,th96,96,0.25,500.),(dow,tdow,7,0.25,500.)]:
        add(w,one_rate(tr.astype(np.int64),ytr,tg.astype(np.int64),gm,a,train=is_train))
    maxu=int(max(Xtr[:,0].max(),Xtg[:,0].max()))+1; maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1
    maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    tabh48=Xtr[:,3].astype(np.int64)*48+h48; ttabh48=Xtg[:,3].astype(np.int64)*48+th48; ntab48=maxt*48
    durh24=Xtr[:,4].astype(np.int64)*24+h24; tdurh24=Xtg[:,4].astype(np.int64)*24+th24; ndur24=maxd*24
    specs=[
        (Xtr[:,0],h24,Xtg[:,0],th24,24,1.20,80.),
        (Xtr[:,0],h48,Xtg[:,0],th48,48,0.90,100.),
        (Xtr[:,0],dow,Xtg[:,0],tdow,7,0.65,80.),
        (Xtr[:,0],tabh48,Xtg[:,0],ttabh48,ntab48,0.85,120.),
        (Xtr[:,1],h24,Xtg[:,1],th24,24,0.75,60.),
        (Xtr[:,2],h24,Xtg[:,2],th24,24,0.75,100.),
        (Xtr[:,3],h48,Xtg[:,3],th48,48,0.75,300.),
        (Xtr[:,4],h24,Xtg[:,4],th24,24,0.35,300.),
        (Xtr[:,1],dow,Xtg[:,1],tdow,7,0.35,80.),
        (Xtr[:,2],dow,Xtg[:,2],tdow,7,0.35,120.),
        (durh24,Xtr[:,3],tdurh24,Xtg[:,3],maxt,0.30,300.),
    ]
    for atr,btr,atg,btg,nb,w,a in specs:
        add(w,pair_rate(atr.astype(np.int64),btr.astype(np.int64),ytr,atg.astype(np.int64),btg.astype(np.int64),gm,nb,a,train=is_train))
    print(f"time_stats dim={dim} gm={gm:.6f} h24_train={np.bincount(h24,minlength=24)[:6].tolist()}")
    return (score/wsum).astype(np.float64), users

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
    pt,_=cached('time_stats',a.seed,cache_split,n,lambda:time_stats_member(splits,target,a.seed,a.data_dir,cache_split),prefix='018')
    base=0.50*per_user_percentile(ps,users)+0.50*per_user_percentile(ph,users)
    preds=0.70*base + 0.30*per_user_percentile(pt,users)
    if a.out: np.save(a.out,preds.astype(np.float64)); print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else: print(preds[:10])
