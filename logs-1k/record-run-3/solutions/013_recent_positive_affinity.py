"""Recent-positive affinity rank-fusion ensemble.

Parent 11 showed recency-smoothed rates raise GAUC but hurt nDCG@5.  This keeps
cached 009 softmax/listwise FM and all-time rate stats, and replaces the bad
rate-style recency member with a sequence-like member: for each candidate, count
how often the same user recently had positive long_view interactions with the
same video/author/tab/duration (plus item/author positive popularity).  Counts
use TRAIN labels only and are converted to within-user percentile ranks before a
readable 30% blend.
"""
import argparse, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate          # noqa

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

# all-time history-rate member, identical to 009
def logit(p): p=np.clip(p,1e-5,1-1e-5); return np.log(p/(1-p))
def pair_keys(a,b,nb): return a.astype(np.int64)*np.int64(nb)+b.astype(np.int64)
def one_rate(ktr,ytr,ktg,gm,alpha,train=False):
    n=int(max(ktr.max(initial=0),ktg.max(initial=0)))+1; cnt=np.bincount(ktr,minlength=n).astype(np.float32); sm=np.bincount(ktr,weights=ytr,minlength=n).astype(np.float32)
    if train: c=np.maximum(cnt[ktr]-1.,0.); s=sm[ktr]-ytr
    else: c=cnt[ktg]; s=sm[ktg]
    return (s+alpha*gm)/(c+alpha)
def pair_rate(atr,btr,ytr,atg,btg,gm,nb,alpha,train=False):
    ktr=pair_keys(atr,btr,nb); uniq,inv=np.unique(ktr,return_inverse=True); cnt=np.bincount(inv).astype(np.float32); sm=np.bincount(inv,weights=ytr).astype(np.float32)
    if train: c=np.maximum(cnt[inv]-1.,0.); s=sm[inv]-ytr
    else:
        ktg=pair_keys(atg,btg,nb); pos=np.searchsorted(uniq,ktg); ok=(pos<len(uniq))&(uniq[np.minimum(pos,len(uniq)-1)]==ktg)
        c=np.zeros(len(ktg),dtype=np.float32); s=np.zeros(len(ktg),dtype=np.float32)
        if ok.any(): c[ok]=cnt[pos[ok]]; s[ok]=sm[pos[ok]]
    return (s+alpha*gm)/(c+alpha)
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

# recent positive affinity member
def date_weights(train_rows, half_life_days=4.0):
    dates=np.asarray([int(r[0]) for r in train_rows],dtype=np.int64)
    uniq=np.asarray(sorted(set(dates.tolist())),dtype=np.int64); mp={int(d):i for i,d in enumerate(uniq)}
    di=np.asarray([mp[int(d)] for d in dates],dtype=np.float32); age=float(di.max())-di
    return np.power(0.5, age/half_life_days).astype(np.float32)

def one_sum(ktr,weights,ktg,train=False):
    n=int(max(ktr.max(initial=0),ktg.max(initial=0)))+1
    sm=np.bincount(ktr,weights=weights,minlength=n).astype(np.float32)
    if train: return np.maximum(sm[ktr]-weights,0.)
    return sm[ktg]

def pair_sum(atr,btr,weights,atg,btg,nb,train=False):
    ktr=pair_keys(atr,btr,nb); uniq,inv=np.unique(ktr,return_inverse=True)
    sm=np.bincount(inv,weights=weights).astype(np.float32)
    if train: return np.maximum(sm[inv]-weights,0.)
    ktg=pair_keys(atg,btg,nb); pos=np.searchsorted(uniq,ktg); ok=(pos<len(uniq))&(uniq[np.minimum(pos,len(uniq)-1)]==ktg)
    out=np.zeros(len(ktg),dtype=np.float32)
    if ok.any(): out[ok]=sm[pos[ok]]
    return out

def recpos_member(splits,target,seed):
    enc,dim=encode(splits); Xtr,ytr,_=enc['train']; Xtg,_,users=enc[target]
    ytr=ytr.astype(np.float32); is_train=target=='train'; rng=np.random.default_rng(seed)
    # positives only; mix a recency-decayed and an all-time channel so older repeated affinities still count
    dw=date_weights(splits['train'],4.0)
    wpos=(ytr*(0.65*dw+0.35)).astype(np.float32)
    score=np.zeros(len(Xtg),dtype=np.float64); wsum=0.0
    def jw(w): return float(w*(1.+rng.normal(0.,0.02)))
    def add(w,c):
        nonlocal score,wsum; w=jw(w); score += w*np.log1p(c.astype(np.float64)); wsum += w
    maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1; maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    # same-user positive history is the sequence-like signal
    add(3.0, pair_sum(Xtr[:,0].astype(np.int64),Xtr[:,1].astype(np.int64),wpos,Xtg[:,0].astype(np.int64),Xtg[:,1].astype(np.int64),maxv,train=is_train))
    add(2.4, pair_sum(Xtr[:,0].astype(np.int64),Xtr[:,2].astype(np.int64),wpos,Xtg[:,0].astype(np.int64),Xtg[:,2].astype(np.int64),maxa,train=is_train))
    add(0.8, pair_sum(Xtr[:,0].astype(np.int64),Xtr[:,3].astype(np.int64),wpos,Xtg[:,0].astype(np.int64),Xtg[:,3].astype(np.int64),maxt,train=is_train))
    add(0.5, pair_sum(Xtr[:,0].astype(np.int64),Xtr[:,4].astype(np.int64),wpos,Xtg[:,0].astype(np.int64),Xtg[:,4].astype(np.int64),maxd,train=is_train))
    # global popularity among positives helps ties within a user's candidate set
    add(0.8, one_sum(Xtr[:,1].astype(np.int64),wpos,Xtg[:,1].astype(np.int64),train=is_train))
    add(0.8, one_sum(Xtr[:,2].astype(np.int64),wpos,Xtg[:,2].astype(np.int64),train=is_train))
    add(0.3, pair_sum(Xtr[:,2].astype(np.int64),Xtr[:,3].astype(np.int64),wpos,Xtg[:,2].astype(np.int64),Xtg[:,3].astype(np.int64),maxt,train=is_train))
    print(f"recpos dim={dim} positives={int(ytr.sum())} mean_wpos={float(wpos.mean()):.6f}")
    return (score/wsum).astype(np.float64), users

def cached(name,seed,split,expected_len,fn,prefix='009'):
    os.makedirs('pred_cache',exist_ok=True)
    path=os.path.join('pred_cache',f'{prefix}_{name}_{split}_seed{seed}.npy'); upath=os.path.join('pred_cache',f'{prefix}_{name}_{split}_seed{seed}_users.npy')
    if os.path.isfile(path) and os.path.isfile(upath):
        p=np.load(path); u=np.load(upath,allow_pickle=True)
        if len(p)==expected_len: print(f"loaded cache {path}"); return p.astype(np.float64),u
    p,u=fn(); np.save(path,p); np.save(upath,np.asarray(u)); return p,u

def per_user_percentile(p,users):
    p=p.astype(np.float64); users=np.asarray(users); order=np.argsort(users,kind='stable'); su=users[order]
    bounds=np.r_[0,np.flatnonzero(su[1:]!=su[:-1])+1,len(su)]; out=np.empty_like(p)
    for a,b in zip(bounds[:-1],bounds[1:]):
        idx=order[a:b]; sidx=idx[np.argsort(p[idx],kind='stable')]; n=len(sidx)
        out[sidx]=0.0 if n<=1 else np.arange(n,dtype=np.float64)/(n-1.0)
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
    pa,_=cached('stats',a.seed,cache_split,n,lambda:stats_member(splits,target,a.seed),prefix='009')
    pr,_=cached('recpos_hl4',a.seed,cache_split,n,lambda:recpos_member(splits,target,a.seed),prefix='013')
    preds=0.35*per_user_percentile(ps,users)+0.35*per_user_percentile(pa,users)+0.30*per_user_percentile(pr,users)
    if a.out:
        np.save(a.out,preds.astype(np.float64)); print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else: print(preds[:10])
