"""Debug auxiliary-feedback direction with a standard multi-task FM.

Node 18 used auxiliary raw feedback as a separate aggregate rank member and badly
hurt ranking.  This version uses the auxiliary signals only as TRAINING targets:
a shared FM is trained with the same per-user softmax long-view ranking loss as
node 10 plus small BCE heads for click/like/follow/comment/forward/profile
signals.  At prediction time only the main long-view head is used, then fused
50/50 with the unchanged cached historical-rate member.
"""
import argparse, csv, glob, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate          # noqa  (early stopping only)

class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_tasks=1, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros((n_tasks,dim),dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros(n_tasks,dtype=torch.float32))
    def forward(self,X,task=0):
        E=self.V[X]; S=E.sum(1); inter=0.5*((S**2).sum(1)-(E**2).sum((1,2)))
        return self.b[task]+self.W[task,X].sum(1)+inter
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device),0).cpu().numpy())
        return np.concatenate(out)

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

def read_aux_binary_train(data_dir,n):
    path=find_train_log(data_dir); print('reading aux train labels from',path)
    wanted=['is_click','is_like','is_follow','is_comment','is_forward','is_profile_enter']
    with open(path,newline='') as f:
        rdr=csv.DictReader(f); lower={c.lower():c for c in (rdr.fieldnames or [])}
        cols=[lower[c] for c in wanted if c in lower]
        arr=np.zeros((n,len(cols)),dtype=np.float32)
        for i,row in enumerate(rdr):
            if i>=n: break
            for j,c in enumerate(cols): arr[i,j]=1.0 if to_float(row.get(c,''))>0 else 0.0
    print('aux binary tasks',cols,'rates',np.round(arr.mean(0),5).tolist() if len(cols) else [])
    return arr, cols

def listwise_mtl_loss(model,Xtr_t,ytr_t,aux_t,posw_t,groups,device,aux_weight=0.18,max_rows=24000):
    losses=[]; rows=[]; nrows=0
    for g in groups:
        if nrows>=max_rows and losses: break
        xt=Xtr_t[g].to(device); s=model(xt,0); yb=ytr_t[g].to(device); pos=yb>0.5
        losses.append(torch.logsumexp(s,0)-torch.logsumexp(s[pos],0)); rows.append(g); nrows+=len(g)
    loss=torch.stack(losses).mean()
    if aux_t is not None and aux_t.shape[1]>0 and aux_weight>0:
        idx=np.concatenate(rows)
        xt=Xtr_t[idx].to(device); yt=aux_t[idx].to(device)
        aux_losses=[]
        for j in range(yt.shape[1]):
            logits=model(xt,j+1)
            aux_losses.append(F.binary_cross_entropy_with_logits(logits,yt[:,j],pos_weight=posw_t[j].to(device)))
        loss = loss + aux_weight*torch.stack(aux_losses).mean()
    return loss

def train_mtl_member(splits,target,seed,data_dir,device='cpu'):
    enc,dim=encode(splits); Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    aux,cols=read_aux_binary_train(data_dir,len(Xtr))
    n_tasks=1+aux.shape[1]
    model=MultiTaskFM(dim,n_tasks=n_tasks,k=16,seed=seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=0.003)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); groups=make_user_groups(utr,ytr)
    aux_t=torch.from_numpy(aux.astype(np.float32)) if aux.shape[1]>0 else None
    if aux.shape[1]>0:
        rates=np.clip(aux.mean(0),1e-5,1-1e-5); posw=np.clip((1-rates)/rates,1.0,20.0).astype(np.float32)
        posw_t=torch.from_numpy(posw); print('aux pos_weight',np.round(posw,2).tolist())
    else:
        posw_t=torch.ones(0,dtype=torch.float32)
    rng=np.random.default_rng(seed); best=-1.; best_state=None; bad=0
    for ep in range(1,81):
        perm=rng.permutation(len(groups)); losses=[]; model.train(); t0=time.time()
        for i in range(0,len(perm),48):
            bg=[groups[j] for j in perm[i:i+48]]; opt.zero_grad(set_to_none=True)
            loss=listwise_mtl_loss(model,Xtr_t,ytr_t,aux_t,posw_t,bg,device); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device)); print(f"mtl ep {ep:02d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
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

def cached(name,seed,split,expected_len,fn,prefix):
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
    pm,users=cached('mtl_softmax',a.seed,cache_split,n,lambda:train_mtl_member(splits,target,a.seed,a.data_dir,a.device),'020')
    ph,_=cached('stats',a.seed,cache_split,n,lambda:stats_member(splits,target,a.seed),'009')
    preds=0.50*per_user_percentile(pm,users)+0.50*per_user_percentile(ph,users)
    if a.out: np.save(a.out,preds.astype(np.float64)); print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else: print(preds[:10])
