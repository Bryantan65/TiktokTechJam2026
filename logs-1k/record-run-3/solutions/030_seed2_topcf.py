"""Seed-2 node25 with top-gated CF.

Node 28 showed averaging stabilizes GAUC but diluted the very strong seed-2 top-5
ranking from node 25.  This keeps the node-25 members unchanged/cached, fixes the
member seed to 2, and replaces the uniform 28% CF blend with a base-rank gated CF
weight: less CF in the tail for GAUC, more CF at the top for nDCG@5.
"""
import argparse, csv, glob, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate          # noqa

class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_tasks=1, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros((n_tasks,dim),dtype=torch.float32)); self.b=torch.nn.Parameter(torch.zeros(n_tasks,dtype=torch.float32))
    def forward(self,X,task=0):
        E=self.V[X]; S=E.sum(1); inter=0.5*((S**2).sum(1)-(E**2).sum((1,2)))
        return self.b[task]+self.W[task,X].sum(1)+inter
    @torch.no_grad()
    def predict_task(self,X,task=0,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device),task).cpu().numpy())
        return np.concatenate(out)
    @torch.no_grad()
    def predict_all(self,X,bs=200000,device='cpu'):
        return [self.predict_task(X,t,bs,device) for t in range(self.W.shape[0])]

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
    except Exception: return 0.0

def read_aux_binary_train(data_dir,n):
    path=find_train_log(data_dir); print('reading aux train labels from',path)
    wanted=['is_click','is_like','is_follow','is_comment','is_forward','is_profile_enter']
    with open(path,newline='') as f:
        rdr=csv.DictReader(f); lower={c.lower():c for c in (rdr.fieldnames or [])}; cols=[lower[c] for c in wanted if c in lower]
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
        idx=np.concatenate(rows); xt=Xtr_t[idx].to(device); yt=aux_t[idx].to(device); aux_losses=[]
        for j in range(yt.shape[1]): aux_losses.append(F.binary_cross_entropy_with_logits(model(xt,j+1),yt[:,j],pos_weight=posw_t[j].to(device)))
        loss=loss+aux_weight*torch.stack(aux_losses).mean()
    return loss

def train_mtl_member(splits,target,seed,data_dir,device='cpu'):
    enc,dim=encode(splits); Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    aux,_=read_aux_binary_train(data_dir,len(Xtr)); model=MultiTaskFM(dim,n_tasks=1+aux.shape[1],k=16,seed=seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=0.003)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); groups=make_user_groups(utr,ytr)
    aux_t=torch.from_numpy(aux.astype(np.float32)) if aux.shape[1]>0 else None
    posw_t=torch.ones(0,dtype=torch.float32)
    if aux.shape[1]>0:
        rates=np.clip(aux.mean(0),1e-5,1-1e-5); posw_t=torch.from_numpy(np.clip((1-rates)/rates,1.0,20.0).astype(np.float32)); print('aux pos_weight',np.round(posw_t.numpy(),2).tolist())
    rng=np.random.default_rng(seed); best=-1.; best_state=None; bad=0
    for ep in range(1,81):
        perm=rng.permutation(len(groups)); losses=[]; model.train(); t0=time.time()
        for i in range(0,len(perm),48):
            bg=[groups[j] for j in perm[i:i+48]]; opt.zero_grad(set_to_none=True); loss=listwise_mtl_loss(model,Xtr_t,ytr_t,aux_t,posw_t,bg,device); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict_task(Xva,0,device=device)); print(f"mtl seed {seed} ep {ep:02d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5: best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=6: break
    model.load_state_dict(best_state); Xtg,_,users=enc[target]; outs=model.predict_all(Xtg,device=device); main=outs[0].astype(np.float64)
    if len(outs)>1:
        weights=np.array([0.55,0.20,0.20,0.15,0.15,0.10],dtype=np.float64)[:len(outs)-1]; aux_score=np.zeros_like(main)
        for w,o in zip(weights,outs[1:]): aux_score += w*o.astype(np.float64)
    else: aux_score=np.zeros_like(main)
    return main, aux_score, users

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
    def add(w,r):
        nonlocal score,wsum; w=float(w*(1.+rng.normal(0.,0.015))); score+=w*logit(r).astype(np.float64); wsum+=w
    for col,w,a in [(1,1.4,30.),(2,1.1,30.),(3,0.9,80.),(4,0.5,80.)]: add(w,one_rate(Xtr[:,col].astype(np.int64),ytr,Xtg[:,col].astype(np.int64),gm,a,train=is_train))
    maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1; maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    specs=[(Xtr[:,0],Xtr[:,1],Xtg[:,0],Xtg[:,1],maxv,1.6,15.),(Xtr[:,0],Xtr[:,2],Xtg[:,0],Xtg[:,2],maxa,1.3,25.),(Xtr[:,0],Xtr[:,3],Xtg[:,0],Xtg[:,3],maxt,1.0,40.),(Xtr[:,1],Xtr[:,3],Xtg[:,1],Xtg[:,3],maxt,0.6,40.),(Xtr[:,2],Xtr[:,3],Xtg[:,2],Xtg[:,3],maxt,0.5,40.),(Xtr[:,1],Xtr[:,4],Xtg[:,1],Xtg[:,4],maxd,0.4,40.),(Xtr[:,2],Xtr[:,4],Xtg[:,2],Xtg[:,4],maxd,0.3,40.)]
    for atr,btr,atg,btg,nb,w,a in specs: add(w,pair_rate(atr.astype(np.int64),btr.astype(np.int64),ytr,atg.astype(np.int64),btg.astype(np.int64),gm,nb,a,train=is_train))
    print(f"stats seed={seed} gm={gm:.6f}"); return (score/wsum).astype(np.float64), users

def svd_one(Xtr,ytr,Xtg,u_col,i_col,n_comp,seed):
    u_all=np.r_[Xtr[:,u_col],Xtg[:,u_col]].astype(np.int64); i_all=np.r_[Xtr[:,i_col],Xtg[:,i_col]].astype(np.int64)
    _,u_tr_all=np.unique(u_all,return_inverse=True); u_tr=u_tr_all[:len(Xtr)]; u_tg=u_tr_all[len(Xtr):]
    _,i_tr_all=np.unique(i_all,return_inverse=True); i_tr=i_tr_all[:len(Xtr)]; i_tg=i_tr_all[len(Xtr):]
    pos=ytr>0.5; R=sparse.csr_matrix((np.ones(int(pos.sum()),dtype=np.float32),(u_tr[pos],i_tr[pos])),shape=(int(u_tr_all.max())+1,int(i_tr_all.max())+1),dtype=np.float32)
    R.data=np.log1p(R.data); k=max(2,min(n_comp,min(R.shape)-1)); print(f"svd matrix {R.shape} nnz={R.nnz} k={k} seed={seed}")
    svd=TruncatedSVD(n_components=k,algorithm='randomized',n_iter=7,random_state=seed); U=svd.fit_transform(R).astype(np.float32); V=svd.components_.T.astype(np.float32)
    return np.einsum('ij,ij->i',U[u_tg],V[i_tg]).astype(np.float64)
def cf_svd_member(splits,target,seed):
    enc,dim=encode(splits); Xtr,ytr,_=enc['train']; Xtg,_,users=enc[target]
    return (0.70*svd_one(Xtr,ytr,Xtg,0,1,64,seed)+0.30*svd_one(Xtr,ytr,Xtg,0,2,48,seed+17)).astype(np.float64), users

def cached(name,seed,split,expected_len,fn,prefix):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'{prefix}_{name}_{split}_seed{seed}.npy'); upath=os.path.join('pred_cache',f'{prefix}_{name}_{split}_seed{seed}_users.npy')
    if os.path.isfile(path) and os.path.isfile(upath):
        p=np.load(path,allow_pickle=False); u=np.load(upath,allow_pickle=True)
        if len(p)==expected_len: print('loaded cache',path); return p,u
    p,u=fn(); np.save(path,p); np.save(upath,np.asarray(u)); return p,u

def per_user_percentile(p,users):
    p=np.asarray(p,dtype=np.float64); users=np.asarray(users); order=np.argsort(users,kind='stable'); su=users[order]
    bounds=np.r_[0,np.flatnonzero(su[1:]!=su[:-1])+1,len(su)]; out=np.empty_like(p)
    for a,b in zip(bounds[:-1],bounds[1:]):
        idx=order[a:b]; sidx=idx[np.argsort(p[idx],kind='stable')]; n=len(sidx); out[sidx]=0.0 if n<=1 else np.arange(n,dtype=np.float64)/(n-1.0)
    return out

def components(splits,target,cache_split,member_seed,data_dir,device,n):
    def train_pack():
        main,aux,users=train_mtl_member(splits,target,member_seed,data_dir,device)
        return np.vstack([main,aux]).T.astype(np.float64), users
    pack,users=cached('mtl_auxheads',member_seed,cache_split,n,train_pack,'022')
    ph,_=cached('stats',member_seed,cache_split,n,lambda:stats_member(splits,target,member_seed),'009')
    pc,_=cached('svd_cf',member_seed,cache_split,n,lambda:cf_svd_member(splits,target,member_seed),'026')
    return per_user_percentile(pack[:,0],users), per_user_percentile(pack[:,1],users), per_user_percentile(ph,users), per_user_percentile(pc,users), users

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args(); torch.manual_seed(2)
    print(f"loading {a.data_dir} ...")
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; cache_split='dev'
    else:
        splits=load(a.data_dir); target=a.split; cache_split=a.split
    print({k:len(v) for k,v in splits.items()},f"fields={FIELDS}"); n=len(splits[target])
    rm,ra,rh,rc,users=components(splits,target,cache_split,2,a.data_dir,a.device,n)
    base=0.50*rm+0.50*rh
    top_aux=base+0.35*np.power(np.clip(base,0.0,1.0),16.0)*(ra-base)
    wcf=0.20+0.25*np.power(np.clip(base,0.0,1.0),8.0)
    preds=(1.0-wcf)*top_aux+wcf*rc
    preds=per_user_percentile(preds,users)
    if a.out: np.save(a.out,preds.astype(np.float64)); print(f"wrote {len(preds):,d} fixed-seed2 top-CF predictions for split={a.split}")
    else: print(preds[:10])
