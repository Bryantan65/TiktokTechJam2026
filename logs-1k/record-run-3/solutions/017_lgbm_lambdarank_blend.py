"""LightGBM LambdaRank member blended with the node-10 rank-fusion ensemble.

This debugs the earlier LambdaRank attempt by keeping every LightGBM query below
10k rows via same-user chunks, and by feeding it the same leakage-safe historical
rate features that worked as a direct predictor.  The ranker is then given a
readable 30% vote against the cached softmax/stat rank-fusion backbone.
"""
import argparse, os, sys, time
import numpy as np
import torch
import lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate          # noqa  (early stopping only for torch member)

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

def logit(p):
    p=np.clip(p,1e-5,1-1e-5); return np.log(p/(1-p))

def one_rate_count(ktr,ytr,ktg,gm,alpha,train=False):
    n=int(max(ktr.max(initial=0),ktg.max(initial=0)))+1
    cnt=np.bincount(ktr,minlength=n).astype(np.float32); sm=np.bincount(ktr,weights=ytr,minlength=n).astype(np.float32)
    if train: c=cnt[ktr]-1.; s=sm[ktr]-ytr
    else: c=cnt[ktg]; s=sm[ktg]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha), c

def pair_keys(a,b,nb): return a.astype(np.int64)*np.int64(nb)+b.astype(np.int64)

def pair_rate_count(atr,btr,ytr,atg,btg,gm,nb,alpha,train=False):
    ktr=pair_keys(atr,btr,nb); uniq,inv=np.unique(ktr,return_inverse=True)
    cnt=np.bincount(inv).astype(np.float32); sm=np.bincount(inv,weights=ytr).astype(np.float32)
    if train:
        c=cnt[inv]-1.; s=sm[inv]-ytr
    else:
        ktg=pair_keys(atg,btg,nb); pos=np.searchsorted(uniq,ktg); ok=(pos<len(uniq))&(uniq[np.minimum(pos,len(uniq)-1)]==ktg)
        c=np.zeros(len(ktg),dtype=np.float32); s=np.zeros(len(ktg),dtype=np.float32)
        if ok.any(): c[ok]=cnt[pos[ok]]; s[ok]=sm[pos[ok]]
    c=np.maximum(c,0.); return (s+alpha*gm)/(c+alpha), c

def stats_member(splits,target,seed):
    enc,dim=encode(splits); Xtr,ytr,_=enc['train']; Xtg,_,users=enc[target]; ytr=ytr.astype(np.float32); gm=float(ytr.mean()); is_train=target=='train'; rng=np.random.default_rng(seed)
    score=np.zeros(len(Xtg),dtype=np.float64); wsum=0.
    def jw(w): return float(w*(1.+rng.normal(0.,0.015)))
    def add(w,r):
        nonlocal score,wsum; w=jw(w); score+=w*logit(r).astype(np.float64); wsum+=w
    for col,w,a in [(1,1.4,30.),(2,1.1,30.),(3,0.9,80.),(4,0.5,80.)]:
        r,_=one_rate_count(Xtr[:,col].astype(np.int64),ytr,Xtg[:,col].astype(np.int64),gm,a,train=is_train); add(w,r)
    maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1; maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    specs=[(Xtr[:,0],Xtr[:,1],Xtg[:,0],Xtg[:,1],maxv,1.6,15.),(Xtr[:,0],Xtr[:,2],Xtg[:,0],Xtg[:,2],maxa,1.3,25.),(Xtr[:,0],Xtr[:,3],Xtg[:,0],Xtg[:,3],maxt,1.0,40.),(Xtr[:,1],Xtr[:,3],Xtg[:,1],Xtg[:,3],maxt,0.6,40.),(Xtr[:,2],Xtr[:,3],Xtg[:,2],Xtg[:,3],maxt,0.5,40.),(Xtr[:,1],Xtr[:,4],Xtg[:,1],Xtg[:,4],maxd,0.4,40.),(Xtr[:,2],Xtr[:,4],Xtg[:,2],Xtg[:,4],maxd,0.3,40.)]
    for atr,btr,atg,btg,nb,w,a in specs:
        r,_=pair_rate_count(atr.astype(np.int64),btr.astype(np.int64),ytr,atg.astype(np.int64),btg.astype(np.int64),gm,nb,a,train=is_train); add(w,r)
    print(f"stats dim={dim} gm={gm:.6f}"); return (score/wsum).astype(np.float64), users

def lgb_features_from_encoded(Xtr,ytr,Xtg):
    ytr=ytr.astype(np.float32); gm=float(ytr.mean()); feats_tr=[]; feats_tg=[]
    # raw encoded fields as categorical / ordinal signals
    for c in range(Xtr.shape[1]):
        feats_tr.append(Xtr[:,c].astype(np.float32)); feats_tg.append(Xtg[:,c].astype(np.float32))
    single_specs=[(1,30.),(2,30.),(3,80.),(4,80.)]
    for col,a in single_specs:
        r,cnt=one_rate_count(Xtr[:,col].astype(np.int64),ytr,Xtr[:,col].astype(np.int64),gm,a,train=True)
        rt,ct=one_rate_count(Xtr[:,col].astype(np.int64),ytr,Xtg[:,col].astype(np.int64),gm,a,train=False)
        feats_tr += [logit(r).astype(np.float32), np.log1p(cnt).astype(np.float32)]
        feats_tg += [logit(rt).astype(np.float32), np.log1p(ct).astype(np.float32)]
    maxv=int(max(Xtr[:,1].max(),Xtg[:,1].max()))+1; maxa=int(max(Xtr[:,2].max(),Xtg[:,2].max()))+1; maxt=int(max(Xtr[:,3].max(),Xtg[:,3].max()))+1; maxd=int(max(Xtr[:,4].max(),Xtg[:,4].max()))+1
    pair_specs=[(0,1,maxv,15.),(0,2,maxa,25.),(0,3,maxt,40.),(1,3,maxt,40.),(2,3,maxt,40.),(1,4,maxd,40.),(2,4,maxd,40.)]
    for a,b,nb,alpha in pair_specs:
        r,cnt=pair_rate_count(Xtr[:,a].astype(np.int64),Xtr[:,b].astype(np.int64),ytr,Xtr[:,a].astype(np.int64),Xtr[:,b].astype(np.int64),gm,nb,alpha,train=True)
        rt,ct=pair_rate_count(Xtr[:,a].astype(np.int64),Xtr[:,b].astype(np.int64),ytr,Xtg[:,a].astype(np.int64),Xtg[:,b].astype(np.int64),gm,nb,alpha,train=False)
        feats_tr += [logit(r).astype(np.float32), np.log1p(cnt).astype(np.float32)]
        feats_tg += [logit(rt).astype(np.float32), np.log1p(ct).astype(np.float32)]
    return np.column_stack(feats_tr).astype(np.float32), np.column_stack(feats_tg).astype(np.float32)

def make_lgb_order_groups(users, y, seed, max_group=8000):
    users=np.asarray(users); y=np.asarray(y); order=np.argsort(users,kind='stable'); su=users[order]
    bounds=np.r_[0,np.flatnonzero(su[1:]!=su[:-1])+1,len(su)]
    rng=np.random.default_rng(seed); parts=[]; groups=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        idx=order[a:b].copy(); rng.shuffle(idx)
        # Chunk oversized users.  Keep chunks mixed by randomization and skip chunks
        # with no pairwise label contrast because they contribute no lambdas.
        for st in range(0,len(idx),max_group):
            ch=idx[st:st+max_group]
            if len(ch)<2: continue
            sy=float(y[ch].sum())
            if sy<=0.0 or sy>=len(ch): continue
            parts.append(ch); groups.append(len(ch))
    return np.concatenate(parts), groups

def train_lgbm_rank_member(splits,target,seed):
    enc,dim=encode(splits); Xtr,ytr,utr=enc['train']; Xtg,_,users=enc[target]
    t0=time.time(); Ftr,Ftg=lgb_features_from_encoded(Xtr,ytr,Xtg); print(f"lgb features {Ftr.shape}->{Ftg.shape} built in {time.time()-t0:.1f}s")
    ord_idx,groups=make_lgb_order_groups(utr,ytr,seed,max_group=8000)
    print(f"lgb rows={len(ord_idx)} groups={len(groups)} max_group={max(groups)} pos={float(ytr[ord_idx].mean()):.4f}")
    dtrain=lgb.Dataset(Ftr[ord_idx],label=ytr[ord_idx].astype(np.float32),group=groups,free_raw_data=True)
    params=dict(objective='lambdarank',metric='ndcg',ndcg_eval_at=[5],label_gain=[0,1],learning_rate=0.045,
                num_leaves=63,min_data_in_leaf=250,feature_fraction=0.90,bagging_fraction=0.85,bagging_freq=1,
                lambda_l2=8.0,verbosity=-1,seed=seed,num_threads=4,force_col_wise=True)
    booster=lgb.train(params,dtrain,num_boost_round=180)
    pred=booster.predict(Ftg,num_iteration=booster.best_iteration).astype(np.float64)
    return pred, users

def cached_node10(name,seed,split,expected_len,fn):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'009_{name}_{split}_seed{seed}.npy'); upath=os.path.join('pred_cache',f'009_{name}_{split}_seed{seed}_users.npy')
    if os.path.isfile(path) and os.path.isfile(upath):
        p=np.load(path); u=np.load(upath,allow_pickle=True)
        if len(p)==expected_len: print(f"loaded cache {path}"); return p.astype(np.float64),u
    p,u=fn(); np.save(path,p); np.save(upath,np.asarray(u)); return p,u

def cached_lgbm(seed,split,expected_len,fn):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'017_lgbmrank_{split}_seed{seed}.npy'); upath=os.path.join('pred_cache',f'017_lgbmrank_{split}_seed{seed}_users.npy')
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
    ps,users=cached_node10('softmax',a.seed,cache_split,n,lambda:train_softmax_member(splits,target,a.seed,a.device))
    ph,_=cached_node10('stats',a.seed,cache_split,n,lambda:stats_member(splits,target,a.seed))
    pl,_=cached_lgbm(a.seed,cache_split,n,lambda:train_lgbm_rank_member(splits,target,a.seed))
    base=0.50*per_user_percentile(ps,users)+0.50*per_user_percentile(ph,users)
    rl=per_user_percentile(pl,users)
    preds=0.70*base + 0.30*rl
    if a.out: np.save(a.out,preds.astype(np.float64)); print(f"wrote {len(preds):,d} predictions for split={a.split}")
    else: print(preds[:10])
