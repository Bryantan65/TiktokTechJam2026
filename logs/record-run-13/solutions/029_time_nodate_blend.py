"""Improve node 25 by removing the raw date categorical from the time-aware member.

The date values in valid/test are largely future categories relative to train, so their FM embeddings
are untrained random vectors.  Keep the stable cyclical-ish time signals (day-of-week and 4-hour
bucket) and blend 50/50 with the unchanged cached base bag from node 13.
"""
import argparse, csv, datetime as _dt, os, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32)); self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float32))
    def forward(self,X):
        E=self.V[X]; S=E.sum(1)
        return self.b+self.W[X].sum(1)+0.5*((S*S).sum(1)-(E*E).sum((1,2)))
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def fit_bce(splits,enc,dim,k=16,lr=0.001,l2=1e-6,epochs=40,bs=8192,patience=4,seed=0,device='cpu',verbose=True):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']
    model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    lossfn=torch.nn.BCEWithLogitsLoss(); Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr)
    rng=np.random.default_rng(seed); best=-1.0; best_state=None; bad=0
    for ep in range(1,epochs+1):
        idx=rng.permutation(len(ytr)); model.train(); losses=[]; t0=time.time()
        for i in range(0,len(idx),bs):
            sel=torch.from_numpy(idx[i:i+bs]); xb=Xtr_t[sel].to(device); yb=ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  BCE(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def make_user_pair_sources(y,users):
    pos=defaultdict(list); neg=defaultdict(list)
    for i,(yy,uu) in enumerate(zip(y,users)): (pos if yy>0.5 else neg)[uu].append(i)
    pos_idx=[]; neg_pools=[]
    for uu,ps in pos.items():
        ns=neg.get(uu)
        if ns:
            arr=np.asarray(ns,dtype=np.int64)
            for p in ps: pos_idx.append(p); neg_pools.append(arr)
    return np.asarray(pos_idx,dtype=np.int64),neg_pools

def sample_uniform_pairs(pos_idx,neg_pools,rng,neg_per_pos=3):
    total=len(pos_idx)*neg_per_pos; p_out=np.empty(total,dtype=np.int64); n_out=np.empty(total,dtype=np.int64); k=0
    for p,pool in zip(pos_idx,neg_pools):
        m=len(pool)
        for _ in range(neg_per_pos): p_out[k]=p; n_out[k]=pool[rng.integers(0,m)]; k+=1
    order=rng.permutation(total); return p_out[order],n_out[order]

def fit_bpr(splits,enc,dim,k=16,lr=0.001,l2=1e-6,epochs=40,bs=8192,patience=4,neg_per_pos=3,seed=0,device='cpu',verbose=True):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); pos_idx,neg_pools=make_user_pair_sources(ytr,utr)
    rng=np.random.default_rng(seed); best=-1.0; best_state=None; bad=0
    for ep in range(1,epochs+1):
        t0=time.time(); p_idx,n_idx=sample_uniform_pairs(pos_idx,neg_pools,rng,neg_per_pos); model.train(); losses=[]
        for i in range(0,len(p_idx),bs):
            ps=torch.from_numpy(p_idx[i:i+bs]); ns=torch.from_numpy(n_idx[i:i+bs])
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.softplus(-(model(Xtr_t[ps].to(device))-model(Xtr_t[ns].to(device)))).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  BPR(seed={seed}) epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def user_groups(users):
    d=defaultdict(list)
    for i,u in enumerate(users): d[u].append(i)
    return [np.asarray(v,dtype=np.int64) for v in d.values()]

def per_user_zscore(scores,groups):
    scores=scores.astype(np.float64,copy=False); out=np.empty_like(scores,dtype=np.float64)
    for idx in groups:
        s=scores[idx]; sd=s.std(); out[idx]=(s-s.mean())/sd if sd>1e-12 else s-s.mean()
    return out

def per_user_rank01(scores,groups):
    scores=scores.astype(np.float64,copy=False); out=np.empty_like(scores,dtype=np.float64)
    for idx in groups:
        n=len(idx)
        if n<=1: out[idx]=0.0; continue
        order=np.argsort(scores[idx],kind='mergesort'); ranks=np.empty(n,dtype=np.float64)
        ranks[order]=np.arange(n,dtype=np.float64)/(n-1.0); out[idx]=ranks
    return out

def node8_seed_score(bce_preds,bpr_preds,groups):
    z=0.35*per_user_zscore(bce_preds,groups)+0.65*per_user_zscore(bpr_preds,groups)
    r=0.35*per_user_rank01(bce_preds,groups)+0.65*per_user_rank01(bpr_preds,groups)
    return 0.70*z+0.30*r

def cached_predict(prefix,member_name,train_fn,enc,target,member_seed,device,use_cache=True):
    os.makedirs('pred_cache',exist_ok=True); X,_,_=enc[target]
    path=os.path.join('pred_cache',f'{prefix}_{member_name}_{target}_seed{member_seed}.npy')
    if use_cache and os.path.isfile(path):
        p=np.load(path)
        if len(p)==len(X): return p.astype(np.float64,copy=False)
    model=train_fn(); p=model.predict(X,device=device).astype(np.float64)
    if use_cache: np.save(path,p)
    return p

def _norm(x):
    s='' if x is None else str(x).strip(); return s[:-2] if s.endswith('.0') else s

def _first(rec,names):
    for n in names:
        if n in rec and rec[n]!='': return rec[n]
    return None

def _hour_bucket(hourmin):
    if hourmin is None: return 'hUNK'
    s=str(hourmin).strip()
    if not s: return 'hUNK'
    try:
        if ':' in s: h=int(s.split(':',1)[0])
        else:
            v=int(float(s)); h=v//100 if v>=100 else v
        if 0<=h<=23: return 'h%02d'%(h//4)
    except Exception: pass
    return 'hUNK'

def build_hour_maps(data_dir):
    full=defaultdict(list); short=defaultdict(list)
    for fn in ['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']:
        path=os.path.join(data_dir,fn)
        if not os.path.isfile(path): continue
        with open(path,'r',encoding='utf-8',newline='') as f:
            for rec in csv.DictReader(f):
                date=_norm(_first(rec,['date'])); uid=_norm(_first(rec,['user_id','userId','user']))
                vid=_norm(_first(rec,['video_id','videoId','item_id','itemId'])); aid=_norm(_first(rec,['author_id','authorId']))
                tab=_norm(_first(rec,['tab'])); dur=_norm(_first(rec,['duration_ms','duration','video_duration']))
                hb=_hour_bucket(_first(rec,['hourmin','hour_min','time','timestamp']))
                if date and uid and vid and tab and dur:
                    short[(date,uid,vid,tab,dur)].append(hb)
                    if aid: full[(date,uid,vid,aid,tab,dur)].append(hb)
    return full,short

def _dow_feat(d):
    ds=_norm(d)
    try: return 'dow%d'%_dt.datetime.strptime(ds,'%Y%m%d').date().weekday()
    except Exception: return 'dowUNK'

def augment_with_time(splits,enc_base,dim_base,data_dir):
    full,short=build_hour_maps(data_dir); full_pos=defaultdict(int); short_pos=defaultdict(int); extras={}; hit=total=0
    for name,rows in splits.items():
        cur=[]
        for r in rows:
            date,uid,vid,aid,tab,dur=r[0],r[1],r[2],r[3],r[4],r[5]
            fkey=(_norm(date),_norm(uid),_norm(vid),_norm(aid),_norm(tab),_norm(dur)); skey=(_norm(date),_norm(uid),_norm(vid),_norm(tab),_norm(dur))
            hb=None; p=full_pos[fkey]
            if p<len(full.get(fkey,[])): hb=full[fkey][p]; full_pos[fkey]+=1
            else:
                p=short_pos[skey]
                if p<len(short.get(skey,[])): hb=short[skey][p]; short_pos[skey]+=1
            if hb is None: hb='hUNK'
            else: hit+=1
            total+=1; cur.append((_dow_feat(date),hb))
        extras[name]=cur
    print(f'time feature hour lookup hits: {hit}/{total}')
    maps=[]
    for j in range(2):
        mp={}
        for name in splits.keys():
            for tup in extras[name]:
                if tup[j] not in mp: mp[tup[j]]=len(mp)
        maps.append(mp)
    offsets=[]; off=dim_base
    for mp in maps: offsets.append(off); off+=len(mp)
    enc={}
    for name,(X,y,u) in enc_base.items():
        extra=np.empty((len(X),2),dtype=np.int64)
        for i,tup in enumerate(extras[name]):
            for j in range(2): extra[i,j]=offsets[j]+maps[j][tup[j]]
        enc[name]=(np.hstack([X.astype(np.int64),extra]),y,u)
    print(f"augmented fields={FIELDS+['dow','hour4']} dim {dim_base}->{off}")
    return enc,off

def make_bag(splits,enc,dim,target,groups,a,prefix):
    member_seeds=[0,1,2,3,4]; weights=np.asarray([0.12,0.12,0.12,0.32,0.32],dtype=np.float64); seed_scores=[]
    use_cache=a.out is not None and a.split!='dev'; verbose=a.out is None
    for ms in member_seeds:
        bce=cached_predict(prefix,'bce',lambda ms=ms: fit_bce(splits,enc,dim,k=a.k,lr=a.lr,epochs=a.epochs,seed=ms,device=a.device,verbose=verbose),enc,target,ms,a.device,use_cache)
        bpr=cached_predict(prefix,'bpr_uniform_np3',lambda ms=ms: fit_bpr(splits,enc,dim,k=a.k,lr=a.lr,epochs=a.epochs,neg_per_pos=a.neg_per_pos,seed=ms,device=a.device,verbose=verbose),enc,target,ms,a.device,use_cache)
        seed_scores.append(per_user_zscore(node8_seed_score(bce,bpr,groups),groups))
    mat=np.vstack(seed_scores); bag=weights@mat; cur_idx=member_seeds.index(a.seed) if a.seed in member_seeds else (a.seed%len(member_seeds))
    return 0.995*bag+0.005*seed_scores[cur_idx]

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data')
    ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None)
    ap.add_argument('--k',type=int,default=16); ap.add_argument('--lr',type=float,default=0.001); ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--neg_per_pos',type=int,default=3); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=ap.parse_args(); torch.manual_seed(a.seed)
    print(f'loading {a.data_dir} ...')
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}')
    enc_base,dim_base=encode(splits); X,y,users=enc_base[target]; groups=user_groups(users)
    base=make_bag(splits,enc_base,dim_base,target,groups,a,'006')
    enc_time,dim_time=augment_with_time(splits,enc_base,dim_base,a.data_dir)
    timebag=make_bag(splits,enc_time,dim_time,target,groups,a,'029_time_nodate')
    base_f=0.70*per_user_zscore(base,groups)+0.30*per_user_rank01(base,groups)
    time_f=0.70*per_user_zscore(timebag,groups)+0.30*per_user_rank01(timebag,groups)
    scores=0.50*per_user_zscore(base_f,groups)+0.50*per_user_zscore(time_f,groups)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        r=evaluate(users,y,scores); print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
