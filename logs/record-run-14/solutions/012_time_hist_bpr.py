import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402
from evaluate import evaluate  # noqa: E402


def _s(x):
    if x is None:
        return ''
    if isinstance(x, bytes):
        x = x.decode('utf8')
    try:
        if isinstance(x, (float, np.floating)) and np.isfinite(x) and abs(x - int(x)) < 1e-6:
            return str(int(x))
    except Exception:
        pass
    return str(x)


def date_to_dow(d):
    try:
        return datetime.strptime(_s(d), '%Y%m%d').weekday()
    except Exception:
        return 7


def cb(c):
    c = int(c)
    if c <= 0: return '0'
    if c == 1: return '1'
    if c == 2: return '2'
    if c <= 4: return '3-4'
    if c <= 8: return '5-8'
    return '9+'


def rb(p, n):
    # smoothed rate bin; returns 0..5.  Only train labels accumulated so this is causal.
    t = p + n
    r = (p + 1.0) / (t + 2.0)
    b = int(r * 6.0)
    return str(min(5, max(0, b)))


def hist_values(rows, frozen=None, update=True):
    if frozen is None:
        uvp=defaultdict(int); uvn=defaultdict(int); uap=defaultdict(int); uan=defaultdict(int)
        utp=defaultdict(int); utn=defaultdict(int); vp=defaultdict(int); vn=defaultdict(int)
        ap=defaultdict(int); an=defaultdict(int)
    else:
        uvp,uvn,uap,uan,utp,utn,vp,vn,ap,an = frozen
    vals = []
    for r in rows:
        u,v,a,t,y = _s(r[1]), _s(r[2]), _s(r[3]), _s(r[4]), float(r[6])
        kuv=(u,v); kua=(u,a); kut=(u,t)
        vals.append((
            'uvp='+cb(uvp[kuv]), 'uvn='+cb(uvn[kuv]),
            'uap='+cb(uap[kua]), 'uan='+cb(uan[kua]), 'uar='+rb(uap[kua], uan[kua]),
            'utp='+cb(utp[kut]), 'utn='+cb(utn[kut]), 'utr='+rb(utp[kut], utn[kut]),
            'vp='+cb(vp[v]), 'vn='+cb(vn[v]), 'vr='+rb(vp[v], vn[v]),
            'ap='+cb(ap[a]), 'an='+cb(an[a]), 'ar='+rb(ap[a], an[a]),
        ))
        if update:
            if y > 0.5:
                uvp[kuv]+=1; uap[kua]+=1; utp[kut]+=1; vp[v]+=1; ap[a]+=1
            else:
                uvn[kuv]+=1; uan[kua]+=1; utn[kut]+=1; vn[v]+=1; an[a]+=1
    return vals, (uvp,uvn,uap,uan,utp,utn,vp,vn,ap,an)


def time_default_values(rows):
    vals=[]
    for r in rows:
        date=_s(r[0]); tab=_s(r[4]); dow=date_to_dow(date)
        # Keep the successful node-10 behaviour after debugging: raw hour/order did not help,
        # while these coarse date/tab-date fields did.
        vals.append((
            'd='+date, 'dow='+str(dow), 'h=24', 'hb=6',
            'th='+tab+'_24', 'td='+tab+'_'+date, 'gb=20', 'db='+date+'_10'
        ))
    return vals


def build_augmented(splits):
    enc0, dim0 = encode(splits)
    aux={}
    htrain, frozen = hist_values(splits['train'], None, True)
    aux['train']=[tv+hv for tv,hv in zip(time_default_values(splits['train']), htrain)]
    for sp in splits:
        if sp=='train': continue
        hv,_ = hist_values(splits[sp], frozen, False)
        aux[sp]=[tv+hh for tv,hh in zip(time_default_values(splits[sp]), hv)]
    n_extra=len(aux['train'][0]) if len(aux['train']) else 0
    maps=[]; off=dim0
    for j in range(n_extra):
        mp={}
        for sp in splits:
            for v in aux[sp]:
                key=v[j]
                if key not in mp:
                    mp[key]=off+len(mp)
        maps.append(mp); off+=len(mp)
    enc={}
    for sp in splits:
        X0,y,u=enc0[sp]
        H=np.empty((len(X0), n_extra), dtype=np.int64)
        for i,v in enumerate(aux[sp]):
            for j,key in enumerate(v):
                H[i,j]=maps[j][key]
        enc[sp]=(np.concatenate([X0.astype(np.int64), H], axis=1), y, u)
    print(f'time_default_hist extra_fields={n_extra} dim={off}', flush=True)
    return enc, off


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float32))
    def forward(self,X):
        E=self.V[X]; s=E.sum(1)
        inter=0.5*((s*s).sum(1)-(E*E).sum((1,2)))
        return self.b+self.W[X].sum(1)+inter
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs):
            xb=torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)
            out.append(self(xb).detach().cpu().numpy())
        return np.concatenate(out)


def build_pairs(y, users):
    pos=defaultdict(list); neg=defaultdict(list)
    for i,(u,yy) in enumerate(zip(users,y)):
        (pos if yy>0.5 else neg)[u].append(i)
    pidx=[]; pools=[]
    for u,ps in pos.items():
        ns=neg.get(u)
        if not ns: continue
        arr=np.asarray(ns,dtype=np.int64)
        for p in ps:
            pidx.append(p); pools.append(arr)
    return np.asarray(pidx,dtype=np.int64), pools


def train_bpr(enc, dim, seed, k=16, lr=0.001, l2=3e-6, epochs=24, bs=8192, patience=4, device='cpu'):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32))
    pos_idx, neg_pools=build_pairs(ytr,utr)
    rng=np.random.default_rng(seed+17); bce=torch.nn.BCEWithLogitsLoss()
    n_steps=max(len(ytr),len(pos_idx)); best=-1; best_state=None; bad=0
    for ep in range(1,epochs+1):
        model.train(); order=rng.integers(0,len(pos_idx),size=n_steps,dtype=np.int64)
        for st in range(0,len(order),bs):
            which=order[st:st+bs]; pp=pos_idx[which]; nn=np.empty(len(which),dtype=np.int64)
            for j,w in enumerate(which):
                pool=neg_pools[int(w)]; nn[j]=pool[rng.integers(0,len(pool))]
            both=np.concatenate([pp,nn]); xb=Xtr_t[torch.from_numpy(both)].to(device)
            logits=model(xb); lp=logits[:len(pp)]; ln=logits[len(pp):]
            yb=ytr_t[torch.from_numpy(both)].to(device)
            loss=torch.nn.functional.softplus(-(lp-ln)).mean()+0.02*bce(logits,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pred=model.predict(Xva,device=device)
        m=evaluate(uva,yva,pred)['primary']
        print(f'bpr epoch={ep} valid_primary={m:.6f}', flush=True)
        if m>best+1e-5:
            best=m; bad=0; best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); model.to(device); return model


def z_by_user(scores, users):
    scores=np.asarray(scores,dtype=np.float64); out=np.zeros_like(scores); groups=defaultdict(list)
    for i,u in enumerate(users): groups[u].append(i)
    for idx in groups.values():
        v=scores[idx]; sd=v.std(); out[idx]=(v-v.mean())/sd if sd>1e-8 else (v-v.mean())
    return out


def member_pred(enc, dim, target, seed, split_name, device):
    os.makedirs('pred_cache',exist_ok=True)
    path=os.path.join('pred_cache',f'012_time_default_hist_bpr_{split_name}_seed{seed}.npy')
    if os.path.isfile(path): return np.load(path)
    model=train_bpr(enc,dim,seed,device=device)
    X,_,_=enc[target]; p=model.predict(X,device=device).astype(np.float64)
    np.save(path,p); return p


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',required=True); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev'])
    ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; split_name='dev'
    else:
        splits=load(a.data_dir); target=a.split; split_name=a.split
    print({k:len(v) for k,v in splits.items()}, 'base_fields=', FIELDS, flush=True)
    t0=time.time(); enc,dim=build_augmented(splits); _,_,users=enc[target]
    p=member_pred(enc,dim,target,a.seed,split_name,a.device)
    scores=z_by_user(p,users)
    print(f'done in {time.time()-t0:.1f}s', flush=True)
    np.save(a.out,scores.astype(np.float64))
