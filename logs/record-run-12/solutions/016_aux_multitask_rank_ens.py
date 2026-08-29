"""Add an auxiliary-feedback multi-task BPR member to the best rank ensemble.

The incumbent is node 11: seed-bagged per-user rank fusion of BPR-FM and BCE-FM.
This script adds a new member trained with the primary BPR objective plus BCE
auxiliary heads for raw feedback signals (click/like/follow/comment/forward),
then fuses it at a readable 30% rank weight.  All unchanged incumbent member
cache names are reused from 007; the new auxiliary member uses its own cache.
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

AUX_SPECS = [
    ('click', ['is_click', 'click']),
    ('like', ['is_like', 'like']),
    ('follow', ['is_follow', 'follow']),
    ('comment', ['is_comment', 'comment']),
    ('forward', ['is_forward', 'forward', 'share']),
]

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float32))
    def base_inter(self,X):
        E=self.V[X]; S=E.sum(1); return 0.5*((S*S).sum(1)-(E*E).sum((1,2)))
    def forward(self,X):
        return self.b+self.W[X].sum(1)+self.base_inter(X)
    @torch.no_grad()
    def predict(self,X,bs=200_000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_aux, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(n_aux+1, dim, dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros(n_aux+1, dtype=torch.float32))
    def inter(self,X):
        E=self.V[X]; S=E.sum(1); return 0.5*((S*S).sum(1)-(E*E).sum((1,2)))
    def forward_task(self,X,t=0):
        return self.b[t]+self.W[t,X].sum(1)+self.inter(X)
    def forward_all(self,X):
        inter=self.inter(X).unsqueeze(1)
        lin=self.W[:,X].sum(2).transpose(0,1)
        return self.b.unsqueeze(0)+lin+inter
    @torch.no_grad()
    def predict(self,X,bs=200_000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs):
            out.append(self.forward_task(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device),0).cpu().numpy())
        return np.concatenate(out)

def train_bce_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']
    model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr,betas=(0.9,0.999),eps=1e-8)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); yt=torch.from_numpy(ytr.astype(np.float32)); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0; n=len(Xtr)
    for ep in range(1,epochs+1):
        t0=time.time(); model.train(); losses=[]; perm=rng.permutation(n)
        for i in range(0,n,bs):
            idx=torch.from_numpy(perm[i:i+bs]); xb=Xt[idx].to(device); yb=yt[idx].to(device)
            opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(model(xb),yb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  bce seed {seed} ep {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def make_positive_index_pairs(y,users):
    y=np.asarray(y); users=np.asarray(users); order=np.argsort(users,kind='mergesort'); us=users[order]
    pos_indices=[]; pos_gids=[]; neg_by_gid=[]; start=0; gid=0; n=len(order)
    while start<n:
        end=start+1
        while end<n and us[end]==us[start]: end+=1
        idx=order[start:end]; yy=y[idx]; pos=idx[yy>0.5]; neg=idx[yy<=0.5]
        if len(pos)>0 and len(neg)>0:
            pos_indices.append(pos.astype(np.int64)); pos_gids.append(np.full(len(pos),gid,dtype=np.int32)); neg_by_gid.append(neg.astype(np.int64)); gid+=1
        start=end
    if not pos_indices: raise RuntimeError('No users with both positive and negative impressions')
    return np.concatenate(pos_indices),np.concatenate(pos_gids),neg_by_gid

def sample_negatives_for_batch(gids,neg_by_gid,rng):
    neg=np.empty(len(gids),dtype=np.int64)
    for g in np.unique(gids):
        m=(gids==g); pool=neg_by_gid[int(g)]; neg[m]=pool[rng.integers(0,len(pool),size=int(m.sum()))]
    return neg

def train_bpr_member(enc, dim, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5, repeats=2, bce_weight=0.10, device='cpu', verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr,betas=(0.9,0.999),eps=1e-8)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); pos_base,pos_gid_base,neg_by_gid=make_positive_index_pairs(ytr,utr); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0
    for ep in range(1,epochs+1):
        t0=time.time(); model.train(); losses=[]
        for _ in range(repeats):
            perm=rng.permutation(len(pos_base))
            for i in range(0,len(perm),bs):
                psel=perm[i:i+bs]; pos_idx=pos_base[psel]; neg_idx=sample_negatives_for_batch(pos_gid_base[psel],neg_by_gid,rng)
                xb=torch.cat([Xt[torch.from_numpy(pos_idx)].to(device),Xt[torch.from_numpy(neg_idx)].to(device)],0)
                opt.zero_grad(set_to_none=True); logits=model(xb); m=len(pos_idx)
                loss=F.softplus(-(logits[:m]-logits[m:])).mean()
                if bce_weight>0:
                    labels=torch.cat([torch.ones(m,device=device),torch.zeros(m,device=device)])
                    loss=loss+bce_weight*F.binary_cross_entropy_with_logits(logits,labels)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  bpr seed {seed} ep {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def getv(rec,names,default='0'):
    for n in names:
        if n in rec and rec[n] != '': return rec[n]
    return default

def to_int(x,default=0):
    try: return int(float(str(x).strip()))
    except Exception: return default

def load_aux_for_train(data_dir, train_rows):
    raw=[]; found=None
    for fn in ['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']:
        path=os.path.join(data_dir,fn)
        if not os.path.isfile(path): continue
        with open(path,newline='',encoding='utf-8') as f:
            rdr=csv.DictReader(f)
            if found is None: found=[any(c in (rdr.fieldnames or []) for c in names) for _,names in AUX_SPECS]
            for rec in rdr:
                date=to_int(getv(rec,['date','request_date','day']))
                vals=[float(to_int(getv(rec,names,'0'))) for _,names in AUX_SPECS]
                raw.append((date,vals))
    n=len(train_rows)
    if not raw:
        print('WARNING no raw logs for aux; disabling auxiliary tasks')
        return np.zeros((n,0),dtype=np.float32), []
    dates=set(to_int(r[0]) for r in train_rows)
    vals=np.array([v for d,v in raw if d in dates],dtype=np.float32)
    if len(vals)!=n:
        print(f'WARNING aux row count {len(vals)} != train {n}; disabling aux')
        return np.zeros((n,0),dtype=np.float32), []
    names=[]; keep=[]
    for j,(nm,_) in enumerate(AUX_SPECS):
        if vals[:,j].max()>0 and vals[:,j].mean()<0.98:
            keep.append(j); names.append(nm)
    vals=vals[:,keep] if keep else np.zeros((n,0),dtype=np.float32)
    if vals.shape[1]: print('aux tasks', names, 'rates', np.round(vals.mean(0),4).tolist())
    else: print('no nonconstant aux tasks found')
    return vals.astype(np.float32), names

def train_aux_bpr_member(enc, dim, data_dir, splits, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=5, repeats=2, bce_weight=0.10, aux_weight=0.06, device='cpu', verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    aux_np, aux_names = load_aux_for_train(data_dir, splits['train'])
    naux=aux_np.shape[1]
    model=MultiTaskFM(dim,naux,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr,betas=(0.9,0.999),eps=1e-8)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); aux_t=torch.from_numpy(aux_np.astype(np.float32))
    if naux:
        rates=np.clip(aux_np.mean(0),1e-4,1-1e-4); posw=torch.from_numpy(np.minimum((1-rates)/rates,5.0).astype(np.float32)).to(device)
    pos_base,pos_gid_base,neg_by_gid=make_positive_index_pairs(ytr,utr); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0
    for ep in range(1,epochs+1):
        t0=time.time(); model.train(); losses=[]
        for _ in range(repeats):
            perm=rng.permutation(len(pos_base))
            for i in range(0,len(perm),bs):
                psel=perm[i:i+bs]; pos_idx=pos_base[psel]; neg_idx=sample_negatives_for_batch(pos_gid_base[psel],neg_by_gid,rng)
                both=np.concatenate([pos_idx,neg_idx]); xb=Xt[torch.from_numpy(both)].to(device)
                opt.zero_grad(set_to_none=True); logits0=model.forward_task(xb,0); m=len(pos_idx)
                loss=F.softplus(-(logits0[:m]-logits0[m:])).mean()
                labels0=torch.cat([torch.ones(m,device=device),torch.zeros(m,device=device)])
                loss=loss+bce_weight*F.binary_cross_entropy_with_logits(logits0,labels0)
                if naux:
                    aux_logits=model.forward_all(xb)[:,1:]
                    yaux=aux_t[torch.from_numpy(both)].to(device)
                    aux_loss=F.binary_cross_entropy_with_logits(aux_logits,yaux,pos_weight=posw)
                    loss=loss+aux_weight*aux_loss
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  auxbpr seed {seed} ep {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def percentile_rank_by_user(scores,users):
    scores=np.asarray(scores,dtype=np.float64); users=np.asarray(users); out=np.empty_like(scores,dtype=np.float64)
    order=np.argsort(users,kind='mergesort'); us=users[order]; start=0; n=len(order)
    while start<n:
        end=start+1
        while end<n and us[end]==us[start]: end+=1
        idx=order[start:end]; ord2=idx[np.argsort(scores[idx],kind='mergesort')]; m=len(ord2)
        out[ord2]=0.0 if m<=1 else np.arange(m,dtype=np.float64)/(m-1.0)
        start=end
    return out

def get_member_preds(name, train_fn, enc, dim, Xtar, seed, device, verbose, extra_args=()):
    os.makedirs('pred_cache',exist_ok=True)
    cache_path=os.path.join('pred_cache',f'{name}_seed{seed}_n{len(Xtar)}.npy')
    if os.path.isfile(cache_path):
        print(f'loading cached {name} seed {seed} predictions {cache_path}'); return np.load(cache_path).astype(np.float64)
    print(f'training {name} seed {seed} member')
    model=train_fn(enc,dim,*extra_args,seed=seed,device=device,verbose=verbose)
    preds=model.predict(Xtar,device=device).astype(np.float64); np.save(cache_path,preds); return preds

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(0); print(f"loading {a.data_dir} ...")
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k_:len(v) for k_,v in splits.items()}, f"fields={FIELDS}")
    enc,dim=encode(splits); Xtar,_,utar=enc[target]; verbose=(a.out is None)
    blended=[]
    for member_seed in (0,1,2):
        bpr=get_member_preds('007_bpr_anchor_v1',train_bpr_member,enc,dim,Xtar,member_seed,a.device,verbose)
        bce=get_member_preds('007_bce_v1',train_bce_member,enc,dim,Xtar,member_seed,a.device,verbose)
        aux=get_member_preds('016_aux_bpr_v1',train_aux_bpr_member,enc,dim,Xtar,member_seed,a.device,verbose,extra_args=(a.data_dir,splits))
        inc=0.70*percentile_rank_by_user(bpr,utar)+0.30*percentile_rank_by_user(bce,utar)
        blended.append(0.70*inc+0.30*percentile_rank_by_user(aux,utar))
    scores=np.mean(blended,axis=0)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print('done')
