"""Fast debug of auxiliary multi-task member blended with the incumbent rank ensemble.

Node 15 timed out because it trained three expensive auxiliary BPR members inside
each harness seed.  This keeps the mechanism readable but cheap: load the
cached incumbent BPR/BCE seed-bag (or train a single-seed fallback if caches are
empty), train one pointwise multi-task FM member for the harness seed, and blend
it at 30% by per-user percentile rank.
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

AUX_SPECS=[('click',['is_click','click']),('like',['is_like','like']),('follow',['is_follow','follow']),('comment',['is_comment','comment']),('forward',['is_forward','forward','share'])]

class TorchFM(torch.nn.Module):
    def __init__(self,dim,k=16,seed=0,n_tasks=1):
        super().__init__(); rng=np.random.default_rng(seed); self.n_tasks=n_tasks
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(n_tasks,dim,dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros(n_tasks,dtype=torch.float32))
    def inter(self,X):
        E=self.V[X]; S=E.sum(1); return 0.5*((S*S).sum(1)-(E*E).sum((1,2)))
    def forward_all(self,X):
        lin=self.W[:,X].sum(2).transpose(0,1)
        return self.b.unsqueeze(0)+lin+self.inter(X).unsqueeze(1)
    def forward(self,X): return self.forward_all(X)[:,0]
    @torch.no_grad()
    def predict(self,X,bs=200_000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def getv(rec,names,default='0'):
    for n in names:
        if n in rec and rec[n] != '': return rec[n]
    return default

def to_int(x,default=0):
    try: return int(float(str(x).strip()))
    except Exception: return default

def load_aux_for_train(data_dir,train_rows):
    raw=[]
    for fn in ['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']:
        path=os.path.join(data_dir,fn)
        if not os.path.isfile(path): continue
        with open(path,newline='',encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                raw.append((to_int(getv(rec,['date','request_date','day'])),[float(to_int(getv(rec,names,'0'))) for _,names in AUX_SPECS]))
    n=len(train_rows)
    if not raw:
        print('WARNING no raw aux logs'); return np.zeros((n,0),dtype=np.float32)
    dates=set(to_int(r[0]) for r in train_rows)
    vals=np.array([v for d,v in raw if d in dates],dtype=np.float32)
    if len(vals)!=n:
        print(f'WARNING aux row count {len(vals)} != train {n}; disabling aux'); return np.zeros((n,0),dtype=np.float32)
    keep=[j for j in range(vals.shape[1]) if vals[:,j].max()>0 and vals[:,j].mean()<0.98]
    vals=vals[:,keep] if keep else np.zeros((n,0),dtype=np.float32)
    print('aux dim',vals.shape[1],'rates',np.round(vals.mean(0),4).tolist() if vals.shape[1] else [])
    return vals.astype(np.float32)

def train_aux_bce(enc,dim,data_dir,splits,seed=0,k=16,lr=0.001,l2=1e-6,epochs=18,bs=8192,patience=3,aux_weight=0.15,device='cpu',verbose=False):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']
    aux_np=load_aux_for_train(data_dir,splits['train']); naux=aux_np.shape[1]
    model=TorchFM(dim,k,seed,n_tasks=1+naux).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr,betas=(0.9,0.999),eps=1e-8)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); yt=torch.from_numpy(ytr.astype(np.float32)); aux_t=torch.from_numpy(aux_np.astype(np.float32))
    if naux:
        rates=np.clip(aux_np.mean(0),1e-4,1-1e-4); posw=torch.from_numpy(np.minimum((1-rates)/rates,8.0).astype(np.float32)).to(device)
    rng=np.random.default_rng(seed); n=len(Xtr); best=-1.; best_state=None; bad=0
    for ep in range(1,epochs+1):
        t0=time.time(); model.train(); losses=[]; perm=rng.permutation(n)
        for i in range(0,n,bs):
            idx_np=perm[i:i+bs]; idx=torch.from_numpy(idx_np); xb=Xt[idx].to(device); yb=yt[idx].to(device)
            opt.zero_grad(set_to_none=True); logits=model.forward_all(xb)
            loss=F.binary_cross_entropy_with_logits(logits[:,0],yb)
            if naux:
                yaux=aux_t[idx].to(device)
                loss=loss+aux_weight*F.binary_cross_entropy_with_logits(logits[:,1:],yaux,pos_weight=posw)
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f" aux ep {ep:2d} loss {np.mean(losses):.4f} primary {va['primary']:.4f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

# fallback incumbent training: compact copies of node-11 members
def train_bce_member(enc,dim,seed=0,device='cpu',verbose=False):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,16,seed,1).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=0.001)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); yt=torch.from_numpy(ytr.astype(np.float32)); rng=np.random.default_rng(seed); best=-1.; best_state=None; bad=0
    for ep in range(1,40):
        perm=rng.permutation(len(Xtr)); model.train()
        for i in range(0,len(Xtr),8192):
            idx=torch.from_numpy(perm[i:i+8192]); xb=Xt[idx].to(device); yb=yt[idx].to(device); opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(model(xb),yb); loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if va['primary']>best+1e-5: best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=4: break
    model.load_state_dict(best_state); return model

def make_pairs(y,users):
    order=np.argsort(users,kind='mergesort'); us=np.asarray(users)[order]; y=np.asarray(y); pos=[]; gids=[]; negs=[]; s=0; gid=0
    while s<len(order):
        e=s+1
        while e<len(order) and us[e]==us[s]: e+=1
        idx=order[s:e]; p=idx[y[idx]>0.5]; n=idx[y[idx]<=0.5]
        if len(p) and len(n): pos.append(p.astype(np.int64)); gids.append(np.full(len(p),gid,dtype=np.int32)); negs.append(n.astype(np.int64)); gid+=1
        s=e
    return np.concatenate(pos),np.concatenate(gids),negs

def samp(gids,negs,rng):
    out=np.empty(len(gids),dtype=np.int64)
    for g in np.unique(gids):
        m=gids==g; pool=negs[int(g)]; out[m]=pool[rng.integers(0,len(pool),size=int(m.sum()))]
    return out

def train_bpr_member(enc,dim,seed=0,device='cpu',verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,16,seed,1).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=0.001)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); pos,gids,negs=make_pairs(ytr,utr); rng=np.random.default_rng(seed); best=-1.; best_state=None; bad=0
    for ep in range(1,40):
        for _ in range(2):
            perm=rng.permutation(len(pos)); model.train()
            for i in range(0,len(perm),8192):
                sel=perm[i:i+8192]; pi=pos[sel]; ni=samp(gids[sel],negs,rng); xb=torch.cat([Xt[torch.from_numpy(pi)].to(device),Xt[torch.from_numpy(ni)].to(device)],0)
                opt.zero_grad(set_to_none=True); logits=model(xb); m=len(pi); loss=F.softplus(-(logits[:m]-logits[m:])).mean()+0.10*F.binary_cross_entropy_with_logits(logits,torch.cat([torch.ones(m,device=device),torch.zeros(m,device=device)])); loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if va['primary']>best+1e-5: best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=5: break
    model.load_state_dict(best_state); return model

def percentile_rank_by_user(scores,users):
    scores=np.asarray(scores,dtype=np.float64); users=np.asarray(users); out=np.empty_like(scores,dtype=np.float64); order=np.argsort(users,kind='mergesort'); us=users[order]; s=0
    while s<len(order):
        e=s+1
        while e<len(order) and us[e]==us[s]: e+=1
        idx=order[s:e]; ord2=idx[np.argsort(scores[idx],kind='mergesort')]; m=len(ord2); out[ord2]=0.0 if m<=1 else np.arange(m,dtype=np.float64)/(m-1.0); s=e
    return out

def cached_pred(cache_name,train_fn,enc,dim,Xtar,seed,device,verbose,extra=()):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'{cache_name}_seed{seed}_n{len(Xtar)}.npy')
    if os.path.isfile(path): print('loading cached',path); return np.load(path).astype(np.float64)
    print('training',cache_name,'seed',seed); model=train_fn(enc,dim,*extra,seed=seed,device=device,verbose=verbose); pred=model.predict(Xtar,device=device).astype(np.float64); np.save(path,pred); return pred

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed); print(f'loading {a.data_dir} ...')
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}')
    enc,dim=encode(splits); Xtar,_,utar=enc[target]; verbose=(a.out is None)
    incs=[]
    for s in (0,1,2):
        bpr=cached_pred('007_bpr_anchor_v1',train_bpr_member,enc,dim,Xtar,s,a.device,verbose)
        bce=cached_pred('007_bce_v1',train_bce_member,enc,dim,Xtar,s,a.device,verbose)
        incs.append(0.70*percentile_rank_by_user(bpr,utar)+0.30*percentile_rank_by_user(bce,utar))
    incumbent=np.mean(incs,axis=0)
    aux=cached_pred('017_aux_bce_v1',train_aux_bce,enc,dim,Xtar,a.seed,a.device,verbose,extra=(a.data_dir,splits))
    scores=0.70*incumbent+0.30*percentile_rank_by_user(aux,utar)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else: print('done')
