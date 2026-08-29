"""Best cached content-FM/DeepFM ensemble plus behavior prior, with a top-heavy RRF blend."""
import argparse, csv, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate

CONTENT_FIELDS = ['music_id', 'tag', 'video_type', 'upload_type', 'server_width', 'server_height', 'music_type']

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32)); self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float32))
    def forward(self,X):
        E=self.V[X]; S=E.sum(1)
        return self.b + self.W[X].sum(1) + 0.5*((S*S).sum(1)-(E*E).sum((1,2)))
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

class DeepFM(torch.nn.Module):
    def __init__(self, dim, n_fields, k=8, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32)); self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float32))
        self.D=torch.nn.Embedding(dim,k); torch.nn.init.normal_(self.D.weight,0.0,0.01)
        self.mlp=torch.nn.Sequential(torch.nn.Linear(n_fields*k,64),torch.nn.ReLU(),torch.nn.Dropout(0.10),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    def forward(self,X):
        E=self.V[X]; S=E.sum(1)
        fm=self.b + self.W[X].sum(1) + 0.5*((S*S).sum(1)-(E*E).sum((1,2)))
        return fm + self.mlp(self.D(X).reshape(X.shape[0],-1)).squeeze(1)
    @torch.no_grad()
    def predict(self,X,bs=65536,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def norm_val(v):
    if v is None or v=='': return '__MISS__'
    s=str(v).strip(); return s if s else '__MISS__'

def load_video_content(data_dir):
    path=os.path.join(data_dir,'video_features_basic_pure.csv'); mp={}
    if not os.path.isfile(path): return mp
    with open(path,newline='',encoding='utf-8') as f:
        rdr=csv.DictReader(f); names=rdr.fieldnames or []
        vid_col='video_id' if 'video_id' in names else ('item_id' if 'item_id' in names else None)
        if vid_col is None: return mp
        for rec in rdr: mp[norm_val(rec.get(vid_col))]=tuple(norm_val(rec.get(c)) for c in CONTENT_FIELDS)
    print('loaded video content rows',len(mp)); return mp

def append_content_features(splits, enc, base_dim, data_dir):
    content=load_video_content(data_dir); allrows=[]
    for sp in ['train','valid','test']:
        if sp in splits: allrows.extend(splits[sp])
    maps=[{'__MISS__':0} for _ in CONTENT_FIELDS]
    for r in allrows:
        vals=content.get(norm_val(r[2]),('__MISS__',)*len(CONTENT_FIELDS))
        for j,v in enumerate(vals):
            if v not in maps[j]: maps[j][v]=len(maps[j])
    offsets=[]; off=base_dim
    for m in maps: offsets.append(off); off+=len(m)
    print('content cardinalities',{CONTENT_FIELDS[i]:len(maps[i]) for i in range(len(maps))},'new_dim',off)
    out={}
    for sp,(X,y,u) in enc.items():
        C=np.empty((len(X),len(CONTENT_FIELDS)),dtype=np.int64)
        for i,r in enumerate(splits[sp]):
            vals=content.get(norm_val(r[2]),('__MISS__',)*len(CONTENT_FIELDS))
            for j,v in enumerate(vals): C[i,j]=offsets[j]+maps[j].get(v,0)
        out[sp]=(np.concatenate([X.astype(np.int64),C],axis=1),y,u)
    return out,off,content

def make_pairs(y,users):
    y=np.asarray(y); users=np.asarray(users); order=np.argsort(users,kind='mergesort'); us=users[order]
    pos_all=[]; gid_all=[]; neg_by=[]; s=0; gid=0
    while s<len(order):
        e=s+1
        while e<len(order) and us[e]==us[s]: e+=1
        idx=order[s:e]; yy=y[idx]; pos=idx[yy>0.5]; neg=idx[yy<=0.5]
        if len(pos)>0 and len(neg)>0:
            pos_all.append(pos.astype(np.int64)); gid_all.append(np.full(len(pos),gid,dtype=np.int32)); neg_by.append(neg.astype(np.int64)); gid+=1
        s=e
    return np.concatenate(pos_all),np.concatenate(gid_all),neg_by

def sample_negs(gids,neg_by,rng):
    out=np.empty(len(gids),dtype=np.int64)
    for g in np.unique(gids):
        m=(gids==g); pool=neg_by[int(g)]; out[m]=pool[rng.integers(0,len(pool),size=int(m.sum()))]
    return out

def train_bce_member(enc,dim,seed=0,k=16,lr=0.001,l2=1e-6,epochs=40,bs=8192,patience=4,device='cpu',verbose=False):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); yt=torch.from_numpy(ytr.astype(np.float32)); rng=np.random.default_rng(seed); best=-1; state=None; bad=0
    for ep in range(1,epochs+1):
        model.train(); perm=rng.permutation(len(Xtr))
        for i in range(0,len(Xtr),bs):
            idx=torch.from_numpy(perm[i:i+bs]); xb=Xt[idx].to(device); yb=yt[idx].to(device)
            opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(model(xb),yb); loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if va['primary']>best+1e-5: best=va['primary']; bad=0; state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(state); return model

def train_bpr_member(enc,dim,seed=0,k=16,lr=0.001,l2=1e-6,epochs=40,bs=8192,patience=5,repeats=2,bce_weight=0.10,device='cpu',verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); pos,gids,neg_by=make_pairs(ytr,utr); rng=np.random.default_rng(seed); best=-1; state=None; bad=0
    for ep in range(1,epochs+1):
        model.train()
        for _ in range(repeats):
            perm=rng.permutation(len(pos))
            for i in range(0,len(perm),bs):
                sel=perm[i:i+bs]; pi=pos[sel]; ni=sample_negs(gids[sel],neg_by,rng)
                xb=torch.cat([Xt[torch.from_numpy(pi)],Xt[torch.from_numpy(ni)]],0).to(device); m=len(pi)
                opt.zero_grad(set_to_none=True); logits=model(xb); loss=F.softplus(-(logits[:m]-logits[m:])).mean()
                if bce_weight>0:
                    lab=torch.cat([torch.ones(m,device=device),torch.zeros(m,device=device)]); loss=loss+bce_weight*F.binary_cross_entropy_with_logits(logits,lab)
                loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if va['primary']>best+1e-5: best=va['primary']; bad=0; state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(state); return model

def train_deep_bpr(enc,dim,seed=0,k=8,lr=0.001,l2=1e-6,epochs=14,bs=8192,patience=3,repeats=1,bce_weight=0.05,device='cpu',verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; model=DeepFM(dim,Xtr.shape[1],k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W,model.D.weight]+list(model.mlp.parameters()),'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); pos,gids,neg_by=make_pairs(ytr,utr); rng=np.random.default_rng(seed+12345); best=-1; state=None; bad=0
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for _ in range(repeats):
            perm=rng.permutation(len(pos))
            for i in range(0,len(perm),bs):
                sel=perm[i:i+bs]; pi=pos[sel]; ni=sample_negs(gids[sel],neg_by,rng)
                xb=torch.cat([Xt[torch.from_numpy(pi)],Xt[torch.from_numpy(ni)]],0).to(device); m=len(pi)
                opt.zero_grad(set_to_none=True); logits=model(xb); loss=F.softplus(-(logits[:m]-logits[m:])).mean()
                if bce_weight>0:
                    lab=torch.cat([torch.ones(m,device=device),torch.zeros(m,device=device)]); loss=loss+bce_weight*F.binary_cross_entropy_with_logits(logits,lab)
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va=evaluate(uva,yva,model.predict(Xva,device=device)); print(f'deep seed {seed} epoch {ep} loss {np.mean(losses):.4f} primary {va["primary"]:.5f}')
        if va['primary']>best+1e-5: best=va['primary']; bad=0; state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(state); return model

def percentile_rank_by_user(scores,users):
    scores=np.asarray(scores,dtype=np.float64); users=np.asarray(users); out=np.empty_like(scores,dtype=np.float64); order=np.argsort(users,kind='mergesort'); us=users[order]; s=0
    while s<len(order):
        e=s+1
        while e<len(order) and us[e]==us[s]: e+=1
        idx=order[s:e]; ord2=idx[np.argsort(scores[idx],kind='mergesort')]; m=len(ord2)
        out[ord2]=0.0 if m<=1 else np.arange(m,dtype=np.float64)/(m-1.0); s=e
    return out

def rrf_by_user(score_list, users, weights=None, c=10.0):
    users=np.asarray(users); n=len(users); out=np.zeros(n,dtype=np.float64)
    if weights is None: weights=[1.0/len(score_list)]*len(score_list)
    order=np.argsort(users,kind='mergesort'); us=users[order]; s=0
    while s<n:
        e=s+1
        while e<n and us[e]==us[s]: e+=1
        idx=order[s:e]
        for sc,w in zip(score_list,weights):
            # descending ranks: rank 1 is best for this member
            ord2=idx[np.argsort(-np.asarray(sc)[idx],kind='mergesort')]
            ranks=np.arange(1,len(ord2)+1,dtype=np.float64)
            out[ord2] += w/(c+ranks)
        s=e
    return percentile_rank_by_user(out,users)

def get_preds(prefix,name,train_fn,enc,dim,Xtar,seed,device,verbose):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'{prefix}_{name}_seed{seed}_n{len(Xtar)}.npy')
    if os.path.isfile(path): print('loading cached',path); return np.load(path).astype(np.float64)
    print('training',name,'seed',seed); model=train_fn(enc,dim,seed=seed,device=device,verbose=verbose); p=model.predict(Xtar,device=device).astype(np.float64); np.save(path,p); return p

def train_video_counts(splits, target):
    cnt={}
    for r in splits['train']:
        v=norm_val(r[2]); cnt[v]=cnt.get(v,0)+1
    return np.array([cnt.get(norm_val(r[2]),0) for r in splits[target]],dtype=np.float64)

def make_behavior_prior(splits, target, content):
    y=np.array([r[6] for r in splits['train']],dtype=np.float64); global_mean=float(y.mean())
    specs=[
        ('uv', lambda r: (norm_val(r[1]), norm_val(r[2])), 5.0, 1.6),
        ('ua', lambda r: (norm_val(r[1]), norm_val(r[3])), 12.0, 1.3),
        ('utag', lambda r: (norm_val(r[1]), content.get(norm_val(r[2]),('__MISS__',)*7)[1]), 15.0, 1.0),
        ('umusic', lambda r: (norm_val(r[1]), content.get(norm_val(r[2]),('__MISS__',)*7)[0]), 12.0, 0.8),
        ('utab', lambda r: (norm_val(r[1]), norm_val(r[4])), 20.0, 0.45),
        ('udur', lambda r: (norm_val(r[1]), str(int(float(r[5])//10000))), 25.0, 0.35),
    ]
    maps=[]
    for name,fn,alpha,wt in specs:
        d={}
        for r in splits['train']:
            k=fn(r); n,s=d.get(k,(0,0.0)); d[k]=(n+1,s+float(r[6]))
        maps.append((name,fn,alpha,wt,d)); print('behavior map',name,'keys',len(d))
    raw=np.empty(len(splits[target]),dtype=np.float64); strength=np.empty(len(splits[target]),dtype=np.float64)
    for i,r in enumerate(splits[target]):
        num=0.0; den=0.0; st=0.0
        for name,fn,alpha,wt,d in maps:
            n,s=d.get(fn(r),(0,0.0)); rate=(s+alpha*global_mean)/(n+alpha); shrink=n/(n+alpha)
            num += wt*shrink*rate; den += wt*shrink; st += wt*shrink
        raw[i]=global_mean if den<=0 else num/den; strength[i]=st
    print('behavior prior mean/std/coverage/strength',float(raw.mean()),float(raw.std()),float((strength>0).mean()),float(strength.mean()))
    return raw

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}+{CONTENT_FIELDS}')
    enc0,dim0=encode(splits); enc,dim,content=append_content_features(splits,enc0,dim0,a.data_dir); Xtar,_,utar=enc[target]; verbose=(a.out is None)
    bags=[]
    for s in (0,1,2):
        bpr=get_preds('018','content_bpr_anchor_v1',train_bpr_member,enc,dim,Xtar,s,a.device,verbose)
        bce=get_preds('018','content_bce_v1',train_bce_member,enc,dim,Xtar,s,a.device,verbose)
        bags.append(0.70*percentile_rank_by_user(bpr,utar)+0.30*percentile_rank_by_user(bce,utar))
    fm_base=np.mean(bags,axis=0)
    deep_bags=[]
    for s in (0,1,2):
        dp=get_preds('024','deepfm_content_bpr_v1',train_deep_bpr,enc,dim,Xtar,s,a.device,verbose)
        deep_bags.append(percentile_rank_by_user(dp,utar))
    deep=np.mean(deep_bags,axis=0)
    vc=train_video_counts(splits,target); w=0.20 + 0.25/(1.0 + vc/5.0)
    base=(1.0-w)*fm_base + w*deep
    prior=percentile_rank_by_user(make_behavior_prior(splits,target,content),utar)
    current=0.88*base + 0.12*prior
    # RRF is top-heavy inside each user; blend at 30% so any nDCG@5 signal is readable.
    rrf=rrf_by_user([fm_base,deep,prior],utar,weights=[0.58,0.30,0.12],c=10.0)
    scores=0.70*current + 0.30*rrf
    print('node24 deep weight mean/min/max',float(w.mean()),float(w.min()),float(w.max()),'behavior blend 0.12 plus 30% RRF(c=10)')
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,} predictions')
    else: print('done')
