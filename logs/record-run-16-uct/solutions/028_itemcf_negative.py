import argparse, os, sys, time
import numpy as np
import torch
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype('float32')))
        self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
    def forward(self,X):
        E=self.V[X]; S=E.sum(1); inter=0.5*((S*S).sum(1)-(E*E).sum((1,2)))
        return self.b+self.W[X].sum(1)+inter
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype('int64')).to(device)).cpu().numpy())
        return np.concatenate(out)

def make_pairs(y,u):
    pos={}; neg={}
    for i,(yy,uu) in enumerate(zip(y,u)):
        (pos if yy>0.5 else neg).setdefault(uu,[]).append(i)
    ps=[]; pools=[]
    for uu,p in pos.items():
        n=neg.get(uu)
        if n:
            a=np.asarray(n,dtype=np.int64)
            for x in p: ps.append(x); pools.append(a)
    return np.asarray(ps,dtype=np.int64), pools

def sample(pools,rng,n):
    if n==1:
        z=np.empty(len(pools),dtype=np.int64)
        for i,p in enumerate(pools): z[i]=p[rng.integers(len(p))]
        return z
    z=np.empty((len(pools),n),dtype=np.int64)
    for i,p in enumerate(pools): z[i]=p[rng.integers(len(p),size=n)]
    return z

def train_member(enc,dim,target,seed,soft=False,device='cpu'):
    X,y,u=enc['train']; Xv,yv,uv=enc['valid']; ps,pools=make_pairs(y,u)
    m=TorchFM(dim,16,seed).to(device)
    opt=torch.optim.Adam([{'params':[m.V,m.W],'weight_decay':1e-6},{'params':[m.b],'weight_decay':0}],lr=0.001)
    Xt=torch.from_numpy(X.astype('int64')); rng=np.random.default_rng(seed)
    best=-9; state=None; bad=0; nneg=5 if soft else 1
    for ep in range(40):
        neg=sample(pools,rng,nneg); order=rng.permutation(len(ps)); m.train()
        for j in range(0,len(order),8192):
            sel=order[j:j+8192]; xp=Xt[torch.from_numpy(ps[sel])].to(device); sp=m(xp)
            opt.zero_grad(set_to_none=True)
            if soft:
                xn=Xt[torch.from_numpy(neg[sel].reshape(-1))].to(device); sn=m(xn).view(len(sel),nneg)
                loss=(torch.nn.functional.softplus(-(sp[:,None]-sn))*torch.softmax(sn.detach(),dim=1)).sum(1).mean()
            else:
                xn=Xt[torch.from_numpy(neg[sel])].to(device); loss=torch.nn.functional.softplus(-(sp-m(xn))).mean()
            loss.backward(); opt.step()
        va=evaluate(uv,yv,m.predict(Xv,device=device))['primary']
        if va>best+1e-5: best=va; bad=0; state={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=4: break
    m.load_state_dict(state)
    return m.predict(enc[target][0],device=device).astype(np.float64)

def member(name,enc,dim,target,split,seed,device):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'010_{name}_seed{seed}_{split}.npy')
    if os.path.isfile(path):
        p=np.load(path)
        if len(p)==len(enc[target][0]): return p.astype(np.float64)
    p=train_member(enc,dim,target,seed,soft=(name=='soft5_tau1'),device=device); np.save(path,p); return p

def groups(users):
    d={}
    for i,u in enumerate(users): d.setdefault(u,[]).append(i)
    return [np.asarray(v,dtype=np.int64) for v in d.values()]

def zscore(p,gs):
    o=np.empty_like(p,dtype=np.float64)
    for ix in gs:
        v=p[ix]; s=v.std(); o[ix]=(v-v.mean())/s if s>1e-12 else 0.0
    return o

def rankpct(p,gs,pow=2.0):
    o=np.empty_like(p,dtype=np.float64)
    for ix in gs:
        n=len(ix)
        if n<=1: o[ix]=0; continue
        ord=np.argsort(p[ix],kind='mergesort'); r=np.empty(n); r[ord]=np.arange(n); o[ix]=(r/(n-1.0))**pow
    return o

def add(d,k,y):
    s,c=d.get(k,(0.0,0)); d[k]=(s+float(y),c+1)

def dev(d,k,base,a):
    s,c=d.get(k,(0.0,0))
    return ((s+a*base)/(c+a)-base) if c>0 else 0.0

def hist_signal(splits,target):
    us={}; uv={}; ua={}; ut={}; sm=0.0; cnt=0
    for r in splits['train']:
        u,v,au,t,y=r[1],r[2],r[3],r[4],r[6]; add(us,u,y); add(uv,(u,v),y); add(ua,(u,au),y); add(ut,(u,t),y); sm+=float(y); cnt+=1
    gm=sm/max(1,cnt); out=np.zeros(len(splits[target]))
    for i,r in enumerate(splits[target]):
        u,v,au,t=r[1],r[2],r[3],r[4]; s,c=us.get(u,(gm,0)); base=s/c if c else gm
        out[i]=dev(uv,(u,v),base,1.0)+0.45*dev(ua,(u,au),base,5.0)+0.20*dev(ut,(u,t),base,10.0)
    return out

def itemcf_signal(splits,target,topk=80):
    umap={}; amap={}; uh={}
    def uid(x):
        if x not in umap: umap[x]=len(umap)
        return umap[x]
    def aid(x):
        if x not in amap: amap[x]=len(amap)
        return amap[x]
    for r in splits[target]: aid(r[3])
    for r in splits['train']:
        ui=uid(r[1]); ai=aid(r[3])
        if float(r[6])>0.5: uh.setdefault(ui,set()).add(ai)
    rows=[]; cols=[]; vals=[]
    for ui,s in uh.items():
        w=1.0/np.sqrt(max(1,len(s)))
        for ai in s: rows.append(ui); cols.append(ai); vals.append(w)
    if not rows: return np.zeros(len(splits[target]))
    M=csr_matrix((np.asarray(vals,dtype=np.float32),(np.asarray(rows),np.asarray(cols))),shape=(max(1,len(umap)),max(1,len(amap))),dtype=np.float32)
    C=(M.T@M).tocsr(); diag=np.asarray(C.diagonal(),dtype=np.float64); inv=np.zeros_like(diag); inv[diag>1e-12]=1/np.sqrt(diag[diag>1e-12])
    neigh=[]
    for a in range(C.shape[0]):
        st,en=C.indptr[a],C.indptr[a+1]; idx=C.indices[st:en]; dat=C.data[st:en].astype(np.float64)
        sim=dat*inv[a]*inv[idx] if inv[a]>0 and len(idx) else np.asarray([])
        keep=(idx!=a)&(sim>0) if len(idx) else []
        idx=idx[keep]; sim=sim[keep]
        if len(idx)>topk:
            part=np.argpartition(sim,-topk)[-topk:]; idx=idx[part]; sim=sim[part]
        neigh.append((idx.astype(np.int32),sim.astype(np.float64)))
    out=np.zeros(len(splits[target]))
    for i,r in enumerate(splits[target]):
        ui=umap.get(r[1]); ai=amap.get(r[3]); h=uh.get(ui) if ui is not None else None
        if h is None or ai is None: continue
        idx,sim=neigh[ai]; s=0.0
        for j,x in enumerate(idx):
            if int(x) in h: s+=sim[j]
        out[i]=s
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu')
    a=ap.parse_args(); torch.manual_seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; split='dev'
    else:
        splits=load(a.data_dir); target=a.split; split=a.split
    enc,dim=encode(splits); gs=groups(enc[target][2])
    zb=[]; zs=[]; rb=[]; rs=[]
    for s in [0,1,2]:
        pb=member('bpr1',enc,dim,target,split,s,a.device); ps=member('soft5_tau1',enc,dim,target,split,s,a.device)
        zb.append(zscore(pb,gs)); zs.append(zscore(ps,gs)); rb.append(rankpct(pb,gs,2.0)); rs.append(rankpct(ps,gs,2.0))
    score_z=0.60*np.mean(zb,0)+0.40*np.mean(zs,0)
    score_rank=zscore(0.60*np.mean(rb,0)+0.40*np.mean(rs,0),gs)
    ens=0.40*score_z+0.60*score_rank
    hist=zscore(hist_signal(splits,target),gs)
    best=0.90*ens+0.10*hist
    cf=zscore(itemcf_signal(splits,target),gs)
    # Debug node 27: the positive 20% CF blend was strongly harmful, so test
    # whether this co-occurrence score is actually an anti-novelty residual.
    scores=best-0.10*cf
    if a.out: np.save(a.out,scores.astype(np.float64))
    else: print({k:len(v) for k,v in splits.items()}, FIELDS)
