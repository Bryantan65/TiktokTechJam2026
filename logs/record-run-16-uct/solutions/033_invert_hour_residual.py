"""Invert the live hour residual from node 032.

Node 032 proved the raw hour join is live but an 8% positive residual hurt GAUC.
If the simple hour encoding is directionally anti-correlated with within-user
ranking errors, a small negative blend should recover signal without disturbing
node-024's strong cached ensemble.
"""
import argparse, csv, os, sys
from collections import defaultdict, deque
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

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

def make_user_pairs(y,users):
    pos,neg={},{}
    for i,(yy,u) in enumerate(zip(y,users)): (pos if yy>0.5 else neg).setdefault(u,[]).append(i)
    pa=[]; pools=[]
    for u,ps in pos.items():
        ns=neg.get(u)
        if ns:
            na=np.asarray(ns,dtype=np.int64)
            for p in ps: pa.append(p); pools.append(na)
    return np.asarray(pa,dtype=np.int64),pools

def sample_negatives(pools,rng,n):
    if n==1:
        out=np.empty(len(pools),dtype=np.int64)
        for i,p in enumerate(pools): out[i]=p[rng.integers(len(p))]
    else:
        out=np.empty((len(pools),n),dtype=np.int64)
        for i,p in enumerate(pools): out[i]=p[rng.integers(len(p),size=n)]
    return out

def train_bpr(enc,dim,target,seed=0,n_neg=1,soft_hard=False,tau=1.0,device='cpu',verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; pos,pools=make_user_pairs(ytr,utr)
    model=TorchFM(dim,seed=seed).to(device); opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=0.001)
    Xtrt=torch.from_numpy(Xtr.astype(np.int64)); rng=np.random.default_rng(seed); best=-1; state=None; bad=0
    for ep in range(40):
        neg=sample_negatives(pools,rng,n_neg); order=rng.permutation(len(pos)); model.train()
        for j in range(0,len(order),8192):
            sel=order[j:j+8192]; xp=Xtrt[torch.from_numpy(pos[sel])].to(device); sp=model(xp); opt.zero_grad(set_to_none=True)
            if soft_hard:
                xn=Xtrt[torch.from_numpy(neg[sel].reshape(-1))].to(device); sn=model(xn).view(len(sel),n_neg)
                loss=(torch.nn.functional.softplus(-(sp.view(-1,1)-sn))*torch.softmax((sn/tau).detach(),dim=1)).sum(1).mean()
            else:
                xn=Xtrt[torch.from_numpy(neg[sel])].to(device); loss=torch.nn.functional.softplus(-(sp-model(xn))).mean()
            loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=4: break
    model.load_state_dict(state); return model.predict(enc[target][0],device=device).astype(np.float64)

def get_member(name,enc,dim,target,split_name,seed,device,verbose):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'010_{name}_seed{seed}_{split_name}.npy')
    if os.path.isfile(path):
        p=np.load(path)
        if len(p)==len(enc[target][0]): return p.astype(np.float64)
    p=train_bpr(enc,dim,target,seed=seed,n_neg=(1 if name=='bpr1' else 5),soft_hard=(name!='bpr1'),tau=1.0,device=device,verbose=verbose)
    np.save(path,p); return p

def user_groups(users):
    d={}
    for i,u in enumerate(users): d.setdefault(u,[]).append(i)
    return [np.asarray(v,dtype=np.int64) for v in d.values()]

def per_user_z(p,groups):
    out=np.empty_like(p,dtype=np.float64)
    for idx in groups:
        v=p[idx]; sd=v.std(); out[idx]=(v-v.mean())/sd if sd>1e-12 else 0.0
    return out

def per_user_rank(p,groups,power=2.0):
    out=np.empty_like(p,dtype=np.float64)
    for idx in groups:
        n=len(idx)
        if n<=1: out[idx]=0.0; continue
        order=np.argsort(p[idx],kind='mergesort'); r=np.empty(n,dtype=np.float64); r[order]=np.arange(n,dtype=np.float64)
        out[idx]=(r/(n-1.0))**power
    return out

def add_stat(d,k,y):
    s,c=d.get(k,(0.0,0)); d[k]=(s+float(y),c+1)
def smoothed_dev(d,k,base,alpha):
    s,c=d.get(k,(0.0,0)); return ((s+alpha*base)/(c+alpha)-base) if c else 0.0

def history_signal(splits,target):
    us,uv,ua,ut={},{},{},{}; gs=0.0; gc=0
    for row in splits['train']:
        u,v,a,t,y=row[1],row[2],row[3],row[4],float(row[6]); add_stat(us,u,y); add_stat(uv,(u,v),y); add_stat(ua,(u,a),y); add_stat(ut,(u,t),y); gs+=y; gc+=1
    gm=gs/max(1,gc); out=np.zeros(len(splits[target]),dtype=np.float64)
    for i,row in enumerate(splits[target]):
        u,v,a,t=row[1],row[2],row[3],row[4]; s,c=us.get(u,(gm,0)); base=s/c if c else gm
        out[i]=smoothed_dev(uv,(u,v),base,1.0)+0.45*smoothed_dev(ua,(u,a),base,5.0)+0.20*smoothed_dev(ut,(u,t),base,10.0)
    return out

def norm_int(x):
    try: return str(int(float(str(x))))
    except Exception: return str(x)
def norm_id(x):
    s=str(x)
    if s.endswith('.0'):
        try: return str(int(float(s)))
        except Exception: pass
    return s
def row_key(row): return (norm_int(row[0]),norm_id(row[1]),norm_id(row[2]),norm_id(row[3]),norm_int(row[4]),norm_int(row[5]))
def pick(rec,names):
    for n in names:
        if n in rec and rec[n]!='': return rec[n]
    return ''
def parse_hour(x):
    if x is None or x=='': return -1
    try: v=int(float(str(x)))
    except Exception: return -1
    if 0<=v<=23: return v
    if 0<=v<=2359:
        h=v//100; return h if 0<=h<=23 else -1
    if 0<=v<86400:
        h=v//3600; return h if 0<=h<=23 else -1
    return -1

def raw_files(data_dir):
    bases=[data_dir,os.path.join(data_dir,'data'),os.path.dirname(data_dir),os.path.join(os.path.dirname(data_dir),'data')]
    out=[]
    for nm in ['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']:
        fp=None
        for b in bases:
            p=os.path.join(b,nm)
            if os.path.isfile(p): fp=p; break
        if fp is None: return []
        out.append(fp)
    return out

def load_raw_hours(data_dir):
    keyq,dateq=defaultdict(deque),defaultdict(deque)
    for fp in raw_files(data_dir):
        with open(fp,newline='',encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                h=parse_hour(pick(rec,['hourmin','hour_min','request_time','time','timestamp'])); date=norm_int(pick(rec,['date']))
                key=(date,norm_id(pick(rec,['user_id','user_id_str'])),norm_id(pick(rec,['video_id','photo_id'])),norm_id(pick(rec,['author_id'])),norm_int(pick(rec,['tab'])),norm_int(pick(rec,['duration_ms','duration','video_duration'])))
                keyq[key].append(h); dateq[date].append(h)
    return keyq,dateq

def aligned_hours(splits,data_dir):
    keyq,dateq_all=load_raw_hours(data_dir); out={}
    for sp in ['train','valid','test']:
        if sp not in splits: continue
        arr=np.full(len(splits[sp]),-1,dtype=np.int16)
        for i,row in enumerate(splits[sp]):
            q=keyq.get(row_key(row))
            if q: arr[i]=q.popleft()
        out[sp]=arr
    alln=sum(len(v) for v in out.values()); hit=sum(int((v>=0).sum()) for v in out.values())
    if alln and hit<0.5*alln and dateq_all:
        dq={k:deque(v) for k,v in dateq_all.items()}
        for sp in ['train','valid','test']:
            if sp not in splits: continue
            arr=np.full(len(splits[sp]),-1,dtype=np.int16)
            for i,row in enumerate(splits[sp]):
                q=dq.get(norm_int(row[0]))
                if q: arr[i]=q.popleft()
            out[sp]=arr
    return out

def hour_signal(splits,target,data_dir):
    hrs=aligned_hours(splits,data_dir); htr=hrs.get('train',np.full(len(splits['train']),-1,dtype=np.int16)); hta=hrs.get(target,np.full(len(splits[target]),-1,dtype=np.int16))
    us,ts,uh,uth,th,gh={},{},{},{},{},{}; gs=0.0; gc=0
    for row,h in zip(splits['train'],htr):
        u,t,y=row[1],row[4],float(row[6]); add_stat(us,u,y); add_stat(ts,t,y); gs+=y; gc+=1
        if 0<=int(h)<=23:
            hh=int(h); add_stat(uh,(u,hh),y); add_stat(uth,(u,t,hh//4),y); add_stat(th,(t,hh),y); add_stat(gh,hh,y)
    gm=gs/max(1,gc); out=np.zeros(len(splits[target]),dtype=np.float64)
    for i,(row,h) in enumerate(zip(splits[target],hta)):
        if 0<=int(h)<=23:
            u,t=row[1],row[4]; hh=int(h); s,c=us.get(u,(gm,0)); ub=s/c if c else gm; s,c=ts.get(t,(gm,0)); tb=s/c if c else gm
            out[i]=0.75*smoothed_dev(uh,(u,hh),ub,6.0)+0.40*smoothed_dev(uth,(u,t,hh//4),ub,10.0)+0.30*smoothed_dev(th,(t,hh),tb,60.0)+0.15*smoothed_dev(gh,hh,gm,150.0)
    return out, hta

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; split_name='dev'
    else:
        splits=load(a.data_dir); target=a.split; split_name=a.split
    enc,dim=encode(splits); groups=user_groups(enc[target][2])
    zb=[]; zs=[]; rb=[]; rs=[]
    for s in [0,1,2]:
        pb=get_member('bpr1',enc,dim,target,split_name,s,a.device,a.out is None); ps=get_member('soft5_tau1',enc,dim,target,split_name,s,a.device,a.out is None)
        zb.append(per_user_z(pb,groups)); zs.append(per_user_z(ps,groups)); rb.append(per_user_rank(pb,groups,2.0)); rs.append(per_user_rank(ps,groups,2.0))
    score_z=0.60*np.mean(zb,axis=0)+0.40*np.mean(zs,axis=0); score_rank=0.60*np.mean(rb,axis=0)+0.40*np.mean(rs,axis=0)
    ensemble=0.40*score_z+0.60*per_user_z(score_rank,groups); hist=per_user_z(history_signal(splits,target),groups); base=0.90*ensemble+0.10*hist
    raw,hta=hour_signal(splits,target,a.data_dir); hour=per_user_z(raw,groups)
    if float(np.std(hour))<1e-12: hour=per_user_z(np.where(hta>=0,np.sin(2*np.pi*hta/24.0),0.0).astype(np.float64),groups)
    scores=1.04*base-0.04*hour
    if a.out: np.save(a.out,scores.astype(np.float64))
    else: print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}', 'hour_std', float(np.std(hour)))
