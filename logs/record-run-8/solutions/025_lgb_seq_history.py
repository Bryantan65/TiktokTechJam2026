import argparse, csv, os, sys, time
from collections import defaultdict, deque
import numpy as np
import torch
import lightgbm as lgb


def _add_path():
    here=os.path.abspath(os.path.dirname(__file__))
    for p in [os.path.join(os.getcwd(),'kuairand-starter-kit'), os.path.join(here,'..','kuairand-starter-kit'), os.path.join(here,'..','..','kuairand-starter-kit'), os.path.join(here,'..','..','..','kuairand-starter-kit')]:
        p=os.path.abspath(p)
        if os.path.isdir(p): sys.path.insert(0,p); return
    sys.path.insert(0,'kuairand-starter-kit')
_add_path()
from data import load, encode, FIELDS
from evaluate import evaluate


def toi(x, default=0):
    try:
        if x is None or x=='': return default
        return int(float(x))
    except Exception:
        return default

def raw_logs(data_dir):
    for base in [data_dir, os.path.join(data_dir,'data'), os.path.dirname(data_dir)]:
        fs=[os.path.join(base,'log_standard_4_08_to_4_21_pure.csv'), os.path.join(base,'log_standard_4_22_to_5_08_pure.csv')]
        if all(os.path.exists(f) for f in fs): return fs
    return [os.path.join(data_dir,'log_standard_4_08_to_4_21_pure.csv'), os.path.join(data_dir,'log_standard_4_22_to_5_08_pure.csv')]

def read_raw_time(data_dir):
    rec=[]; dc=hc=None
    for fp in raw_logs(data_dir):
        if not os.path.exists(fp): continue
        with open(fp, newline='', encoding='utf-8') as f:
            rd=csv.DictReader(f); fields=rd.fieldnames or []
            dcol=next((c for c in ['date','request_date','upload_date'] if c in fields), None)
            hcol=next((c for c in ['hourmin','hour_min','time','request_time'] if c in fields), None)
            dc=dc or dcol; hc=hc or hcol
            for r in rd:
                d=toi(r.get(dcol,0)) if dcol else 0; hm=toi(r.get(hcol,0)) if hcol else 0
                if hm>2359: hm%=10000
                rec.append((d,hm))
    print(f'raw time rows={len(rec):,d}; date_col={dc}; hour_col={hc}')
    return rec

def dord(yyyymmdd):
    z=int(yyyymmdd); return (((z//100)%100)-4)*31 + z%100

def build_time_maps(data_dir, splits):
    raw=read_raw_time(data_dir); out={}
    for sp, rows in splits.items():
        dates=set(toi(r[0]) for r in rows)
        sel=[(d,hm) for d,hm in raw if d in dates]
        if len(sel)<len(rows):
            print(f'time alignment {sp}: only {len(sel):,d}/{len(rows):,d}; falling back for missing rows')
            sel=sel+[ (toi(r[0]),0) for r in rows[len(sel):] ]
        else: sel=sel[:len(rows)]
        out[sp]=sel; print(f'time alignment {sp}: {min(len(sel),len(rows)):,d}/{len(rows):,d}')
    return out

def time_feats(rows, rt):
    X=np.zeros((len(rows),7),np.float32)
    for i,r in enumerate(rows):
        d,hm=(rt[i] if i<len(rt) else (toi(r[0]),0)); d=d or toi(r[0])
        h=max(0,min(23,hm//100)); m=max(0,min(59,hm%100)); tab=toi(r[4]); od=dord(d)
        X[i]=[h,h//4,tab*24+h,od,od/40.0,(h*60+m)/1440.0,(od%7)/6.0]
    return X

class FM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
    def forward(self,X):
        E=self.V[X]; S=E.sum(1)
        return self.b+self.W[X].sum(1)+0.5*((S*S).sum(1)-(E*E).sum((1,2)))
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def bpr_groups(y,u):
    p=defaultdict(list); n=defaultdict(list)
    for i,(yy,uu) in enumerate(zip(y,u)): (p if yy>0.5 else n)[uu].append(i)
    return [(np.asarray(pp,np.int64),np.asarray(n[uu],np.int64)) for uu,pp in p.items() if uu in n]

def sample(gs,rng):
    ps=[]; ns=[]
    for p,n in gs: ps.append(p); ns.append(rng.choice(n,size=len(p),replace=True))
    pi=np.concatenate(ps); ni=np.concatenate(ns); o=rng.permutation(len(pi)); return pi[o],ni[o]

def train_fm(enc,dim,seed=0,k=16,lr=0.001,epochs=40,patience=4,bs=8192,bce_weight=0.15,device='cpu',verbose=False,tag=''):
    Xtr,ytr,utr=enc['train']; Xv,yv,uv=enc['valid']; gs=bpr_groups(ytr,utr)
    if verbose: print(f'{tag} BPR eligible users={len(gs):,d}, pairs={sum(len(p) for p,_ in gs):,d}')
    torch.manual_seed(seed); model=FM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); rng=np.random.default_rng(seed); best=-1; state=None; bad=0
    for ep in range(1,epochs+1):
        t=time.time(); pi,ni=sample(gs,rng); model.train(); losses=[]
        for i in range(0,len(pi),bs):
            xp=Xt[torch.from_numpy(pi[i:i+bs])].to(device); xn=Xt[torch.from_numpy(ni[i:i+bs])].to(device)
            opt.zero_grad(set_to_none=True); sp=model(xp); sn=model(xn)
            loss=-torch.nn.functional.logsigmoid(sp-sn).mean()+bce_weight*0.5*(torch.nn.functional.softplus(-sp).mean()+torch.nn.functional.softplus(sn).mean())
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        val=evaluate(uv,yv,model.predict(Xv,device=device))['primary']
        if verbose: print(f'  {tag} epoch {ep:2d} loss {np.mean(losses):.4f} valid {val:.4f} {time.time()-t:.1f}s')
        if val>best+1e-5: best=val; bad=0; state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(state); return model

def pct(scores,users):
    out=np.zeros(len(scores),np.float64); g=defaultdict(list)
    for i,u in enumerate(users): g[u].append(i)
    for idx in g.values():
        idx=np.asarray(idx,np.int64); n=len(idx)
        if n==1: out[idx[0]]=0.0; continue
        o=np.argsort(scores[idx],kind='mergesort'); r=np.empty(n,np.float64); r[o]=np.arange(n)/float(n-1); out[idx]=r
    return out

def fm_ens(ms,X,u,device='cpu'):
    z=np.zeros(len(X),np.float64)
    for m in ms: z+=pct(m.predict(X,device=device),u)
    return z/len(ms)

def dur_bucket(r): return int(min(20,max(0,int(r[5])//5000)))
def keys_for(r):
    db=dur_bucket(r)
    return db, [r[2],r[3],r[4],db,(r[3],r[4])], [(r[1],r[2]),(r[1],r[3]),(r[1],r[4]),(r[1],db),(r[1],r[3],r[4])]

def stat_maps(rows):
    gc=sum(float(r[6]) for r in rows)/max(1,len(rows)); gnames=['video','author','tab','dur','author_tab']; unames=['u_video','u_author','u_tab','u_dur','u_author_tab']
    sums={k:defaultdict(float) for k in gnames+unames}; cnts={k:defaultdict(int) for k in gnames+unames}
    for r in rows:
        y=float(r[6]); _,gkeys,ukeys=keys_for(r)
        for k,key in zip(gnames,gkeys): sums[k][key]+=y; cnts[k][key]+=1
        for k,key in zip(unames,ukeys): sums[k][key]+=y; cnts[k][key]+=1
    return gc,sums,cnts

def stat_feats(rows,pack,loo=False):
    gc,sums,cnts=pack; gnames=['video','author','tab','dur','author_tab']; unames=['u_video','u_author','u_tab','u_dur','u_author_tab']
    X=np.zeros((len(rows),21),np.float32)
    for i,r in enumerate(rows):
        y=float(r[6]) if loo else 0.0; _,gkeys,ukeys=keys_for(r); vals=[]
        for k,key in zip(gnames,gkeys):
            c=cnts[k].get(key,0); s=sums[k].get(key,0.0); vals += [np.log1p(c),(s+20*gc)/(c+20)]
        for k,key in zip(unames,ukeys):
            c=cnts[k].get(key,0); s=sums[k].get(key,0.0)
            if loo: c=max(0,c-1); s=max(0.0,s-y)
            vals += [np.log1p(c),(s+10*gc)/(c+10)]
        X[i,:20]=vals; X[i,20]=np.log1p(max(float(r[5]),0.0))
    return X

def build_seq_maps(train_rows):
    mp={}
    for r in train_rows:
        u=r[1]; y=float(r[6]); db=dur_bucket(r)
        if u not in mp:
            mp[u]={'n':0,'pos':0,'auth':defaultdict(int),'tab':defaultdict(int),'dur':defaultdict(int),'vid':defaultdict(int),
                   'all_auth':defaultdict(int),'all_tab':defaultdict(int),'last_pos':deque(maxlen=20),'last_all':deque(maxlen=20)}
        s=mp[u]; s['n']+=1; s['all_auth'][r[3]]+=1; s['all_tab'][r[4]]+=1; s['last_all'].append((r[2],r[3],r[4],db,y))
        if y>0.5:
            s['pos']+=1; s['auth'][r[3]]+=1; s['tab'][r[4]]+=1; s['dur'][db]+=1; s['vid'][r[2]]+=1; s['last_pos'].append((r[2],r[3],r[4],db))
    return mp

def seq_feats(rows,mp):
    X=np.zeros((len(rows),14),np.float32)
    for i,r in enumerate(rows):
        u=r[1]; db=dur_bucket(r); s=mp.get(u)
        if not s: continue
        n=max(1,s['n']); p=max(1,s['pos'])
        la=list(s['last_all']); lp=list(s['last_pos'])
        vals=[np.log1p(s['n']),np.log1p(s['pos']),s['pos']/n,
              np.log1p(s['auth'].get(r[3],0)),s['auth'].get(r[3],0)/p,
              np.log1p(s['tab'].get(r[4],0)),s['tab'].get(r[4],0)/p,
              np.log1p(s['dur'].get(db,0)),s['dur'].get(db,0)/p,
              np.log1p(s['vid'].get(r[2],0)),
              np.log1p(s['all_auth'].get(r[3],0)),s['all_auth'].get(r[3],0)/n,
              np.log1p(s['all_tab'].get(r[4],0)),s['all_tab'].get(r[4],0)/n]
        # overwrite last two with recency-style matches to keep compact? no, append not possible; use strong recency via additions to ratios
        if lp:
            vals[4]+=0.5*sum((a==r[3])*(0.90**j) for j,(_,a,_,_) in enumerate(reversed(lp)))/len(lp)
            vals[6]+=0.5*sum((t==r[4])*(0.90**j) for j,(_,_,t,_) in enumerate(reversed(lp)))/len(lp)
            vals[8]+=0.5*sum((d==db)*(0.90**j) for j,(_,_,_,d) in enumerate(reversed(lp)))/len(lp)
        if la:
            vals[11]+=0.25*sum((a==r[3])*(0.90**j) for j,(_,a,_,_,_) in enumerate(reversed(la)))/len(la)
            vals[13]+=0.25*sum((t==r[4])*(0.90**j) for j,(_,_,t,_,_) in enumerate(reversed(la)))/len(la)
        X[i]=vals
    return X

def lgb_mat(encX,rows,pack,fmr,tf,seq,loo=False):
    return np.hstack([encX[:,1:].astype(np.int32),tf.astype(np.float32),stat_feats(rows,pack,loo=loo),seq.astype(np.float32),fmr.reshape(-1,1).astype(np.float32)]).astype(np.float32)

def sort_user(X,y,u,init):
    ua=np.asarray(u); o=np.argsort(ua,kind='mergesort'); su=ua[o]; gr=[]; last=None; c=0
    for uu in su:
        if last is None or uu==last: c+=1
        else: gr.append(c); c=1
        last=uu
    if c: gr.append(c)
    return X[o],y[o],gr,init[o]

def train_lgb(enc,splits,fmtr,fmva,tmap,seed=0,verbose=False):
    pack=stat_maps(splits['train']); smap=build_seq_maps(splits['train']); Xte,ytr,utr=enc['train']; Xve,yva,uva=enc['valid']
    Xtr=lgb_mat(Xte,splits['train'],pack,fmtr,time_feats(splits['train'],tmap['train']),seq_feats(splits['train'],smap),loo=True)
    Xva=lgb_mat(Xve,splits['valid'],pack,fmva,time_feats(splits['valid'],tmap['valid']),seq_feats(splits['valid'],smap),loo=False)
    Xs,ys,gs,ins=sort_user(Xtr,ytr.astype(np.float32),utr,fmtr.astype(np.float64)); Xvs,yvs,gvs,invs=sort_user(Xva,yva.astype(np.float32),uva,fmva.astype(np.float64))
    cat=list(range(8))
    dtr=lgb.Dataset(Xs,label=ys,group=gs,init_score=ins,categorical_feature=cat,free_raw_data=False)
    dva=lgb.Dataset(Xvs,label=yvs,group=gvs,init_score=invs,categorical_feature=cat,reference=dtr,free_raw_data=False)
    params={'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[5],'label_gain':[0,1],'learning_rate':0.03,'num_leaves':31,'min_data_in_leaf':120,'feature_fraction':0.90,'bagging_fraction':0.90,'bagging_freq':1,'lambda_l2':5.0,'verbosity':-1,'seed':int(seed),'feature_fraction_seed':int(seed)+11,'bagging_seed':int(seed)+17,'num_threads':4}
    cb=[lgb.early_stopping(25,verbose=verbose)]
    if not verbose: cb.append(lgb.log_evaluation(period=0))
    model=lgb.train(params,dtr,num_boost_round=250,valid_sets=[dva],valid_names=['valid'],callbacks=cb)
    return model,pack,smap

def train_all(splits,data_dir,seed=0,device='cpu',verbose=False,n_models=5):
    enc,dim=encode(splits); fms=[]
    for m in range(n_models): fms.append(train_fm(enc,dim,seed=int(seed+997*m),device=device,verbose=verbose,tag=f'm{m}/seed{seed+997*m}'))
    Xtr,_,utr=enc['train']; Xva,_,uva=enc['valid']; fmtr=fm_ens(fms,Xtr,utr,device); fmva=fm_ens(fms,Xva,uva,device)
    tmap=build_time_maps(data_dir,splits); model,pack,smap=train_lgb(enc,splits,fmtr,fmva,tmap,seed=seed+4242,verbose=verbose)
    return fms,model,pack,smap,tmap,enc

def predict(fms,model,pack,smap,tmap,encX,rows,users,sp,device='cpu',fm_weight=0.50):
    fmr=fm_ens(fms,encX,users,device); seq=seq_feats(rows,smap); X=lgb_mat(encX,rows,pack,fmr,time_feats(rows,tmap.get(sp,[])),seq,loo=False)
    corr=pct(fmr + model.predict(X,num_iteration=model.best_iteration), users)
    return fm_weight*fmr + (1.0-fm_weight)*corr

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); ap.add_argument('--n_models',type=int,default=5); ap.add_argument('--fm_weight',type=float,default=0.50)
    a=ap.parse_args(); torch.manual_seed(a.seed)
    print(f'loading {a.data_dir} ...')
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}, lgb_seq_history fm_weight={a.fm_weight}')
    fms,model,pack,smap,tmap,enc=train_all(splits,a.data_dir,seed=a.seed,device=a.device,verbose=(a.out is None),n_models=a.n_models)
    X,y,u=enc[target]; scores=predict(fms,model,pack,smap,tmap,X,splits[target],u,target,device=a.device,fm_weight=a.fm_weight)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        for sp in ('valid','test'):
            if sp in enc:
                Xs,ys,us=enc[sp]; r=evaluate(us,ys,predict(fms,model,pack,smap,tmap,Xs,splits[sp],us,sp,device=a.device,fm_weight=a.fm_weight)); print(sp,r)
