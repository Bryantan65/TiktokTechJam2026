import argparse, csv, glob, os, sys, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, FIELDS
from evaluate import evaluate

DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)

class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32)); self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        return self.b + self.W[X].sum(1) + 0.5 * ((S * S).sum(1) - (E * E).sum((1, 2)))
    @torch.no_grad()
    def predict(self, X, bs=200000, device='cpu'):
        self.eval(); out=[]
        for i in range(0, len(X), bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def norm(x):
    try: return str(int(float(x)))
    except Exception: return str(x)

def row_key(r): return (norm(r[DATE]), norm(r[USER]), norm(r[VIDEO]), norm(r[TAB]))
def raw_key(rec): return (norm(rec.get('date','')), norm(rec.get('user_id','')), norm(rec.get('video_id','')), norm(rec.get('tab','')))

def find_logs(data_dir):
    files=[os.path.join(data_dir,n) for n in ['log_standard_4_08_to_4_21_1k.csv','log_standard_4_22_to_5_08_1k.csv']]
    if all(os.path.isfile(p) for p in files): return files
    out=[]
    for pat in ['log_standard_4_08_to_4_21*.csv','log_standard_4_22_to_5_08*.csv']:
        g=sorted(glob.glob(os.path.join(data_dir,pat)))
        if g: out.append(g[0])
    return out

def parse_hourmin(x):
    try: hm=int(float(x))
    except Exception: return -1,-1,-1
    h,m=hm//100,hm%100
    if h<0 or h>23 or m<0 or m>59: return -1,-1,-1
    return h,h*6+(m//10),h//4

def read_time_ordered(data_dir, rows, name):
    n=len(rows); hour=np.full(n,-1,np.int16); ten=np.full(n,-1,np.int16); block=np.full(n,-1,np.int16)
    files=find_logs(data_dir)
    if not files or n==0:
        print('warning: raw logs not found for', name, flush=True); return hour,ten,block
    i=0; cur=row_key(rows[0]); seen=0; t0=time.time()
    for path in files:
        if i>=n: break
        with open(path,'r',encoding='utf-8',newline='') as f:
            for rec in csv.DictReader(f):
                seen+=1
                if raw_key(rec)==cur:
                    h,te,bl=parse_hourmin(rec.get('hourmin','')); hour[i],ten[i],block[i]=h,te,bl; i+=1
                    if i>=n: break
                    cur=row_key(rows[i])
    print(f'aligned hourmin for {name}: {i:,d}/{n:,d} rows after scanning {seen:,d} raw rows in {time.time()-t0:.1f}s', flush=True)
    return hour,ten,block

def dur_bucket(x): return int(np.log1p(float(x)) // 1)

def make_encoded(splits, aux, names):
    maps=[{} for _ in range(14)]; feats={}; ys={}; raw_users={}
    for sp in names:
        rows=splits[sp]; h,ten,block=aux[sp]; fs=[]; y=np.empty(len(rows),np.float32); ru=[]
        for i,r in enumerate(rows):
            u=r[USER]; t=r[TAB]; hh=int(h[i]); te=int(ten[i]); bl=int(block[i])
            vals=[u,r[VIDEO],r[AUTHOR],t,dur_bucket(r[DUR]),r[DATE],hh,te,bl,(t,hh),(t,te),(u,t),(u,hh),(u,bl)]
            fs.append(vals); y[i]=float(r[LABEL]); ru.append(u)
            for j,v in enumerate(vals):
                if v not in maps[j]: maps[j][v]=len(maps[j])
        feats[sp]=fs; ys[sp]=y; raw_users[sp]=ru
    offsets=np.cumsum([0]+[len(m) for m in maps[:-1]]).astype(np.int64); dim=int(sum(len(m) for m in maps)); enc={}; user_map={}
    for sp in names:
        fs=feats[sp]; X=np.empty((len(fs),len(maps)),np.int64); uarr=np.empty(len(fs),np.int64)
        for i,vals in enumerate(fs):
            for j,v in enumerate(vals): X[i,j]=maps[j][v]+offsets[j]
        for i,raw_u in enumerate(raw_users[sp]):
            if raw_u not in user_map: user_map[raw_u]=len(user_map)
            uarr[i]=user_map[raw_u]
        enc[sp]=(X,ys[sp],uarr)
    return enc,dim

def make_sampler(y, users, tabs):
    users=np.asarray(users); tabs=np.asarray(tabs); y=np.asarray(y); order=np.argsort(users,kind='mergesort'); su=users[order]
    pos_list=[]; neg_by_user={}; neg_by_ut={}; s=0; n=len(users)
    while s<n:
        e=s+1
        while e<n and su[e]==su[s]: e+=1
        rows=order[s:e]; pos=rows[y[rows]>0.5]; neg=rows[y[rows]<=0.5]
        if len(pos) and len(neg):
            u=su[s]; pos_list.append(pos); neg_by_user[u]=neg.astype(np.int64); nt=tabs[neg]
            for t in np.unique(nt): neg_by_ut[(u,t)]=neg[nt==t].astype(np.int64)
        s=e
    return np.concatenate(pos_list).astype(np.int64), neg_by_user, neg_by_ut

def sample_pairs(pos_rows, users, tabs, neg_by_user, neg_by_ut, rng):
    perm=rng.permutation(len(pos_rows)); p=pos_rows[perm]; pu=users[p]; pt=tabs[p]; negs=np.empty((len(p),3),np.int64)
    for i,(u,t) in enumerate(zip(pu,pt)):
        pool=neg_by_user[u]; negs[i,:2]=pool[rng.integers(len(pool),size=2)]; hp=neg_by_ut.get((u,t),pool); negs[i,2]=hp[rng.integers(len(hp))]
    return p,negs

def prepare(splits,data_dir,target):
    names=['train','valid']
    if target not in names: names.append(target)
    aux={sp:read_time_ordered(data_dir,splits[sp],sp) for sp in names}
    return make_encoded(splits,aux,names)

def train_predict_member(enc, dim, target, seed, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k=k,seed=seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); pos_rows,neg_by_user,neg_by_ut=make_sampler(ytr,utr,Xtr[:,3]); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0; n_neg=3
    for ep in range(1,epochs+1):
        pidx,nidx=sample_pairs(pos_rows,utr,Xtr[:,3],neg_by_user,neg_by_ut,rng); model.train(); losses=[]
        for i in range(0,len(pidx),bs):
            ps=torch.from_numpy(pidx[i:i+bs]).long(); ns=torch.from_numpy(nidx[i:i+bs].reshape(-1)).long(); xp=Xtr_t[ps].to(device); xn=Xtr_t[ns].to(device)
            opt.zero_grad(set_to_none=True); loss=-torch.nn.functional.logsigmoid(model(xp).repeat_interleave(n_neg)-model(xn)).mean(); loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  seed {seed} epoch {ep:2d} | member {np.mean(losses):.4f} | valid primary {va['primary']:.4f}")
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; best_state={kk:vv.detach().clone() for kk,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); Xtar,_,_=enc[target]
    return model.predict(Xtar,device=device).astype(np.float32)

def user_zscore(scores, users):
    out=np.empty_like(scores,dtype=np.float32); order=np.argsort(users,kind='mergesort'); su=users[order]; s=0; n=len(users)
    while s<n:
        e=s+1
        while e<n and su[e]==su[s]: e+=1
        idx=order[s:e]; x=scores[idx].astype(np.float32); out[idx]=(x-float(x.mean()))/(float(x.std())+1e-6); s=e
    return out

def get_users_for_target(splits,target):
    umap={}; users=np.empty(len(splits[target]),np.int64)
    for i,r in enumerate(splits[target]):
        u=r[USER]
        if u not in umap: umap[u]=len(umap)
        users[i]=umap[u]
    return users

def add_stat(d, key, y):
    if key in d:
        c,s=d[key]; d[key]=(c+1,s+y)
    else: d[key]=(1,y)

def smooth_logit(d, key, base, alpha):
    c,s=d.get(key,(0,0.0)); p=(s+alpha*base)/(c+alpha); p=min(max(p,1e-4),1-1e-4)
    return math.log(p/(1.0-p))

def safe_logit(p):
    p=min(max(float(p),1e-4),1-1e-4); return math.log(p/(1-p))

def build_stats_and_bonus(train_rows, target_rows):
    t0=time.time(); uc={}; uv={}; ua={}; ut={}; ud={}; gv={}; ga={}; last_uv={}; last_ua={}; global_s=0.0; global_c=0
    for i,r in enumerate(train_rows):
        y=float(r[LABEL]); u=r[USER]; v=r[VIDEO]; a=r[AUTHOR]; tb=r[TAB]; db=dur_bucket(r[DUR])
        global_s += y; global_c += 1
        add_stat(uc,u,y); add_stat(uv,(u,v),y); add_stat(ua,(u,a),y); add_stat(ut,(u,tb),y); add_stat(ud,(u,db),y); add_stat(gv,v,y); add_stat(ga,a,y)
        last_uv[(u,v)]=(i,y); last_ua[(u,a)]=(i,y)
    gp=(global_s+1.0)/(global_c+2.0); gl=safe_logit(gp)
    stat=np.empty(len(target_rows),np.float32); bonus=np.zeros(len(target_rows),np.float32); hit_uv=hit_ua=0
    ntr=max(1,len(train_rows))
    for i,r in enumerate(target_rows):
        u=r[USER]; v=r[VIDEO]; a=r[AUTHOR]; tb=r[TAB]; db=dur_bucket(r[DUR])
        cu,su=uc.get(u,(0,gp)); ub=(su+20*gp)/(cu+20) if cu else gp; base_log=safe_logit(ub)
        sv=smooth_logit(uv,(u,v),ub,3.0)-base_log
        sa=smooth_logit(ua,(u,a),ub,8.0)-base_log
        st=smooth_logit(ut,(u,tb),ub,20.0)-base_log
        sd=smooth_logit(ud,(u,db),ub,20.0)-base_log
        gvscore=smooth_logit(gv,v,gp,50.0)-gl if (u,v) not in uv else 0.0
        gascore=smooth_logit(ga,a,gp,80.0)-gl if (u,a) not in ua else 0.0
        stat[i]=2.2*sv + 1.2*sa + 0.25*st + 0.15*sd + 0.15*gvscore + 0.10*gascore
        b=0.0
        if (u,v) in uv:
            c,s=uv[(u,v)]; hit_uv+=1; mean=(s+1.0)/(c+2.0); b += 1.8*(safe_logit(mean)-base_log)*math.log1p(c)
            li,ly=last_uv[(u,v)]; rec=0.5+0.5*li/ntr; b += rec*(1.1 if ly>0.5 else -0.8)
        if (u,a) in ua:
            c,s=ua[(u,a)]; hit_ua+=1; mean=(s+3.0*ub)/(c+3.0); b += 0.35*(safe_logit(mean)-base_log)*math.log1p(c)
            li,ly=last_ua[(u,a)]; rec=0.5+0.5*li/ntr; b += rec*(0.18 if ly>0.5 else -0.12)
        bonus[i]=b
    print(f'enhanced stats built in {time.time()-t0:.1f}s; uv hits {hit_uv:,d}/{len(target_rows):,d}, ua hits {hit_ua:,d}/{len(target_rows):,d}', flush=True)
    return stat, bonus

def top3_repeat_promote_blend(bpr_z, stat_z, bonus_z, users, w=0.35, thresh=1.0):
    out=np.empty_like(bpr_z,dtype=np.float32); order=np.argsort(users,kind='mergesort'); su=users[order]; s=0; n=len(users)
    while s<n:
        e=s+1
        while e<n and su[e]==su[s]: e+=1
        idx=order[s:e]; bz=bpr_z[idx].astype(np.float32); sz=stat_z[idx].astype(np.float32); bon=bonus_z[idx].astype(np.float32)
        rord=np.argsort(-bz,kind='mergesort'); final=(bz + w*sz).astype(np.float32)
        k3=min(3,len(idx))
        if k3>0:
            top=rord[:k3]; final[top]=1000.0-np.arange(k3,dtype=np.float32)
        # Fill ranks 4-5 with strong exact-repeat/memory candidates from a wider BPR shortlist.
        start=k3; cend=min(60,len(idx)); promoted=[]
        if cend>start:
            cand=rord[start:cend]
            good=cand[bon[cand] > thresh]
            if len(good):
                # exact evidence dominates, but retain a small BPR rank prior for ties/noisy author memory
                rel_rank=np.empty(len(idx),dtype=np.float32); rel_rank[rord]=np.arange(len(idx),dtype=np.float32)
                score=bon[good] - 0.015*rel_rank[good]
                promoted=list(good[np.argsort(-score,kind='mergesort')[:2]])
        rank=0
        for p in promoted:
            final[p]=996.0-rank; rank+=1
        # protect original high BPR candidates not displaced so no weak memory row can enter top5
        fill_needed=max(0, min(5,len(idx))-k3-len(promoted))
        if fill_needed>0:
            for p in rord[k3:]:
                if p in promoted: continue
                final[p]=994.0-rank; rank+=1
                fill_needed-=1
                if fill_needed<=0: break
        out[idx]=final; s=e
    return user_zscore(out,users)

def get_preds(splits,data_dir,target,split_name,base_seed,k=16,lr=0.001,epochs=40,device='cpu',verbose=False):
    os.makedirs('pred_cache',exist_ok=True); member_seeds=[base_seed,base_seed+100,base_seed+200,base_seed+300]
    paths=[os.path.join('pred_cache',f'025_v19_seedens_{split_name}_member_seed{ms}.npy') for ms in member_seeds]
    preds=[np.load(p).astype(np.float32) if os.path.isfile(p) else None for p in paths]
    if any(p is None for p in preds):
        enc,dim=prepare(splits,data_dir,target)
        for j,ms in enumerate(member_seeds):
            if preds[j] is None:
                preds[j]=train_predict_member(enc,dim,target,ms,k=k,lr=lr,epochs=epochs,device=device,verbose=verbose); np.save(paths[j],preds[j])
        users=enc[target][2]
    else:
        users=get_users_for_target(splits,target)
    bpr_z=sum(user_zscore(p,users) for p in preds)/float(len(preds))
    stat_path=os.path.join('pred_cache',f'036_enh_userstats_{split_name}.npy'); bon_path=os.path.join('pred_cache',f'036_exact_bonus_{split_name}.npy')
    if os.path.isfile(stat_path) and os.path.isfile(bon_path):
        stat=np.load(stat_path).astype(np.float32); bonus=np.load(bon_path).astype(np.float32)
    else:
        stat,bonus=build_stats_and_bonus(splits['train'],splits[target]); np.save(stat_path,stat); np.save(bon_path,bonus)
    return top3_repeat_promote_blend(bpr_z,user_zscore(stat,users),user_zscore(bonus,users),users,w=0.35,thresh=1.0).astype(np.float32)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None)
    ap.add_argument('--k',type=int,default=16); ap.add_argument('--lr',type=float,default=0.001); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=ap.parse_args(); torch.manual_seed(a.seed); print(f'loading {a.data_dir} ...',flush=True)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}', flush=True)
    scores=get_preds(splits,a.data_dir,target,a.split,a.seed,k=a.k,lr=a.lr,epochs=a.epochs,device=a.device,verbose=a.out is None)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else: print(scores[:10])
