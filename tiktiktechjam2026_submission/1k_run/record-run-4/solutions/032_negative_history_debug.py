"""Debug node 31: use the explicit history-affinity scorer as a possible anti-signal.

Node 31 diluted the node-30 ensemble with a positive handcrafted repeat/author
history member and lost mostly nDCG@5.  This keeps the node-30 fusion unchanged
and adds the same cached/computable history scorer only as a negative per-user
z-scored residual, testing whether the heuristic is systematically over-ranking
bad top-5 candidates rather than merely weak.
"""
import argparse, os, sys, math
from collections import defaultdict
import numpy as np
import lightgbm as lgb
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS


def duration_array(rows):
    return np.asarray([float(r[5]) for r in rows], dtype=np.float32)


def factorize_cols(Xtr, Xva, Xt, target_is_valid):
    Xtr=Xtr.astype(np.int64,copy=False); Xva=Xva.astype(np.int64,copy=False); Xt=Xt.astype(np.int64,copy=False)
    trs=[]; vas=[]; tes=[]
    for j in range(Xtr.shape[1]):
        if target_is_valid:
            vals=np.concatenate([Xtr[:,j],Xva[:,j]]); inv=np.unique(vals,return_inverse=True)[1]; n=len(Xtr)
            trs.append(inv[:n]); vas.append(inv[n:]); tes.append(inv[n:])
        else:
            vals=np.concatenate([Xtr[:,j],Xva[:,j],Xt[:,j]]); inv=np.unique(vals,return_inverse=True)[1]; n=len(Xtr); m=len(Xva)
            trs.append(inv[:n]); vas.append(inv[n:n+m]); tes.append(inv[n+m:])
    return np.column_stack(trs).astype(np.int32),np.column_stack(vas).astype(np.int32),np.column_stack(tes).astype(np.int32)


def train_minimal(splits,target,split_name,seed=0):
    os.makedirs('pred_cache',exist_ok=True)
    enc,_=encode(splits); Xtr0,ytr,_=enc['train']; Xva0,yva,_=enc['valid']; Xt0,_,_=enc[target]
    Xtr,Xva,Xt=factorize_cols(Xtr0,Xva0,Xt0,target=='valid')
    dtr=duration_array(splits['train']); dva=duration_array(splits['valid']); dt=dva if target=='valid' else duration_array(splits[target])
    Ftr=np.column_stack([Xtr.astype(np.float32),np.log1p(np.maximum(dtr,0))[:,None],(dtr/100000.0)[:,None]]).astype(np.float32)
    Fva=np.column_stack([Xva.astype(np.float32),np.log1p(np.maximum(dva,0))[:,None],(dva/100000.0)[:,None]]).astype(np.float32)
    Ft=np.column_stack([Xt.astype(np.float32),np.log1p(np.maximum(dt,0))[:,None],(dt/100000.0)[:,None]]).astype(np.float32)
    clf=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',boosting_type='gbdt',n_estimators=220,learning_rate=0.05,num_leaves=63,min_child_samples=500,subsample=0.85,subsample_freq=1,colsample_bytree=0.9,reg_alpha=0.1,reg_lambda=2.0,random_state=int(seed),n_jobs=-1,verbose=-1,force_col_wise=True)
    clf.fit(Ftr,ytr,eval_set=[(Fva,yva)],categorical_feature=list(range(5)),callbacks=[lgb.early_stopping(20,first_metric_only=True,verbose=False)])
    p=clf.predict(Ft,num_iteration=clf.best_iteration_,raw_score=True).astype(np.float32)
    np.save(f'pred_cache/032_fallback_minimal_split{split_name}_seed{seed}.npy',p)
    return p


def load_first(names,n):
    for name in names:
        path=os.path.join('pred_cache',name)
        if os.path.isfile(path):
            a=np.load(path)
            if len(a)==n: return a.astype(np.float32,copy=False)
    return None


def z_per_user(arr, users):
    users=np.asarray(users); arr=np.asarray(arr,dtype=np.float32)
    out=np.zeros(len(users),dtype=np.float32)
    order=np.argsort(users,kind='mergesort'); us=users[order]
    starts=np.r_[0,np.flatnonzero(us[1:]!=us[:-1])+1,len(us)]
    for a,b in zip(starts[:-1],starts[1:]):
        idx=order[a:b]; v=arr[idx]
        out[idx]=(v-v.mean())/max(float(v.std()),1e-6)
    return out


def z_per_user_blend(arrs,weights,users):
    users=np.asarray(users); arrs=[np.asarray(a,dtype=np.float32) for a in arrs]
    w=np.asarray(weights,dtype=np.float32); w=w/w.sum()
    out=np.zeros(len(users),dtype=np.float32)
    order=np.argsort(users,kind='mergesort'); us=users[order]
    starts=np.r_[0,np.flatnonzero(us[1:]!=us[:-1])+1,len(us)]
    for a,b in zip(starts[:-1],starts[1:]):
        idx=order[a:b]; s=np.zeros(b-a,dtype=np.float32)
        for ww,x in zip(w,arrs):
            v=x[idx]; s += ww*((v-v.mean())/max(float(v.std()),1e-6))
        out[idx]=s
    return out


def rrf_per_user(arrs,weights,users,k=60.0):
    users=np.asarray(users); w=np.asarray(weights,dtype=np.float32); w=w/w.sum()
    out=np.zeros(len(users),dtype=np.float32)
    order=np.argsort(users,kind='mergesort'); us=users[order]
    starts=np.r_[0,np.flatnonzero(us[1:]!=us[:-1])+1,len(us)]
    for a,b in zip(starts[:-1],starts[1:]):
        idx=order[a:b]; m=b-a; s=np.zeros(m,dtype=np.float32)
        for ww,x in zip(w,arrs):
            vals=x[idx]
            ord2=np.argsort(-vals,kind='mergesort')
            ranks=np.empty(m,dtype=np.float32); ranks[ord2]=np.arange(1,m+1,dtype=np.float32)
            s += ww/(k+ranks)
        out[idx]=s
    return out


def node24(exact0,exact2,fast,users):
    zexact=z_per_user_blend([exact0,exact2],[0.5,0.5],users)
    return z_per_user_blend([zexact,fast],[0.77,0.23],users)


def node30_score(exact0,exact2,fast,stat,seq,aux,users):
    inc=node24(exact0,exact2,fast,users)
    zmain=z_per_user_blend([inc,stat,seq,aux],[0.38,0.10,0.22,0.30],users)
    rrf=rrf_per_user([inc,stat,seq,aux],[0.38,0.10,0.22,0.30],users,k=60.0)
    return z_per_user_blend([zmain,rrf],[0.70,0.30],users)


def dur_bucket_ms(x):
    try: v=float(x)
    except Exception: v=0.0
    if v < 5000: return 0
    if v < 10000: return 1
    if v < 30000: return 2
    if v < 60000: return 3
    if v < 120000: return 4
    return 5


def add_stat(d,key,pos,w):
    a=d.get(key)
    if a is None: d[key]=[w*pos,w]
    else: a[0]+=w*pos; a[1]+=w


def history_affinity(train_rows, target_rows, cache_name):
    os.makedirs('pred_cache',exist_ok=True)
    path=os.path.join('pred_cache',cache_name)
    if os.path.isfile(path):
        p=np.load(path)
        if len(p)==len(target_rows): return p.astype(np.float32,copy=False)
    max_date=max(int(r[0]) for r in train_rows) if train_rows else 0
    uv=defaultdict(lambda:[0.0,0.0]); ua=defaultdict(lambda:[0.0,0.0]); ut=defaultdict(lambda:[0.0,0.0]); ud=defaultdict(lambda:[0.0,0.0])
    uglob=defaultdict(lambda:[0.0,0.0]); vglob=defaultdict(lambda:[0.0,0.0]); aglob=defaultdict(lambda:[0.0,0.0]); tglob=defaultdict(lambda:[0.0,0.0])
    gpos=0.0; gcnt=0.0
    for r in train_rows:
        date,u,v,a,tab,dur,y=r[0],r[1],r[2],r[3],r[4],r[5],float(r[6])
        delta=max(0, max_date-int(date)); w=0.60 + 0.40*math.exp(-delta/6.0)
        db=dur_bucket_ms(dur)
        add_stat(uv,(u,v),y,w); add_stat(ua,(u,a),y,w); add_stat(ut,(u,tab),y,w); add_stat(ud,(u,db),y,w)
        add_stat(uglob,u,y,w); add_stat(vglob,v,y,w); add_stat(aglob,a,y,w); add_stat(tglob,tab,y,w)
        gpos += w*y; gcnt += w
    prior=(gpos+1.0)/(gcnt+2.0)
    def ctr(dic,key,alpha):
        s,c=dic.get(key,(0.0,0.0)); return (s+alpha*prior)/(c+alpha)
    def cnt(dic,key):
        s,c=dic.get(key,(0.0,0.0)); return math.log1p(c)
    out=np.empty(len(target_rows),dtype=np.float32)
    for i,r in enumerate(target_rows):
        u,v,a,tab,dur=r[1],r[2],r[3],r[4],r[5]; db=dur_bucket_ms(dur)
        out[i]=(2.20*ctr(uv,(u,v),4.0)+0.22*cnt(uv,(u,v))+
                1.25*ctr(ua,(u,a),8.0)+0.10*cnt(ua,(u,a))+
                0.70*ctr(ut,(u,tab),12.0)+0.45*ctr(ud,(u,db),12.0)+
                0.35*ctr(uglob,u,20.0)+0.30*ctr(vglob,v,25.0)+0.25*ctr(aglob,a,30.0)+0.20*ctr(tglob,tab,30.0))
    np.save(path,out)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu')
    a=ap.parse_args(); np.random.seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}')
    enc,_=encode(splits); users=np.asarray(enc[target][2]); n=len(users); sp=a.split
    exact0=load_first([f'{p}_node11_raw_split{sp}_mseed0.npy' for p in ['018','019','020','021','022','023','024','025','026','028','029']],n)
    exact2=load_first([f'{p}_node11_raw_split{sp}_mseed2.npy' for p in ['018','019','020','021','022','023','024','025','026','028','029']],n)
    fast=load_first([f'{p}_fast_lgbm_split{sp}_mseed4.npy' for p in ['022','023','024','025','026','028','029']],n)
    stat=load_first([f'{p}_oof_target_stats_split{sp}_mseed6.npy' for p in ['025','028','029']],n)
    seq=load_first([f'{p}_user_history_seq_split{sp}_mseed8.npy' for p in ['026','028','029']],n)
    aux=load_first([f'029_aux_soft_seq_split{sp}_mseed10.npy'],n)
    fallback=None
    def fb(seed=0):
        global fallback
        if fallback is None: fallback=train_minimal(splits,target,sp,seed)
        return fallback
    if exact0 is None: exact0=fb(0)
    if exact2 is None: exact2=fb(2)
    if fast is None: fast=fb(4)
    if stat is None: stat=fb(6)
    if seq is None: seq=fb(8)
    if aux is None: aux=fb(10)
    base=node30_score(exact0,exact2,fast,stat,seq,aux,users)
    hist_rows=list(splits['train']) if target=='valid' else list(splits['train'])+list(splits['valid'])
    hist=history_affinity(hist_rows,splits[target],f'031_history_affinity_split{sp}.npy')
    # Negative residual at a readable but not overwhelming magnitude.  Per-user
    # standardization preserves within-user ranking scale and makes subtraction meaningful.
    scores=(z_per_user(base,users) - 0.22*z_per_user(hist,users)).astype(np.float64)
    if a.out:
        np.save(a.out,scores); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores.shape,float(scores.mean()))
