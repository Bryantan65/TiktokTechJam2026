"""Node-24 ensemble plus a target-conditioned user-history member.

This is a cheap sequence/DIN-inspired variant: instead of a neural attention
network, expose LightGBM to chronological same-user history features for the
candidate video/author/tab.  Training rows use only earlier train rows; valid
and test rows use statistics accumulated from the whole training window.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa: E402


def duration_array(rows):
    return np.asarray([float(r[5]) for r in rows], dtype=np.float32)


def factorize_cols(Xtr, Xva, Xt, target_is_valid):
    Xtr = Xtr.astype(np.int64, copy=False); Xva = Xva.astype(np.int64, copy=False); Xt = Xt.astype(np.int64, copy=False)
    trs=[]; vas=[]; tes=[]
    for j in range(Xtr.shape[1]):
        if target_is_valid:
            allv = np.concatenate([Xtr[:, j], Xva[:, j]])
            inv = np.unique(allv, return_inverse=True)[1]
            ntr=len(Xtr); trs.append(inv[:ntr]); vas.append(inv[ntr:]); tes.append(inv[ntr:])
        else:
            allv = np.concatenate([Xtr[:, j], Xva[:, j], Xt[:, j]])
            inv = np.unique(allv, return_inverse=True)[1]
            ntr=len(Xtr); nva=len(Xva)
            trs.append(inv[:ntr]); vas.append(inv[ntr:ntr+nva]); tes.append(inv[ntr+nva:])
    return (np.column_stack(trs).astype(np.int32), np.column_stack(vas).astype(np.int32), np.column_stack(tes).astype(np.int32))


def minimal_mats(splits, target):
    enc, _ = encode(splits)
    Xtr0,ytr,_ = enc['train']; Xva0,yva,_ = enc['valid']; Xt0,_,_ = enc[target]
    Xtr,Xva,Xt = factorize_cols(Xtr0, Xva0, Xt0, target == 'valid')
    dtr=duration_array(splits['train']); dva=duration_array(splits['valid']); dt=dva if target=='valid' else duration_array(splits[target])
    def f(X,d):
        return np.column_stack([X.astype(np.float32), np.log1p(np.maximum(d,0))[:,None], (d/100000.0)[:,None]]).astype(np.float32)
    return f(Xtr,dtr), ytr.astype(np.int32), f(Xva,dva), yva.astype(np.int32), f(Xt,dt), Xtr, Xva, Xt


def fit_lgbm(Ftr,ytr,Fva,yva,Ft, seed, fast=False, verbose=False):
    if fast:
        params = dict(n_estimators=180, learning_rate=0.08, num_leaves=31, min_child_samples=300,
                      subsample=0.80, colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=1.5)
        stop=20
    else:
        params = dict(n_estimators=350, learning_rate=0.05, num_leaves=63, min_child_samples=500,
                      subsample=0.85, colsample_bytree=0.90, reg_alpha=0.1, reg_lambda=2.0)
        stop=30
    clf = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', max_depth=-1,
                             subsample_freq=1, random_state=int(seed), n_jobs=-1, verbose=-1,
                             force_col_wise=True, **params)
    cb=[lgb.early_stopping(stop, first_metric_only=True, verbose=verbose)]
    if verbose: cb.append(lgb.log_evaluation(50))
    clf.fit(Ftr, ytr, eval_set=[(Fva,yva)], categorical_feature=list(range(5)), callbacks=cb)
    return clf.predict(Ft, num_iteration=clf.best_iteration_, raw_score=True).astype(np.float32)


def get_node24_members(splits, target, split_name, verbose=False):
    os.makedirs('pred_cache', exist_ok=True)
    n=len(splits[target])
    exact=[]; need=[]
    for ms in [0,2]:
        found=None
        for p in ['018','019','020','021','022','023','024','025','026']:
            path=f'pred_cache/{p}_node11_raw_split{split_name}_mseed{ms}.npy'
            if os.path.isfile(path):
                arr=np.load(path)
                if len(arr)==n: found=arr.astype(np.float32,copy=False); break
        if found is None: need.append(ms); exact.append(None)
        else: exact.append(found)
    fast=None
    for p in ['022','023','024','025','026']:
        path=f'pred_cache/{p}_fast_lgbm_split{split_name}_mseed4.npy'
        if os.path.isfile(path):
            arr=np.load(path)
            if len(arr)==n: fast=arr.astype(np.float32,copy=False); break
    if need or fast is None:
        Ftr,ytr,Fva,yva,Ft,_,_,_=minimal_mats(splits,target)
        for k,ms in enumerate([0,2]):
            if exact[k] is None:
                arr=fit_lgbm(Ftr,ytr,Fva,yva,Ft,ms,False,verbose); np.save(f'pred_cache/026_node11_raw_split{split_name}_mseed{ms}.npy',arr); exact[k]=arr
        if fast is None:
            fast=fit_lgbm(Ftr,ytr,Fva,yva,Ft,4,True,verbose); np.save(f'pred_cache/026_fast_lgbm_split{split_name}_mseed4.npy',fast)
    return exact, fast


def pair_key(a,b,base):
    return int(a)*base + int(b)


def history_features_train_query(Xtr, ytr, Xq):
    ntr=len(Xtr); nq=len(Xq); prior=float(np.mean(ytr)); alpha=20.0
    nvid=int(max(Xtr[:,1].max(), Xq[:,1].max()))+1
    nauth=int(max(Xtr[:,2].max(), Xq[:,2].max()))+1
    ntab=int(max(Xtr[:,3].max(), Xq[:,3].max()))+1
    ucnt=defaultdict(int); upos=defaultdict(int)
    cnt_uv=defaultdict(int); pos_uv=defaultdict(int); last_uv={}; lpos_uv={}
    cnt_ua=defaultdict(int); pos_ua=defaultdict(int); last_ua={}; lpos_ua={}
    cnt_ut=defaultdict(int); pos_ut=defaultdict(int); last_ut={}; lpos_ut={}
    Ftr=np.zeros((ntr,14),dtype=np.float32)
    def fill(row, i, out, update, yy=0):
        u=int(row[0]); v=int(row[1]); a=int(row[2]); t=int(row[3])
        keys=[pair_key(u,v,nvid), pair_key(u,a,nauth), pair_key(u,t,ntab)]
        dicts=[(cnt_uv,pos_uv,last_uv,lpos_uv),(cnt_ua,pos_ua,last_ua,lpos_ua),(cnt_ut,pos_ut,last_ut,lpos_ut)]
        uc=ucnt[u]; up=upos[u]
        out[0]=np.log1p(uc); out[1]=(up+prior*alpha)/(uc+alpha)
        off=2
        for key,(cd,pd,ld,lpd) in zip(keys,dicts):
            c=cd[key]; p=pd[key]
            out[off]=np.log1p(c); out[off+1]=(p+prior*alpha)/(c+alpha)
            out[off+2]=0.0 if key not in ld else np.log1p(i-ld[key])
            out[off+3]=0.0 if key not in lpd else np.log1p(i-lpd[key])
            off += 4
        if update:
            ucnt[u]=uc+1; upos[u]=up+int(yy)
            for key,(cd,pd,ld,lpd) in zip(keys,dicts):
                cd[key]+=1; pd[key]+=int(yy); ld[key]=i
                if yy: lpd[key]=i
    for i in range(ntr):
        fill(Xtr[i], i, Ftr[i], True, int(ytr[i]))
    Fq=np.zeros((nq,14),dtype=np.float32)
    base_i=ntr
    for j in range(nq):
        fill(Xq[j], base_i+j, Fq[j], False, 0)
    return Ftr, Fq


def sequence_member(splits, target, split_name, verbose=False):
    os.makedirs('pred_cache', exist_ok=True)
    path=f'pred_cache/026_user_history_seq_split{split_name}_mseed8.npy'
    n=len(splits[target])
    if os.path.isfile(path):
        arr=np.load(path)
        if len(arr)==n: return arr.astype(np.float32,copy=False)
    t0=time.time()
    Ftr0,ytr,Fva0,yva,Ft0,Xtr,Xva,Xt = minimal_mats(splits,target)
    Htr,Hva = history_features_train_query(Xtr,ytr,Xva)
    _,Ht = history_features_train_query(Xtr,ytr,Xt) if target!='valid' else (None,Hva)
    Ftr=np.column_stack([Ftr0,Htr]).astype(np.float32); Fva=np.column_stack([Fva0,Hva]).astype(np.float32); Ft=np.column_stack([Ft0,Ht]).astype(np.float32)
    if verbose: print('seq mats',Ftr.shape,Fva.shape,Ft.shape,'built',time.time()-t0)
    clf=lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', n_estimators=260,
                           learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=600,
                           subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.2,
                           reg_lambda=3.0, random_state=8, n_jobs=-1, verbose=-1, force_col_wise=True)
    cb=[lgb.early_stopping(25, first_metric_only=True, verbose=verbose)]
    if verbose: cb.append(lgb.log_evaluation(50))
    clf.fit(Ftr,ytr,eval_set=[(Fva,yva)],categorical_feature=list(range(5)),callbacks=cb)
    arr=clf.predict(Ft,num_iteration=clf.best_iteration_,raw_score=True).astype(np.float32)
    np.save(path,arr); return arr


def z_per_user_blend(arrs, weights, users):
    users=np.asarray(users); arrs=[np.asarray(a,dtype=np.float32) for a in arrs]
    out=np.zeros(len(users),dtype=np.float32); order=np.argsort(users,kind='mergesort'); us=users[order]
    starts=np.r_[0,np.flatnonzero(us[1:]!=us[:-1])+1,len(us)]
    w=np.asarray(weights,dtype=np.float32); w=w/w.sum()
    for a,b in zip(starts[:-1],starts[1:]):
        idx=order[a:b]; s=np.zeros(b-a,dtype=np.float32)
        for ww,x in zip(w,arrs):
            v=x[idx]; s += ww*((v-v.mean())/max(float(v.std()),1e-6))
        out[idx]=s
    return out.astype(np.float64)


def node24(exact,fast,users):
    zexact=z_per_user_blend(exact,[0.5,0.5],users)
    return z_per_user_blend([zexact,fast],[0.77,0.23],users).astype(np.float32)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu')
    a=ap.parse_args(); np.random.seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}')
    enc,_=encode(splits); users=enc[target][2]
    exact,fast=get_node24_members(splits,target,a.split,verbose=a.out is None)
    seq=sequence_member(splits,target,a.split,verbose=a.out is None)
    inc=node24(exact,fast,users)
    scores=z_per_user_blend([inc,seq],[0.70,0.30],users)
    if a.out:
        np.save(a.out,scores); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores.shape, float(scores.mean()))
