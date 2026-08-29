"""Blend node-10 cached FM ensemble with a new content+history+time FM.

The new member reads raw log_standard CSVs only to recover hourmin, then appends
hour/minute/weekday context to the content-history FM fields.  Existing node-10
members keep their original cache keys and feature spaces.
"""
import argparse, csv, os, sys, time
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS
from evaluate import evaluate

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
        for i in range(0,len(X),bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def pos_bucket(c):
    if c<=0: return 0
    if c==1: return 1
    if c==2: return 2
    if c<=5: return 3
    return 4

def neg_bucket(c):
    if c<=0: return 0
    if c<=2: return 1
    if c<=5: return 2
    if c<=10: return 3
    return 4

def ratio_bucket(p,n):
    t=p+n
    if t<=0: return 0
    r=p/t
    if r<0.20: return 1
    if r<0.40: return 2
    if r<0.60: return 3
    if r<0.80: return 4
    return 5

def size_bucket(x):
    try: v=int(float(x))
    except Exception: return 'unk'
    if v<=0: return 'unk'
    if v<=360: return '<=360'
    if v<=540: return '<=540'
    if v<=720: return '<=720'
    if v<=1080: return '<=1080'
    return '>1080'

def norm_val(v): return 'unk' if v is None or v=='' else str(v)

def first_tag(v):
    s=norm_val(v)
    if s=='unk': return s
    for sep in [';',',','|',' ']:
        if sep in s: return s.split(sep)[0]
    return s

def read_video_features(data_dir):
    path=os.path.join(data_dir,'video_features_basic_pure.csv'); feats={}
    if not os.path.isfile(path): return feats
    with open(path,newline='',encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            vid=rec.get('video_id') or rec.get('video') or rec.get('photo_id')
            if vid is None or vid=='': continue
            try: vid=int(vid)
            except Exception: pass
            feats[vid]=[norm_val(rec.get('music_id')), first_tag(rec.get('tag')), norm_val(rec.get('video_type')),
                        norm_val(rec.get('upload_type')), size_bucket(rec.get('server_width')),
                        size_bucket(rec.get('server_height')), norm_val(rec.get('music_type'))]
    return feats

def _ival(rec, names, default=0):
    for n in names:
        if n in rec and rec[n] not in (None,''):
            try: return int(float(rec[n]))
            except Exception: pass
    return default

def read_hour_queues(data_dir):
    qs=defaultdict(deque)
    files=['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']
    for fn in files:
        path=os.path.join(data_dir,fn)
        if not os.path.isfile(path): continue
        with open(path,newline='',encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                date=_ival(rec,['date']); u=_ival(rec,['user_id','user']); v=_ival(rec,['video_id','photo_id','video'])
                a=_ival(rec,['author_id']); tab=_ival(rec,['tab']); dur=_ival(rec,['duration_ms','duration'])
                hm=_ival(rec,['hourmin','hour_min','time','timestamp'],0)
                key=(date,u,v,a,tab,dur); qs[key].append(hm)
    return qs

def weekday_from_date(d):
    try: return datetime.strptime(str(int(d)),'%Y%m%d').weekday()
    except Exception: return 0

def time_feats_from_hm(hm, date):
    try: hm=int(hm)
    except Exception: hm=0
    hour=hm//100 if hm>=100 else hm
    minute=hm%100 if hm>=100 else 0
    if hour<0 or hour>23: hour=0
    mb=min(5, max(0, minute//10))
    # Coarse dayparts give the FM a less sparse temporal signal than raw hour alone.
    if hour<6: part=0
    elif hour<11: part=1
    elif hour<14: part=2
    elif hour<18: part=3
    elif hour<22: part=4
    else: part=5
    return [hour, mb, part, weekday_from_date(date)]

def build_history_raw_features(splits, video_feats=None, add_content=False, hour_qs=None, add_time=False):
    up_v=defaultdict(int); un_v=defaultdict(int); up_a=defaultdict(int); un_a=defaultdict(int)
    raw={}; labels={}; users={}; video_feats=video_feats or {}; unk=['unk']*7; missing=0
    def row_feats(row, update=False):
        nonlocal missing
        date,u,v,a,tab,dur,y=row; dur_bucket=int(dur)//10000; kv=(u,v); ka=(u,a)
        pv,nv=up_v[kv],un_v[kv]; pa,na=up_a[ka],un_a[ka]
        feats=[u,v,a,tab,dur_bucket,pos_bucket(pv),neg_bucket(nv),pos_bucket(pa),neg_bucket(na),ratio_bucket(pa,na),ratio_bucket(pv,nv)]
        if add_content: feats.extend(video_feats.get(v,unk))
        if add_time:
            hm=0; key=(int(date),int(u),int(v),int(a),int(tab),int(dur))
            if hour_qs is not None and key in hour_qs and len(hour_qs[key]): hm=hour_qs[key].popleft()
            else: missing+=1
            feats.extend(time_feats_from_hm(hm,date)); feats.append(f'{tab}_{time_feats_from_hm(hm,date)[2]}')
        if update:
            if y>0.5: up_v[kv]+=1; up_a[ka]+=1
            else: un_v[kv]+=1; un_a[ka]+=1
        return feats
    for sp in ('train','valid','test'):
        fs=[]; ys=[]; us=[]
        for row in splits[sp]:
            fs.append(row_feats(row, update=(sp=='train'))); ys.append(row[6]); us.append(row[1])
        raw[sp]=fs; labels[sp]=np.asarray(ys,dtype=np.float32); users[sp]=np.asarray(us,dtype=np.int64)
    return raw,labels,users

def encode_history(splits, video_feats=None, add_content=False, hour_qs=None, add_time=False):
    raw,labels,users=build_history_raw_features(splits,video_feats,add_content,hour_qs,add_time); n=len(raw['train'][0]); maps=[]; off=0
    for j in range(n):
        vals=set()
        for sp in ('train','valid','test'): vals.update(r[j] for r in raw[sp])
        mp={v:off+i for i,v in enumerate(sorted(vals,key=lambda x:str(x)))}; maps.append(mp); off+=len(mp)
    enc={}
    for sp in ('train','valid','test'):
        X=np.empty((len(raw[sp]),n),dtype=np.int64)
        for i,r in enumerate(raw[sp]):
            for j,v in enumerate(r): X[i,j]=maps[j][v]
        enc[sp]=(X,labels[sp],users[sp])
    return enc,off

def make_bpr_training_arrays(ytr,utr):
    y=np.asarray(ytr); u=np.asarray(utr); order=np.argsort(u,kind='stable'); us=u[order]
    cuts=np.r_[0,np.flatnonzero(us[1:]!=us[:-1])+1,len(order)]; neg_by_user={}; ep=[]; eu=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        rows=order[a:b]; yy=y[rows]; pos=rows[yy>0.5]; neg=rows[yy<=0.5]
        if len(pos) and len(neg):
            uid=int(u[rows[0]]); neg_by_user[uid]=neg.astype(np.int64); ep.append(pos.astype(np.int64)); eu.append(np.full(len(pos),uid,dtype=np.int64))
    if not ep: raise RuntimeError('no users with both positive and negative rows')
    return np.concatenate(ep),np.concatenate(eu),neg_by_user

def train_bce(enc,dim,k=16,lr=0.001,l2=1e-6,epochs=40,bs=8192,patience=4,seed=0,device='cpu',verbose=False):
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr); lossfn=torch.nn.BCEWithLogitsLoss()
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0
    for ep in range(1,epochs+1):
        idx=rng.permutation(len(ytr)); model.train(); t0=time.time()
        for i in range(0,len(idx),bs):
            sel=torch.from_numpy(idx[i:i+bs]); xb=Xtr_t[sel].to(device); yb=ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  BCE seed {seed} epoch {ep:2d} valid {va['primary']:.6f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5: best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def train_bpr(enc,dim,k=16,lr=0.001,l2=1e-6,epochs=40,bs=8192,patience=4,seed=0,device='cpu',verbose=False):
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); pos_rows,pos_users,neg_by_user=make_bpr_training_arrays(ytr,utr); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0
    for ep in range(1,epochs+1):
        perm=rng.permutation(len(pos_rows)); model.train(); t0=time.time()
        for st in range(0,len(perm),bs):
            psel=perm[st:st+bs]; pidx_np=pos_rows[psel]; users_np=pos_users[psel]; nidx_np=np.empty(len(users_np),dtype=np.int64)
            for j,u in enumerate(users_np):
                negs=neg_by_user[int(u)]; nidx_np[j]=negs[rng.integers(len(negs))]
            xp=Xtr_t[torch.from_numpy(pidx_np)].to(device); xn=Xtr_t[torch.from_numpy(nidx_np)].to(device)
            opt.zero_grad(set_to_none=True); loss=torch.nn.functional.softplus(-(model(xp)-model(xn))).mean(); loss.backward(); opt.step()
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose: print(f"  BPR seed {seed} epoch {ep:2d} valid {va['primary']:.6f} {time.time()-t0:.1f}s")
        if va['primary']>best+1e-5: best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state); return model

def within_user_ranks(scores,users):
    scores=np.asarray(scores); users=np.asarray(users); out=np.empty(len(scores),dtype=np.float64); order=np.argsort(users,kind='stable'); us=users[order]
    cuts=np.r_[0,np.flatnonzero(us[1:]!=us[:-1])+1,len(order)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        idx=order[a:b]; n=len(idx)
        if n==1: out[idx]=0.5
        else:
            ord2=np.argsort(scores[idx],kind='mergesort'); r=np.empty(n,dtype=np.float64); r[ord2]=np.arange(n,dtype=np.float64)/(n-1.0); out[idx]=r
    return out

def member_pred(cache_prefix,name,train_fn,enc,dim,Xout,split,outer_seed,member_seed,device,verbose):
    os.makedirs('pred_cache',exist_ok=True); path=os.path.join('pred_cache',f'{cache_prefix}_{name}_split{split}_outer{outer_seed}_mseed{member_seed}.npy')
    if os.path.isfile(path): return np.load(path)
    torch.manual_seed(member_seed); model=train_fn(enc,dim,seed=member_seed,device=device,verbose=verbose)
    preds=model.predict(Xout,device=device).astype(np.float64); np.save(path,preds); return preds

def ensemble_rank(enc,dim,Xout,uout,split,seed,device,verbose,cache_prefix,bpr_name,bce_name):
    bpr_sum=np.zeros(len(Xout),dtype=np.float64)
    for ms in [seed,seed+101,seed+202]: bpr_sum+=within_user_ranks(member_pred(cache_prefix,bpr_name,train_bpr,enc,dim,Xout,split,seed,ms,device,verbose),uout)
    bpr_rank=bpr_sum/3.0; bce_rank=within_user_ranks(member_pred(cache_prefix,bce_name,train_bce,enc,dim,Xout,split,seed,seed+303,device,verbose),uout)
    return 0.70*bpr_rank+0.30*bce_rank

def run_predict(splits,data_dir,split='valid',seed=0,device='cpu',verbose=False):
    enc_base,dim_base=encode(splits); Xbase,_,uout=enc_base[split]
    base_rank=ensemble_rank(enc_base,dim_base,Xbase,uout,split,seed,device,verbose,'006','bpr','bce')
    enc_hist,dim_hist=encode_history(splits,add_content=False); Xhist,_,uhist=enc_hist[split]
    hist_rank=ensemble_rank(enc_hist,dim_hist,Xhist,uhist,split,seed,device,verbose,'008','bprhist','bcehist')
    vf=read_video_features(data_dir)
    enc_cont,dim_cont=encode_history(splits,video_feats=vf,add_content=True); Xcont,_,ucont=enc_cont[split]
    cont_rank=ensemble_rank(enc_cont,dim_cont,Xcont,ucont,split,seed,device,verbose,'010','bprcont','bcecont')
    node10=0.35*base_rank+0.25*hist_rank+0.40*cont_rank
    hour_qs=read_hour_queues(data_dir)
    enc_time,dim_time=encode_history(splits,video_feats=vf,add_content=True,hour_qs=hour_qs,add_time=True); Xtime,_,utime=enc_time[split]
    time_rank=ensemble_rank(enc_time,dim_time,Xtime,utime,split,seed,device,verbose,'013','bprtime','bcetime')
    return 0.60*node10+0.40*time_rank

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test']); ap.add_argument('--out',default=None); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    print(f'loading {a.data_dir} ...'); splits=load(a.data_dir); print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}+history+content+hourmin')
    scores=run_predict(splits,a.data_dir,split=a.split,seed=a.seed,device=a.device,verbose=a.out is None)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else: print(scores[:10])
