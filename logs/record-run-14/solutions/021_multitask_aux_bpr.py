import argparse, csv, glob, os, sys, time
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate  # noqa

AUX_COLS = ['is_click','is_like','is_follow','is_comment','is_forward','is_profile_enter','is_hate']

def _s(x):
    if x is None: return ''
    if isinstance(x, bytes): x = x.decode('utf8')
    try:
        if isinstance(x, (float, np.floating)) and np.isfinite(x) and abs(x-int(x)) < 1e-6:
            return str(int(x))
    except Exception:
        pass
    return str(x)

def row_key(r):
    return (_s(r[0]), _s(r[1]), _s(r[2]), _s(r[3]), _s(r[4]), _s(r[5]))

def csv_key(rec):
    def g(*names):
        for n in names:
            if n in rec: return rec.get(n)
        return ''
    return (_s(g('date')), _s(g('user_id','user')), _s(g('video_id','photo_id','item_id')), _s(g('author_id')), _s(g('tab')), _s(g('duration_ms','duration')))

def find_log_files(data_dir):
    pats=[os.path.join(data_dir,'log_standard*.csv'), os.path.join(data_dir,'*log_standard*.csv'), os.path.join(data_dir,'data','log_standard*.csv')]
    fs=[]
    for p in pats: fs += glob.glob(p)
    fs=sorted(set(f for f in fs if 'random' not in os.path.basename(f).lower()))
    def order(f):
        b=os.path.basename(f)
        return (0 if '4_08_to_4_21' in b else 1 if '4_22_to_5_08' in b else 2, b)
    return sorted(fs,key=order)

def parse_hour(v):
    try:
        x=int(float(v)); h=x//100 if x>=100 else x
        return h if 0<=h<=23 else 24
    except Exception:
        return 24

def date_to_dow(d):
    try: return datetime.strptime(_s(d),'%Y%m%d').weekday()
    except Exception: return 7

def parse_bin(v):
    if v is None or v=='': return np.nan
    try:
        x=float(v)
        if not np.isfinite(x): return np.nan
        return 1.0 if x>0 else 0.0
    except Exception:
        return np.nan

def read_raw_aux(data_dir):
    q=defaultdict(deque); raw=[]; per=defaultdict(int); gi=0; cols_seen=set()
    for fn in find_log_files(data_dir):
        try:
            with open(fn,newline='') as f:
                reader=csv.DictReader(f)
                if reader.fieldnames: cols_seen.update(reader.fieldnames)
                for rec in reader:
                    k=csv_key(rec); hour=parse_hour(rec.get('hourmin','')); loc=per[k[0]]; per[k[0]]+=1
                    aux=[parse_bin(rec.get(c,'')) for c in AUX_COLS]
                    raw.append((k,hour,gi,loc,aux)); gi+=1
        except FileNotFoundError:
            pass
    total=max(1,gi); totals=dict(per)
    for k,hour,idx,loc,aux in raw:
        d=k[0]
        q[k].append((hour, min(19,int(20*idx/total)), min(9,int(10*loc/max(1,totals.get(d,1)))), aux))
    return q,len(raw),sorted(c for c in AUX_COLS if c in cols_seen)

def build_augmented(splits,data_dir):
    enc0,dim0=encode(splits); auxq,nraw,cols=read_raw_aux(data_dir); extra={}; auxlab={}; miss=0
    for sp,rows in splits.items():
        ev=[]; al=[]
        for r in rows:
            k=row_key(r)
            if auxq.get(k):
                hour,gbin,dbin,a=auxq[k].popleft()
            else:
                hour,gbin,dbin,a=24,20,10,[np.nan]*len(AUX_COLS); miss+=1
            tab=_s(r[4]); date=_s(r[0]); dow=date_to_dow(date); hb=hour//4 if hour<24 else 6
            ev.append(('d='+date,'dow='+str(dow),'h='+str(hour),'hb='+str(hb),'th='+tab+'_'+str(hour),'td='+tab+'_'+date,'gb='+str(gbin),'db='+date+'_'+str(dbin)))
            al.append(a)
        extra[sp]=ev; auxlab[sp]=np.asarray(al,dtype=np.float32)
    maps=[]; off=dim0; n_extra=len(next(iter(extra.values()))[0])
    for j in range(n_extra):
        cats={}
        for sp in extra:
            for v in extra[sp]:
                if v[j] not in cats: cats[v[j]]=off+len(cats)
        maps.append(cats); off+=len(cats)
    enc={}
    for sp in splits:
        X0,y,u=enc0[sp]; H=np.empty((len(X0),n_extra),dtype=np.int64)
        for i,v in enumerate(extra[sp]):
            for j in range(n_extra): H[i,j]=maps[j][v[j]]
        enc[sp]=(np.concatenate([X0.astype(np.int64),H],1),y,u,auxlab[sp])
    print(f'raw_aux rows={nraw} missing_matches={miss} aux_cols_present={cols} extra_fields={n_extra} dim={off}',flush=True)
    return enc,off

class MultiTaskFM(torch.nn.Module):
    def __init__(self,dim,k=16,n_aux=7,seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
        self.aux=torch.nn.Linear(k,n_aux)
        torch.nn.init.normal_(self.aux.weight,0,0.01); torch.nn.init.zeros_(self.aux.bias)
    def forward(self,X):
        E=self.V[X]; s=E.sum(1); inter=0.5*((s*s).sum(1)-(E*E).sum((1,2)))
        main=self.b+self.W[X].sum(1)+inter
        aux=self.aux(s)
        return main,aux
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device))[0].cpu().numpy())
        return np.concatenate(out)

class TorchFM(torch.nn.Module):
    def __init__(self,dim,k=16,seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
    def forward(self,X):
        E=self.V[X]; s=E.sum(1); inter=0.5*((s*s).sum(1)-(E*E).sum((1,2)))
        return self.b+self.W[X].sum(1)+inter
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def pair_data(y,users,balanced=False):
    pos=defaultdict(list); neg=defaultdict(list)
    for i,(u,yy) in enumerate(zip(users,y)): (pos if yy>0.5 else neg)[u].append(i)
    pidx=[]; pools=[]; pw=[]
    for u,ps in pos.items():
        ns=neg.get(u)
        if not ns: continue
        arr=np.asarray(ns,dtype=np.int64); w=1.0/float(min(len(ps),5)) if balanced else 1.0
        for p in ps: pidx.append(p); pools.append(arr); pw.append(w)
    pw=np.asarray(pw,dtype=np.float32)
    if len(pw) and balanced: pw/=max(1e-6,pw.mean())
    return np.asarray(pidx,dtype=np.int64),pools,pw

def aux_bce(logits, labels):
    mask=torch.isfinite(labels)
    if mask.sum().item()==0:
        return logits.sum()*0.0
    lab=torch.nan_to_num(labels, nan=0.0)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits, lab, reduction='none')
    return loss[mask].mean()

def train_bpr(enc,dim,seed,balanced=False,k=16,lr=0.001,l2=3e-6,epochs=40,bs=8192,patience=4,device='cpu'):
    Xtr,ytr,utr,_=enc['train']; Xva,yva,uva,_=enc['valid']
    model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); bce=torch.nn.BCEWithLogitsLoss()
    rng=np.random.default_rng(seed+17); pidx,pools,pw=pair_data(ytr,utr,balanced=balanced); pw_t=torch.from_numpy(pw.astype(np.float32))
    n_steps=max(len(ytr),len(pidx)); best=-1; best_state=None; bad=0
    for ep in range(1,epochs+1):
        model.train(); order=rng.integers(0,len(pidx),size=n_steps,dtype=np.int64)
        for st in range(0,len(order),bs):
            which=order[st:st+bs]; pp=pidx[which]; nn=np.empty(len(which),dtype=np.int64)
            for j,w in enumerate(which):
                pool=pools[int(w)]; nn[j]=pool[rng.integers(0,len(pool))]
            both=np.concatenate([pp,nn]); logits=model(Xtr_t[torch.from_numpy(both)].to(device)); lp,ln=logits[:len(pp)],logits[len(pp):]
            base=torch.nn.functional.softplus(-(lp-ln))
            if balanced:
                wt=pw_t[torch.from_numpy(which)].to(device); loss_pair=(base*wt).sum()/torch.clamp(wt.sum(),min=1e-6)
            else:
                loss_pair=base.mean()
            loss=loss_pair+0.02*bce(logits,ytr_t[torch.from_numpy(both)].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pred=model.predict(Xva,device=device); m=evaluate(uva,yva,pred)['primary']
        print(f'bpr balanced={balanced} seed={seed} epoch={ep} valid_primary={m:.6f}',flush=True)
        if m>best+1e-5:
            best=m; bad=0; best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def train_mtbpr(enc,dim,seed,k=16,lr=0.001,l2=3e-6,epochs=35,bs=8192,patience=4,device='cpu'):
    Xtr,ytr,utr,atr=enc['train']; Xva,yva,uva,_=enc['valid']
    model=MultiTaskFM(dim,k,len(AUX_COLS),seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W,model.aux.weight,model.aux.bias],'weight_decay':l2},{'params':[model.b],'weight_decay':0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); atr_t=torch.from_numpy(atr.astype(np.float32))
    bce=torch.nn.BCEWithLogitsLoss(); rng=np.random.default_rng(seed+311); pidx,pools,pw=pair_data(ytr,utr,balanced=True); pw_t=torch.from_numpy(pw.astype(np.float32))
    n_steps=max(len(ytr),len(pidx)); best=-1; best_state=None; bad=0
    for ep in range(1,epochs+1):
        model.train(); order=rng.integers(0,len(pidx),size=n_steps,dtype=np.int64)
        for st in range(0,len(order),bs):
            which=order[st:st+bs]; pp=pidx[which]; nn=np.empty(len(which),dtype=np.int64)
            for j,w in enumerate(which):
                pool=pools[int(w)]; nn[j]=pool[rng.integers(0,len(pool))]
            both=np.concatenate([pp,nn]); both_t=torch.from_numpy(both)
            main,aux=model(Xtr_t[both_t].to(device)); lp,ln=main[:len(pp)],main[len(pp):]
            base=torch.nn.functional.softplus(-(lp-ln)); wt=pw_t[torch.from_numpy(which)].to(device)
            loss_pair=(base*wt).sum()/torch.clamp(wt.sum(),min=1e-6)
            loss_main=bce(main,ytr_t[both_t].to(device))
            loss_aux=aux_bce(aux,atr_t[both_t].to(device))
            loss=loss_pair+0.02*loss_main+0.10*loss_aux
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pred=model.predict(Xva,device=device); m=evaluate(uva,yva,pred)['primary']
        print(f'mtbpr seed={seed} epoch={ep} valid_primary={m:.6f}',flush=True)
        if m>best+1e-5:
            best=m; bad=0; best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def user_groups(users):
    g=defaultdict(list)
    for i,u in enumerate(users): g[u].append(i)
    return g

def z_by_user(s,users):
    s=np.asarray(s,dtype=np.float64); out=np.zeros_like(s)
    for idx in user_groups(users).values():
        v=s[idx]; sd=v.std(); out[idx]=(v-v.mean())/sd if sd>1e-8 else v-v.mean()
    return out

def pct_rank_by_user(s,users):
    s=np.asarray(s,dtype=np.float64); out=np.zeros_like(s)
    for idx in user_groups(users).values():
        idx=np.asarray(idx,dtype=np.int64); n=len(idx)
        if n<=1: continue
        o=np.argsort(s[idx],kind='mergesort'); r=np.empty(n); r[o]=np.arange(n)/float(n-1); out[idx]=r
    return out

def rrf_by_user(s,users,c=20.0):
    s=np.asarray(s,dtype=np.float64); out=np.zeros_like(s)
    for idx in user_groups(users).values():
        idx=np.asarray(idx,dtype=np.int64); n=len(idx)
        if n<=1: continue
        o=np.argsort(-s[idx],kind='mergesort'); r=np.empty(n); r[o]=1.0/(c+1.0+np.arange(n)); out[idx]=r
    return out

def composite(preds,users):
    z=[z_by_user(p,users) for p in preds]; r=[pct_rank_by_user(p,users) for p in preds]; rr=[rrf_by_user(p,users) for p in preds]
    return z_by_user(0.60*z_by_user(np.mean(z,0),users)+0.25*z_by_user(np.mean(r,0),users)+0.15*z_by_user(np.mean(rr,0),users),users)

def member_pred(enc,dim,target,mseed,split_name,device,mode):
    os.makedirs('pred_cache',exist_ok=True)
    prefix={'bpr':'010_time_bpr_v1','balanced':'016_time_bpr_ndcgbal_v1','mtbpr':'021_time_mtbpr_aux_v1'}[mode]
    path=os.path.join('pred_cache',f'{prefix}_{split_name}_seed{mseed}.npy')
    if os.path.isfile(path):
        print('load',path,flush=True); return np.load(path)
    if mode=='bpr': model=train_bpr(enc,dim,mseed,balanced=False,device=device)
    elif mode=='balanced': model=train_bpr(enc,dim,mseed,balanced=True,device=device)
    else: model=train_mtbpr(enc,dim,mseed,device=device)
    p=model.predict(enc[target][0],device=device).astype(np.float64); np.save(path,p); return p

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',required=True); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; split_name='dev'
    else:
        splits=load(a.data_dir); target=a.split; split_name=a.split
    print({k:len(v) for k,v in splits.items()}, 'base_fields=', FIELDS, flush=True)
    t0=time.time(); enc,dim=build_augmented(splits,a.data_dir); users=enc[target][2]
    seeds5=[a.seed,a.seed+101,a.seed+202,a.seed+303,a.seed+404]
    old=[member_pred(enc,dim,target,s,split_name,a.device,'bpr') for s in seeds5]
    bal=[member_pred(enc,dim,target,s,split_name,a.device,'balanced') for s in seeds5]
    parent=z_by_user(0.65*composite(old,users)+0.35*composite(bal,users),users)
    seeds3=[a.seed,a.seed+101,a.seed+202]
    mt=[member_pred(enc,dim,target,s,split_name,a.device,'mtbpr') for s in seeds3]
    mtc=composite(mt,users)
    scores=z_by_user(0.70*parent+0.30*mtc,users)
    print(f'multitask_aux_fusion parent=0.70 mtbpr=0.30 mt_members={len(seeds3)} total_time={time.time()-t0:.1f}s',flush=True)
    np.save(a.out,scores.astype(np.float64))
