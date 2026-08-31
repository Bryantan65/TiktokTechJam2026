import argparse, csv, glob, os, sys, time
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate  # noqa

AUX_COLS=['is_click','is_like','is_follow','is_comment','is_forward','is_profile_enter','is_hate']

def _s(x):
    if x is None: return ''
    if isinstance(x,bytes): x=x.decode('utf8')
    try:
        if isinstance(x,(float,np.floating)) and np.isfinite(x) and abs(x-int(x))<1e-6: return str(int(x))
    except Exception: pass
    return str(x)
def row_key(r): return (_s(r[0]),_s(r[1]),_s(r[2]),_s(r[3]),_s(r[4]),_s(r[5]))
def csv_key(rec):
    def g(*ns):
        for n in ns:
            if n in rec: return rec.get(n)
        return ''
    return (_s(g('date')),_s(g('user_id','user')),_s(g('video_id','photo_id','item_id')),_s(g('author_id')),_s(g('tab')),_s(g('duration_ms','duration')))
def find_log_files(data_dir):
    fs=[]
    for p in [os.path.join(data_dir,'log_standard*.csv'),os.path.join(data_dir,'*log_standard*.csv'),os.path.join(data_dir,'data','log_standard*.csv')]: fs+=glob.glob(p)
    fs=sorted(set(f for f in fs if 'random' not in os.path.basename(f).lower()))
    return sorted(fs,key=lambda f:(0 if '4_08_to_4_21' in os.path.basename(f) else 1 if '4_22_to_5_08' in os.path.basename(f) else 2,os.path.basename(f)))
def parse_hour(v):
    try:
        x=int(float(v)); h=x//100 if x>=100 else x; return h if 0<=h<=23 else 24
    except Exception: return 24
def date_to_dow(d):
    try: return datetime.strptime(_s(d),'%Y%m%d').weekday()
    except Exception: return 7
def parse_bin(v):
    if v is None or v=='': return np.nan
    try:
        x=float(v); return 1.0 if np.isfinite(x) and x>0 else 0.0 if np.isfinite(x) else np.nan
    except Exception: return np.nan

def read_raw_aux(data_dir):
    q=defaultdict(deque); raw=[]; per=defaultdict(int); gi=0
    for fn in find_log_files(data_dir):
        try:
            with open(fn,newline='') as f:
                for rec in csv.DictReader(f):
                    k=csv_key(rec); loc=per[k[0]]; per[k[0]]+=1
                    raw.append((k,parse_hour(rec.get('hourmin','')),gi,loc,[parse_bin(rec.get(c,'')) for c in AUX_COLS])); gi+=1
        except FileNotFoundError: pass
    total=max(1,gi); totals=dict(per)
    for k,hour,idx,loc,aux in raw:
        d=k[0]; q[k].append((hour,min(19,int(20*idx/total)),min(9,int(10*loc/max(1,totals.get(d,1)))),aux))
    return q,len(raw)

def build_augmented(splits,data_dir):
    enc0,dim0=encode(splits); auxq,nraw=read_raw_aux(data_dir); extra={}; auxlab={}; miss=0
    for sp,rows in splits.items():
        ev=[]; al=[]
        for r in rows:
            k=row_key(r)
            if auxq.get(k): hour,gbin,dbin,a=auxq[k].popleft()
            else: hour,gbin,dbin,a=24,20,10,[np.nan]*len(AUX_COLS); miss+=1
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
    print(f'raw_aux rows={nraw} missing={miss} dim={off}',flush=True); return enc,off

def build_hist(enc,K=20):
    hist={}; state=defaultdict(lambda: deque(maxlen=K)); pad=-1
    for sp in ['train','valid','test']:
        if sp not in enc: continue
        X,y,u,_=enc[sp]; hv=np.full((len(X),K),pad,dtype=np.int64); ha=np.full((len(X),K),pad,dtype=np.int64)
        for i,uu in enumerate(u):
            h=list(state[uu]); L=len(h)
            if L:
                vv,aa=zip(*h); hv[i,K-L:]=vv; ha[i,K-L:]=aa
            if sp=='train' and y[i]>0.5: state[uu].append((int(X[i,1]),int(X[i,2])))
        hist[sp]=(hv,ha)
    return hist

class TorchFM(torch.nn.Module):
    def __init__(self,dim,k=16,seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32))); self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
    def forward(self,X):
        E=self.V[X]; s=E.sum(1); return self.b+self.W[X].sum(1)+0.5*((s*s).sum(1)-(E*E).sum((1,2)))
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)
class MultiTaskFM(torch.nn.Module):
    def __init__(self,dim,k=16,n_aux=7,seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32))); self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(())); self.aux=torch.nn.Linear(k,n_aux); torch.nn.init.normal_(self.aux.weight,0,0.01); torch.nn.init.zeros_(self.aux.bias)
    def forward(self,X):
        E=self.V[X]; s=E.sum(1); inter=0.5*((s*s).sum(1)-(E*E).sum((1,2))); return self.b+self.W[X].sum(1)+inter,self.aux(s)
    @torch.no_grad()
    def predict(self,X,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device))[0].cpu().numpy())
        return np.concatenate(out)
class DINFM(torch.nn.Module):
    def __init__(self,dim,k=16,pad_id=None,seed=0):
        super().__init__(); rng=np.random.default_rng(seed); self.pad=pad_id if pad_id is not None else dim-1
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,0.01,(dim,k)).astype(np.float32))); self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
        with torch.no_grad(): self.V[self.pad].zero_()
        self.att=torch.nn.Sequential(torch.nn.Linear(4*k,32),torch.nn.PReLU(),torch.nn.Linear(32,1)); self.out=torch.nn.Linear(3*k,1); torch.nn.init.normal_(self.out.weight,0,0.01); torch.nn.init.zeros_(self.out.bias)
    def forward(self,X,hv,ha):
        E=self.V[X]; s=E.sum(1); base=self.b+self.W[X].sum(1)+0.5*((s*s).sum(1)-(E*E).sum((1,2)))
        item=self.V[X[:,1]]+self.V[X[:,2]]; mask=(hv!=self.pad); hist=self.V[hv]+self.V[ha]; it=item[:,None,:].expand_as(hist)
        a=self.att(torch.cat([hist,it,hist*it,it-hist],2)).squeeze(-1); a=a.masked_fill(~mask,-1e9)
        w=torch.softmax(a,1)*mask.float(); denom=w.sum(1,keepdim=True).clamp_min(1e-6); h=(hist*w[:,:,None]).sum(1)/denom
        return base+self.out(torch.cat([item,h,item*h],1)).squeeze(1)
    @torch.no_grad()
    def predict(self,X,hv,ha,bs=100000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device),torch.from_numpy(hv[i:i+bs].astype(np.int64)).to(device),torch.from_numpy(ha[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
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

def aux_bce(logits,labels):
    mask=torch.isfinite(labels)
    if mask.sum().item()==0: return logits.sum()*0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.nan_to_num(labels,nan=0.0),reduction='none')[mask].mean()

def train_bpr(enc,dim,seed,balanced=False,k=16,lr=0.001,l2=3e-6,epochs=40,bs=8192,patience=4,device='cpu'):
    Xtr,ytr,utr,_=enc['train']; Xva,yva,uva,_=enc['valid']; model=TorchFM(dim,k,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); bce=torch.nn.BCEWithLogitsLoss(); rng=np.random.default_rng(seed+17); pidx,pools,pw=pair_data(ytr,utr,balanced); pw_t=torch.from_numpy(pw.astype(np.float32)); n_steps=max(len(ytr),len(pidx)); best=-1; bad=0; best_state=None
    for ep in range(1,epochs+1):
        model.train(); order=rng.integers(0,len(pidx),size=n_steps,dtype=np.int64)
        for st in range(0,len(order),bs):
            which=order[st:st+bs]; pp=pidx[which]; nn=np.empty(len(which),dtype=np.int64)
            for j,w in enumerate(which):
                pool=pools[int(w)]; nn[j]=pool[rng.integers(0,len(pool))]
            both=np.concatenate([pp,nn]); both_t=torch.from_numpy(both); logits=model(Xtr_t[both_t].to(device)); lp,ln=logits[:len(pp)],logits[len(pp):]; base=torch.nn.functional.softplus(-(lp-ln))
            if balanced:
                wt=pw_t[torch.from_numpy(which)].to(device); loss_pair=(base*wt).sum()/torch.clamp(wt.sum(),min=1e-6)
            else: loss_pair=base.mean()
            loss=loss_pair+0.02*bce(logits,ytr_t[both_t].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m=evaluate(uva,yva,model.predict(Xva,device=device))['primary']; print(f'bpr balanced={balanced} seed={seed} ep={ep} p={m:.6f}',flush=True)
        if m>best+1e-5: best=m; bad=0; best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def train_mtbpr(enc,dim,seed,k=16,lr=0.001,l2=3e-6,epochs=35,bs=8192,patience=4,device='cpu'):
    Xtr,ytr,utr,atr=enc['train']; Xva,yva,uva,_=enc['valid']; model=MultiTaskFM(dim,k,len(AUX_COLS),seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W,model.aux.weight,model.aux.bias],'weight_decay':l2},{'params':[model.b],'weight_decay':0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); atr_t=torch.from_numpy(atr.astype(np.float32)); bce=torch.nn.BCEWithLogitsLoss(); rng=np.random.default_rng(seed+311); pidx,pools,pw=pair_data(ytr,utr,True); pw_t=torch.from_numpy(pw.astype(np.float32)); n_steps=max(len(ytr),len(pidx)); best=-1; bad=0; best_state=None
    for ep in range(1,epochs+1):
        model.train(); order=rng.integers(0,len(pidx),size=n_steps,dtype=np.int64)
        for st in range(0,len(order),bs):
            which=order[st:st+bs]; pp=pidx[which]; nn=np.empty(len(which),dtype=np.int64)
            for j,w in enumerate(which):
                pool=pools[int(w)]; nn[j]=pool[rng.integers(0,len(pool))]
            both=np.concatenate([pp,nn]); bt=torch.from_numpy(both); main,aux=model(Xtr_t[bt].to(device)); lp,ln=main[:len(pp)],main[len(pp):]; wt=pw_t[torch.from_numpy(which)].to(device); loss=(torch.nn.functional.softplus(-(lp-ln))*wt).sum()/torch.clamp(wt.sum(),min=1e-6)+0.02*bce(main,ytr_t[bt].to(device))+0.10*aux_bce(aux,atr_t[bt].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m=evaluate(uva,yva,model.predict(Xva,device=device))['primary']; print(f'mtbpr seed={seed} ep={ep} p={m:.6f}',flush=True)
        if m>best+1e-5: best=m; bad=0; best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def train_din(enc,hist,dim0,seed,k=16,lr=0.001,l2=3e-6,epochs=30,bs=4096,patience=4,device='cpu'):
    pad=dim0; Xtr,ytr,utr,_=enc['train']; Xva,yva,uva,_=enc['valid']; hvtr,hatr=hist['train']; hvva,hava=hist['valid']; hvtr=np.where(hvtr<0,pad,hvtr); hatr=np.where(hatr<0,pad,hatr); hvva=np.where(hvva<0,pad,hvva); hava=np.where(hava<0,pad,hava)
    model=DINFM(dim0+1,k,pad,seed).to(device); att_params=list(model.att.parameters())+list(model.out.parameters())
    opt=torch.optim.Adam([{'params':[model.V,model.W]+att_params,'weight_decay':l2},{'params':[model.b],'weight_decay':0}],lr=lr)
    Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); hv_t=torch.from_numpy(hvtr.astype(np.int64)); ha_t=torch.from_numpy(hatr.astype(np.int64)); ytr_t=torch.from_numpy(ytr.astype(np.float32)); bce=torch.nn.BCEWithLogitsLoss(); rng=np.random.default_rng(seed+701); pidx,pools,pw=pair_data(ytr,utr,True); pw_t=torch.from_numpy(pw.astype(np.float32)); n_steps=max(len(ytr),len(pidx)); best=-1; bad=0; best_state=None
    for ep in range(1,epochs+1):
        model.train(); order=rng.integers(0,len(pidx),size=n_steps,dtype=np.int64)
        for st in range(0,len(order),bs):
            which=order[st:st+bs]; pp=pidx[which]; nn=np.empty(len(which),dtype=np.int64)
            for j,w in enumerate(which):
                pool=pools[int(w)]; nn[j]=pool[rng.integers(0,len(pool))]
            both=np.concatenate([pp,nn]); bt=torch.from_numpy(both); logits=model(Xtr_t[bt].to(device),hv_t[bt].to(device),ha_t[bt].to(device)); lp,ln=logits[:len(pp)],logits[len(pp):]; wt=pw_t[torch.from_numpy(which)].to(device); loss=(torch.nn.functional.softplus(-(lp-ln))*wt).sum()/torch.clamp(wt.sum(),min=1e-6)+0.02*bce(logits,ytr_t[bt].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): model.V[pad].zero_()
        pred=model.predict(Xva,hvva,hava,device=device); m=evaluate(uva,yva,pred)['primary']; print(f'din seed={seed} ep={ep} p={m:.6f}',flush=True)
        if m>best+1e-5: best=m; bad=0; best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
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
        if n>1:
            o=np.argsort(s[idx],kind='mergesort'); r=np.empty(n); r[o]=np.arange(n)/float(n-1); out[idx]=r
    return out
def rrf_by_user(s,users,c=20.0):
    s=np.asarray(s,dtype=np.float64); out=np.zeros_like(s)
    for idx in user_groups(users).values():
        idx=np.asarray(idx,dtype=np.int64); n=len(idx)
        if n>1:
            o=np.argsort(-s[idx],kind='mergesort'); r=np.empty(n); r[o]=1.0/(c+1.0+np.arange(n)); out[idx]=r
    return out
def composite(preds,users):
    z=[z_by_user(p,users) for p in preds]; r=[pct_rank_by_user(p,users) for p in preds]; rr=[rrf_by_user(p,users) for p in preds]
    return z_by_user(0.60*z_by_user(np.mean(z,0),users)+0.25*z_by_user(np.mean(r,0),users)+0.15*z_by_user(np.mean(rr,0),users),users)

def member_pred(enc,dim,target,mseed,split_name,device,mode,hist=None):
    os.makedirs('pred_cache',exist_ok=True); prefix={'bpr':'010_time_bpr_v1','balanced':'016_time_bpr_ndcgbal_v1','mtbpr':'021_time_mtbpr_aux_v1','din':'024_din_histpos_v1'}[mode]; path=os.path.join('pred_cache',f'{prefix}_{split_name}_seed{mseed}.npy')
    if os.path.isfile(path): print('load',path,flush=True); return np.load(path)
    if mode=='bpr': model=train_bpr(enc,dim,mseed,balanced=False,device=device); p=model.predict(enc[target][0],device=device)
    elif mode=='balanced': model=train_bpr(enc,dim,mseed,balanced=True,device=device); p=model.predict(enc[target][0],device=device)
    elif mode=='mtbpr': model=train_mtbpr(enc,dim,mseed,device=device); p=model.predict(enc[target][0],device=device)
    else:
        model=train_din(enc,hist,dim,mseed,device=device); hv,ha=hist[target]; hv=np.where(hv<0,dim,hv); ha=np.where(ha<0,dim,ha); p=model.predict(enc[target][0],hv,ha,device=device)
    p=p.astype(np.float64); np.save(path,p); return p

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',required=True); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'; split_name='dev'
    else:
        splits=load(a.data_dir); target=a.split; split_name=a.split
    print({k:len(v) for k,v in splits.items()}, 'base_fields=', FIELDS, flush=True); t0=time.time(); enc,dim=build_augmented(splits,a.data_dir); hist=build_hist(enc,K=20); users=enc[target][2]
    seeds5=[a.seed,a.seed+101,a.seed+202,a.seed+303,a.seed+404]
    old=[member_pred(enc,dim,target,s,split_name,a.device,'bpr') for s in seeds5]
    bal=[member_pred(enc,dim,target,s,split_name,a.device,'balanced') for s in seeds5]
    mt=[member_pred(enc,dim,target,s,split_name,a.device,'mtbpr') for s in seeds5]
    parent=z_by_user(0.70*z_by_user(0.65*composite(old,users)+0.35*composite(bal,users),users)+0.30*composite(mt,users),users)
    din=[member_pred(enc,dim,target,s,split_name,a.device,'din',hist) for s in seeds5]
    dinc=composite(din,users)
    hv,_=hist[target]; hlen=(hv>=0).sum(1).astype(np.float64)
    # DIN history attention is only informative for users with enough previous positive watches.
    # Use a per-user gate: mostly parent for cold users, up to 35% sequence signal for rich histories.
    alpha=0.10+0.25*np.minimum(hlen,8.0)/8.0
    scores=z_by_user((1.0-alpha)*parent+alpha*dinc,users)
    print(f'din_history_gated_fusion alpha_mean={alpha.mean():.4f} alpha_min={alpha.min():.4f} alpha_max={alpha.max():.4f} total_time={time.time()-t0:.1f}s',flush=True)
    np.save(a.out,scores.astype(np.float64))
