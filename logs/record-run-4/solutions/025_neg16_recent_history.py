"""Sampled-softmax multitask FM plus time and recent history; use more same-user negatives."""
import argparse, csv, datetime as _dt, os, sys, time
from collections import defaultdict, deque
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

AUX_NAMES = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_profile_enter', 'is_hate')

class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, k=16, n_aux=0, seed=0):
        super().__init__(); rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32)); self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.n_aux = int(n_aux)
        if self.n_aux:
            self.W_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dim, dtype=torch.float32)); self.b_aux = torch.nn.Parameter(torch.zeros(self.n_aux, dtype=torch.float32))
    def _inter(self, X):
        E = self.V[X]; S = E.sum(1); return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
    def forward(self, X): return self.b + self.W[X].sum(1) + self._inter(X)
    def aux_forward(self, X): return self.b_aux.view(1, -1) + self.W_aux[:, X].sum(2).t() + self._inter(X).view(-1, 1)
    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval(); out=[]
        for i in range(0, len(X), bs): out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)

def _to_int(x, default=0):
    try:
        if x is None or x == '': return default
        return int(float(x))
    except Exception: return default

def _get(row, *names):
    for n in names:
        if n in row: return row[n]
    return None

def _dow_from_date(d):
    try: return _dt.datetime.strptime(str(int(d)), '%Y%m%d').weekday()
    except Exception: return 0

def _raw_maps(data_dir):
    full=defaultdict(deque); nodur=defaultdict(deque)
    for name in ['log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv']:
        path=os.path.join(data_dir,name)
        if not os.path.exists(path): continue
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                date=_to_int(_get(r,'date')); user=_to_int(_get(r,'user_id','user')); video=_to_int(_get(r,'video_id','item_id','photo_id'))
                author=_to_int(_get(r,'author_id')); tab=_to_int(_get(r,'tab')); tab_clip=max(0,min(tab,7)); dur=_to_int(_get(r,'duration_ms','duration'))
                lab=_to_int(_get(r,'long_view','label')); hm=_to_int(_get(r,'hourmin','hour_min','time')); hour=(hm//100)%24 if hm>=100 else hm%24
                extra=(hour, _dow_from_date(date), hour*8+tab_clip); aux=tuple(float(_to_int(_get(r,n),0)) for n in AUX_NAMES); val=(extra,aux)
                full[(date,user,video,author,tab,dur,lab)].append(val); nodur[(date,user,video,author,tab,lab)].append(val)
    return full,nodur

def encode_with_time_aux(splits, data_dir):
    enc, dim = encode(splits); full,nodur=_raw_maps(data_dir)
    sizes=np.array([24,7,24*8], dtype=np.int64); offsets=dim+np.concatenate([[0],np.cumsum(sizes)[:-1]]).astype(np.int64)
    out={}; aux_out={}; matched={}
    for sp, rows in splits.items():
        Xb,y,users=enc[sp]; extra=np.zeros((len(rows),len(sizes)),dtype=np.int64); aux=np.zeros((len(rows),len(AUX_NAMES)),dtype=np.float32); ok=np.zeros(len(rows),dtype=bool)
        for i,row in enumerate(rows):
            date,user,video,author,tab,dur,lab=row
            q=full.get((_to_int(date),_to_int(user),_to_int(video),_to_int(author),_to_int(tab),_to_int(dur),_to_int(lab)))
            if not q: q=nodur.get((_to_int(date),_to_int(user),_to_int(video),_to_int(author),_to_int(tab),_to_int(lab)))
            if q:
                e,a=q.popleft(); extra[i]=e; aux[i]=a; ok[i]=True
        out[sp]=(np.concatenate([Xb.astype(np.int64), extra+offsets], axis=1), y, users); aux_out[sp]=aux; matched[sp]=ok
    return out, aux_out, matched, int(dim+sizes.sum())

def make_pair_sampler(y, users):
    y=np.asarray(y); users=np.asarray(users); order=np.argsort(users, kind='mergesort'); us=users[order]; pos_chunks=[]; neg_pools=[]; i=0
    while i < len(order):
        j=i+1
        while j < len(order) and us[j]==us[i]: j+=1
        idx=order[i:j]; pos=idx[y[idx]>0.5]; neg=idx[y[idx]<=0.5]
        if len(pos) and len(neg): pos_chunks.append(pos.astype(np.int64,copy=False)); neg=neg.astype(np.int64,copy=False); neg_pools.extend([neg]*len(pos))
        i=j
    if not pos_chunks: raise RuntimeError('no users with both positive and negative rows')
    return np.concatenate(pos_chunks), np.asarray(neg_pools, dtype=object)

def _clip01(x): return min(max(float(x),1e-4),1.0-1e-4)
def _logit(p): p=_clip01(p); return np.log(p/(1.0-p))

def build_history_stats(train_rows, decay=0.97):
    up=defaultdict(float); uc=defaultdict(float); uap=defaultdict(float); uac=defaultdict(float); uvp=defaultdict(float); uvc=defaultdict(float); utp=defaultdict(float); utc=defaultdict(float); gp=0.; gc=0.
    t_user=defaultdict(int); final_t=defaultdict(int); ruap=defaultdict(float); ruac=defaultdict(float); rual={}; ruvp=defaultdict(float); ruvc=defaultdict(float); ruvl={}; rutp=defaultdict(float); rutc=defaultdict(float); rutl={}
    def upd(p,c,l,key,t,lab):
        last=l.get(key)
        if last is not None:
            fac=decay**(t-last); p[key]*=fac; c[key]*=fac
        p[key]+=lab; c[key]+=1.; l[key]=t
    for r in train_rows:
        user=_to_int(r[1]); video=_to_int(r[2]); author=_to_int(r[3]); tab=_to_int(r[4]); lab=float(_to_int(r[6])); gp+=lab; gc+=1.; up[user]+=lab; uc[user]+=1.
        ka=(user,author); kv=(user,video); kt=(user,tab); uap[ka]+=lab; uac[ka]+=1.; uvp[kv]+=lab; uvc[kv]+=1.; utp[kt]+=lab; utc[kt]+=1.
        t_user[user]+=1; t=t_user[user]; final_t[user]=t; upd(ruap,ruac,rual,ka,t,lab); upd(ruvp,ruvc,ruvl,kv,t,lab); upd(rutp,rutc,rutl,kt,t,lab)
    for p,c,l in ((ruap,ruac,rual),(ruvp,ruvc,ruvl),(rutp,rutc,rutl)):
        for key,last in list(l.items()):
            fac=decay**(final_t[key[0]]-last); p[key]*=fac; c[key]*=fac
    return {'global':gp/max(gc,1.),'up':up,'uc':uc,'uap':uap,'uac':uac,'uvp':uvp,'uvc':uvc,'utp':utp,'utc':utc,'ruap':ruap,'ruac':ruac,'ruvp':ruvp,'ruvc':ruvc,'rutp':rutp,'rutc':rutc}

def history_adjust(rows, stats):
    g=stats['global']; out=np.zeros(len(rows),dtype=np.float32)
    for i,r in enumerate(rows):
        user=_to_int(r[1]); video=_to_int(r[2]); author=_to_int(r[3]); tab=_to_int(r[4]); uc=stats['uc'].get(user,0.); ur=(stats['up'].get(user,0.)+8.*g)/(uc+8.); base=_logit(ur); adj=0.
        ka=(user,author); ca=stats['uac'].get(ka,0.)
        if ca>0: adj += 0.45*np.sqrt(ca/(ca+4.))*(_logit((stats['uap'].get(ka,0.)+2.5*ur)/(ca+2.5))-base)
        kv=(user,video); cv=stats['uvc'].get(kv,0.)
        if cv>0: adj += 0.30*np.sqrt(cv/(cv+3.))*(_logit((stats['uvp'].get(kv,0.)+1.5*ur)/(cv+1.5))-base)
        kt=(user,tab); ct=stats['utc'].get(kt,0.)
        if ct>0: adj += 0.20*np.sqrt(ct/(ct+8.))*(_logit((stats['utp'].get(kt,0.)+4.*ur)/(ct+4.))-base)
        rca=stats['ruac'].get(ka,0.)
        if rca>0: adj += 0.10*np.sqrt(rca/(rca+2.))*(_logit((stats['ruap'].get(ka,0.)+1.5*ur)/(rca+1.5))-base)
        rcv=stats['ruvc'].get(kv,0.)
        if rcv>0: adj += 0.08*np.sqrt(rcv/(rcv+2.))*(_logit((stats['ruvp'].get(kv,0.)+1.0*ur)/(rcv+1.0))-base)
        rct=stats['rutc'].get(kt,0.)
        if rct>0: adj += 0.06*np.sqrt(rct/(rct+5.))*(_logit((stats['rutp'].get(kt,0.)+2.5*ur)/(rct+2.5))-base)
        out[i]=np.float32(np.clip(adj,-1.,1.))
    return out

def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=4096, patience=4, neg_k=16, aux_weight=0.05, seed=0, device='cpu', verbose=True):
    enc,aux,matched,dim=encode_with_time_aux(splits,data_dir); Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']
    valid_rows=matched['train']; prev=aux['train'][valid_rows].mean(0) if valid_rows.any() else np.zeros(len(AUX_NAMES)); keep_aux=np.where((prev>0.001)&(prev<0.999))[0]
    if len(keep_aux)==0: aux_weight=0.0
    aux_tr=aux['train'][:,keep_aux] if len(keep_aux) else np.zeros((len(Xtr),0),dtype=np.float32)
    model=MultiTaskFM(dim,k=k,n_aux=len(keep_aux),seed=seed).to(device); params=[{'params':[model.V,model.W],'weight_decay':l2},{'params':[model.b],'weight_decay':0.0}]
    if len(keep_aux): params += [{'params':[model.W_aux],'weight_decay':l2},{'params':[model.b_aux],'weight_decay':0.0}]
    opt=torch.optim.Adam(params,lr=lr,betas=(0.9,0.999),eps=1e-8); Xtr_t=torch.from_numpy(Xtr.astype(np.int64)); aux_t=torch.from_numpy(aux_tr.astype(np.float32))
    pos_idx,neg_pools=make_pair_sampler(ytr,utr); rng=np.random.default_rng(seed); best=-1.; best_state=None; bad=0
    for ep in range(1,epochs+1):
        perm=rng.permutation(len(pos_idx)); t0=time.time(); model.train(); losses=[]
        for i in range(0,len(perm),bs):
            psel=perm[i:i+bs]; bsz=len(psel); pidx=pos_idx[psel]; nidx=np.empty((bsz,neg_k),dtype=np.int64)
            for t,pool in enumerate(neg_pools[psel]): nidx[t]=pool[rng.integers(len(pool),size=neg_k)]
            cand_idx=np.concatenate([pidx.reshape(-1,1),nidx],axis=1); xp=Xtr_t[torch.from_numpy(pidx)].to(device); xn=Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True); logits=torch.cat([model(xp).view(bsz,1), model(xn).view(bsz,neg_k)], dim=1); loss=torch.nn.functional.cross_entropy(logits, torch.zeros(bsz,dtype=torch.long,device=device))
            if aux_weight>0.0 and len(keep_aux):
                flat=cand_idx.reshape(-1); loss = loss + aux_weight*torch.nn.functional.binary_cross_entropy_with_logits(model.aux_forward(Xtr_t[torch.from_numpy(flat)].to(device)), aux_t[torch.from_numpy(flat)].to(device))
            loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,model.predict(Xva,device=device))
        if verbose:
            kept=','.join(AUX_NAMES[i] for i in keep_aux); print(f"  epoch {ep:2d} | mt-softmax-neg16 {np.mean(losses):.4f} aux=[{kept}] | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best=va['primary']; bad=0; best_state={k_:v.detach().clone() for k_,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state); hist=build_history_stats(splits['train']); return model,enc,hist

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test']); ap.add_argument('--out',default=None)
    ap.add_argument('--k',type=int,default=16); ap.add_argument('--lr',type=float,default=0.001); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed); print(f"loading {a.data_dir} ..."); splits=load(a.data_dir); print({k_:len(v) for k_,v in splits.items()}, f"fields={FIELDS}+hour,dow,hour_tab aux={AUX_NAMES} recent_history neg16")
    model,enc,hist=run(splits,a.data_dir,k=a.k,lr=a.lr,epochs=a.epochs,seed=a.seed,device=a.device,verbose=a.out is None)
    X,y,users=enc[a.split]; scores=model.predict(X,device=a.device)+history_adjust(splits[a.split],hist)
    if a.out:
        np.save(a.out,scores.astype(np.float64)); print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== recent_history_neg16 (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid','test'):
            Xs,ys,us=enc[sp]; pred=model.predict(Xs,device=a.device)+history_adjust(splits[sp],hist); r=evaluate(us,ys,pred)
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
