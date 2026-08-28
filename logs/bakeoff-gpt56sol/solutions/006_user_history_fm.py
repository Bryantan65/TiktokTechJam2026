"""FM augmented by causal per-user target-category history affinities."""
import argparse, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate  # noqa

class HistoryFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        self.V=torch.nn.Parameter(torch.from_numpy(rng.normal(0,.01,(dim,k)).astype(np.float32)))
        self.W=torch.nn.Parameter(torch.zeros(dim)); self.b=torch.nn.Parameter(torch.zeros(()))
        self.beta=torch.nn.Parameter(torch.zeros(8))
    def forward(self,X,H):
        E=self.V[X]; S=E.sum(1)
        return self.b+self.W[X].sum(1)+.5*((S*S).sum(1)-(E*E).sum((1,2)))+(H*self.beta).sum(1)
    @torch.no_grad()
    def predict(self,X,H,bs=200000,device='cpu'):
        self.eval(); out=[]
        for i in range(0,len(X),bs):
            out.append(self(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device),torch.from_numpy(H[i:i+bs]).to(device)).cpu().numpy())
        return np.concatenate(out)

def history_features(splits):
    # Four global target rates plus four per-user target-category rates.
    glob=[{} for _ in range(4)]; usr=[{} for _ in range(4)]; out={}
    for sp in ('train','valid','test'):
        if sp not in splits: continue
        rows=splits[sp]; h=np.empty((len(rows),8),np.float32)
        for i,r in enumerate(rows):
            u=r[1]; keys=(r[2],r[3],r[4],int(r[5])//10000)
            for j,key in enumerate(keys):
                gp,gn=glob[j].get(key,(0.,0)); prior=(gp+2.)/(gn+4.)
                h[i,j]=np.log((gp+1.)/(gn-gp+2.))
                up,un=usr[j].get((u,key),(0.,0))
                # Hierarchical smoothing toward the target's global rate.
                q=(up+5.*prior)/(un+5.)
                h[i,4+j]=np.log((q+1e-4)/(1.-q+1e-4))
            if sp=='train':
                y=float(r[6])
                for j,key in enumerate(keys):
                    gp,gn=glob[j].get(key,(0.,0)); glob[j][key]=(gp+y,gn+1)
                    uk=(u,key); up,un=usr[j].get(uk,(0.,0)); usr[j][uk]=(up+y,un+1)
        out[sp]=h
    return out

def run(splits,seed=0,device='cpu',verbose=True):
    enc,dim=encode(splits); hist=history_features(splits); Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']
    m=HistoryFM(dim,16,seed).to(device)
    opt=torch.optim.Adam([{'params':[m.V,m.W],'weight_decay':1e-6},{'params':[m.b,m.beta]}],lr=.001)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); Ht=torch.from_numpy(hist['train']); yt=torch.from_numpy(ytr)
    rng=np.random.default_rng(seed); best=-1.; state=None; bad=0
    for ep in range(1,41):
        idx=rng.permutation(len(ytr)); m.train(); losses=[]; t=time.time()
        for i in range(0,len(idx),8192):
            q=torch.from_numpy(idx[i:i+8192]); xb=Xt[q].to(device); hb=Ht[q].to(device); yb=yt[q].to(device)
            opt.zero_grad(set_to_none=True); loss=torch.nn.functional.binary_cross_entropy_with_logits(m(xb,hb),yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,m.predict(Xva,hist['valid'],device=device))
        if verbose: print(ep,np.mean(losses),va,time.time()-t)
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; state={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=4: break
    m.load_state_dict(state); return m,enc,hist

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data'); ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args(); torch.manual_seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else: splits=load(a.data_dir); target=a.split
    m,enc,hist=run(splits,a.seed,a.device,a.out is None); scores=m.predict(enc[target][0],hist[target],device=a.device)
    if a.out: np.save(a.out,scores.astype(np.float64)); print('wrote',len(scores))
    else: print('done',len(scores),FIELDS)
