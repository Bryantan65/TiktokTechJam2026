"""FM augmented with causal positive-history preference features."""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS  # noqa
from evaluate import evaluate  # noqa


class HistoryFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, .01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim))
        self.b = torch.nn.Parameter(torch.zeros(()))
        self.beta = torch.nn.Parameter(torch.zeros(4))

    def forward(self, X, H):
        E = self.V[X]; S = E.sum(1)
        fm = self.b + self.W[X].sum(1) + .5 * ((S*S).sum(1) - (E*E).sum((1,2)))
        return fm + (H * self.beta).sum(1)

    @torch.no_grad()
    def predict(self, X, H, bs=200000, device='cpu'):
        self.eval(); out=[]
        for i in range(0, len(X), bs):
            xb=torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)
            hb=torch.from_numpy(H[i:i+bs]).to(device)
            out.append(self(xb,hb).cpu().numpy())
        return np.concatenate(out)


def history_features(splits):
    """Smoothed causal positive rates for target video, author, tab, duration bucket."""
    stats=[{} for _ in range(4)]; out={}
    for sp in ('train','valid','test'):
        if sp not in splits: continue
        rows=splits[sp]; h=np.empty((len(rows),4),np.float32)
        for i,r in enumerate(rows):
            keys=(r[2],r[3],r[4],int(r[5])//10000)
            for j,key in enumerate(keys):
                p,n=stats[j].get(key,(0.,0))
                h[i,j]=np.log((p+1.)/(n-p+2.))
            if sp=='train':
                y=float(r[6])
                for j,key in enumerate(keys):
                    p,n=stats[j].get(key,(0.,0)); stats[j][key]=(p+y,n+1)
        out[sp]=h
    return out


def run(splits, seed=0, device='cpu', verbose=True):
    enc,dim=encode(splits); hist=history_features(splits)
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']
    model=HistoryFM(dim,16,seed).to(device)
    opt=torch.optim.Adam([{'params':[model.V,model.W],'weight_decay':1e-6},
                          {'params':[model.b,model.beta]}],lr=.001)
    Xt=torch.from_numpy(Xtr.astype(np.int64)); Ht=torch.from_numpy(hist['train']); yt=torch.from_numpy(ytr)
    rng=np.random.default_rng(seed); best=-1.; state=None; bad=0
    for ep in range(1,41):
        idx=rng.permutation(len(ytr)); model.train(); losses=[]; t=time.time()
        for i in range(0,len(idx),8192):
            q=torch.from_numpy(idx[i:i+8192]); xb=Xt[q].to(device); hb=Ht[q].to(device); yb=yt[q].to(device)
            opt.zero_grad(set_to_none=True); loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xb,hb),yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,model.predict(Xva,hist['valid'],device=device))
        if verbose: print(ep,np.mean(losses),va,time.time()-t)
        if va['primary']>best+1e-5:
            best=va['primary']; bad=0; state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=4: break
    model.load_state_dict(state); return model,enc,hist


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='./KuaiRand-Pure/data')
    ap.add_argument('--split',default='valid',choices=['train','valid','test','dev']); ap.add_argument('--out')
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else: splits=load(a.data_dir); target=a.split
    model,enc,hist=run(splits,a.seed,a.device,a.out is None)
    scores=model.predict(enc[target][0],hist[target],device=a.device)
    if a.out: np.save(a.out,scores.astype(np.float64)); print('wrote',len(scores))
    else: print('done',len(scores),FIELDS)
