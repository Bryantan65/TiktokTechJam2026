"""DeepFM + censored watch-time auxiliary/survival score.

Extends node 17. The raw play_time_ms auxiliary MSE is replaced by a simple
censored log-normal watch-time model: if play reaches the video duration the
latent desired watch time is right-censored, otherwise the observed play time is
an uncensored sample. The predicted survival margin over duration is also added
as a small learned residual to the ranking logit.
"""
import argparse, csv, os, sys, time
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa
from evaluate import evaluate                  # noqa

AUX_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
PLAY_COL = 'play_time_ms'
CWM_SIGMA = 0.25


class Model(torch.nn.Module):
    def __init__(self, dim, n_fields, k=16, n_aux=0, use_cwm=True, seed=0):
        super().__init__(); rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32)); self.b = torch.nn.Parameter(torch.zeros(()))
        self.deep_scale = torch.nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.cwm_scale = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.n_aux = int(n_aux); self.use_cwm = bool(use_cwm)
        self.deep = torch.nn.Sequential(torch.nn.Linear(n_fields * k, 64), torch.nn.ReLU(), torch.nn.Dropout(0.10),
                                        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.05),
                                        torch.nn.Linear(32, 1))
        with torch.no_grad():
            for m in self.deep:
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(torch.from_numpy(rng.normal(0, 0.02, tuple(m.weight.shape)).astype(np.float32))); m.bias.zero_()
        if self.n_aux > 0:
            self.aux_A = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (k, self.n_aux)).astype(np.float32)))
            self.aux_W = torch.nn.Parameter(torch.zeros((dim, self.n_aux))); self.aux_b = torch.nn.Parameter(torch.zeros(self.n_aux))
        if self.use_cwm:
            self.cwm_A = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (k, 1)).astype(np.float32)))
            self.cwm_W = torch.nn.Parameter(torch.zeros((dim, 1))); self.cwm_b = torch.nn.Parameter(torch.ones(1))

    def shared(self, X):
        E = self.V[X]; return E, E.sum(1)

    def cwm_mu(self, X):
        E, S = self.shared(X)
        return (S @ self.cwm_A + self.cwm_W[X].sum(1) + self.cwm_b).squeeze(1)

    def forward(self, X, dlog=None):
        E, S = self.shared(X)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        out = self.b + self.W[X].sum(1) + inter + self.deep_scale * self.deep(E.reshape(E.shape[0], -1)).squeeze(1)
        if self.use_cwm and dlog is not None:
            mu = (S @ self.cwm_A + self.cwm_W[X].sum(1) + self.cwm_b).squeeze(1)
            out = out + self.cwm_scale * ((mu - dlog) / CWM_SIGMA).clamp(-6.0, 6.0)
        return out

    def aux_logits(self, X):
        if self.n_aux <= 0: return None
        _E, S = self.shared(X)
        return S @ self.aux_A + self.aux_W[X].sum(1) + self.aux_b

    @torch.no_grad()
    def predict(self, X, dlog, bs=200000, device='cpu'):
        self.eval(); out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)
            db = torch.from_numpy(dlog[i:i+bs].astype(np.float32)).to(device)
            out.append(self(xb, db).cpu().numpy())
        return np.concatenate(out)


def _weekday(x):
    try: return datetime.strptime(str(int(x)), '%Y%m%d').weekday()
    except Exception: return 7


def _read_raw(data_dir):
    files = [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'), os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]
    q = defaultdict(deque); present = set(); has_play = False; n = 0
    for path in files:
        if not os.path.exists(path): continue
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f); cols = set(rd.fieldnames or [])
            if not {'date','user_id','video_id','tab','hourmin'}.issubset(cols): continue
            present.update([c for c in AUX_COLS if c in cols]); has_play = has_play or (PLAY_COL in cols)
            for r in rd:
                try:
                    key = (int(r['date']), str(r['user_id']), str(r['video_id']), int(r['tab']))
                    hm = int(float(r['hourmin']))
                    aux = tuple(1.0 if c in cols and float(r.get(c, 0.0) or 0.0) > 0 else (np.nan if c not in cols else 0.0) for c in AUX_COLS)
                    play_raw = max(float(r.get(PLAY_COL, '') or 0.0), 0.0) if PLAY_COL in cols else np.nan
                except Exception:
                    continue
                q[key].append((hm, aux, play_raw)); n += 1
    active = [c for c in AUX_COLS if c in present]; active_idx = [AUX_COLS.index(c) for c in active]
    print(f"raw rows loaded: {n:,d} keys={len(q):,d} aux={active} play={has_play}")
    return q, active, active_idx, has_play


def augment(splits, enc, dim, data_dir):
    raw_q, active, active_idx, has_play = _read_raw(data_dir)
    off_w = dim; off_h = off_w + 8; off_p = off_h + 25; new_dim = off_p + 7
    new_enc = {}; aux = {}; cwm = {}; dlogs = {}; total = matched = 0
    for sp, (X, y, users) in enc.items():
        rows = splits[sp]; extra = np.empty((len(rows), 3), dtype=np.int64)
        aa = np.zeros((len(rows), len(active)), dtype=np.float32); mm = np.zeros_like(aa)
        obs = np.zeros(len(rows), dtype=np.float32); cens = np.zeros(len(rows), dtype=np.float32); mask = np.zeros(len(rows), dtype=np.float32)
        dlog = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            date, uid, vid, _aid, tab, dur_ms, _lab = row
            key = (int(date), str(uid), str(vid), int(tab)); hm = None; avals = None; play_raw = np.nan
            if key in raw_q and raw_q[key]: hm, avals, play_raw = raw_q[key].popleft(); matched += 1
            total += 1; wd = _weekday(date)
            if hm is None: hour, part = 24, 6
            else:
                hour = int(hm) // 100
                if hour < 0 or hour > 23: hour, part = 24, 6
                else: part = hour // 4
            extra[i] = (off_w + (wd if 0 <= wd <= 6 else 7), off_h + hour, off_p + part)
            if avals is not None:
                for j, ai in enumerate(active_idx):
                    v = avals[ai]
                    if not np.isnan(v): aa[i, j] = v; mm[i, j] = 1.0
            dur = max(float(dur_ms or 0.0), 1.0); dlog[i] = np.log1p(dur) / 10.0
            if not np.isnan(play_raw):
                obs[i] = np.log1p(max(play_raw, 0.0)) / 10.0; mask[i] = 1.0
                cens[i] = 1.0 if play_raw >= 0.95 * dur else 0.0
        new_enc[sp] = (np.concatenate([X.astype(np.int64), extra], axis=1), y, users)
        aux[sp] = (aa, mm); cwm[sp] = (obs, cens, mask); dlogs[sp] = dlog
    print(f"raw matched {matched:,d}/{total:,d} ({matched/max(total,1):.3%}); dim {dim}->{new_dim}")
    return new_enc, new_dim, aux, cwm, dlogs, active, has_play


def groups(users, y):
    p, n = defaultdict(list), defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)): (p if yy > 0.5 else n)[u].append(i)
    return [(np.asarray(p[u], dtype=np.int64), np.asarray(n[u], dtype=np.int64)) for u in p.keys() if u in n]


def make_pairs(gs, rng, train_scores=None, semi_k=4, semi_frac=0.25, lambda_alpha=2.5):
    left=[]; right=[]; wts=[]; order=np.arange(len(gs)); rng.shuffle(order); disc5=np.asarray([1/np.log2(i+2) for i in range(5)], dtype=np.float32)
    for gi in order:
        ps, ns = gs[gi]; m=len(ps); chosen=rng.choice(ns, size=m, replace=True)
        if train_scores is not None and semi_k > 1 and semi_frac > 0:
            mask = rng.random(m) < semi_frac; hh = int(mask.sum())
            if hh:
                psel=ps[mask]; cand=rng.choice(ns, size=(hh, semi_k), replace=True); cs=train_scores[cand]; ok=cs < train_scores[psel,None]; any_ok=ok.any(1)
                if any_ok.any():
                    rows=np.where(any_ok)[0]; cols=np.argmax(np.where(ok, cs, -np.inf)[rows], axis=1); tmp=chosen[mask]; tmp[rows]=cand[rows,cols]; chosen[mask]=tmp
        weights=np.ones(m, dtype=np.float32)
        if train_scores is not None and lambda_alpha > 0:
            all_idx=np.concatenate([ps,ns]); ord2=np.argsort(-train_scores[all_idx], kind='mergesort'); ranks=np.empty(len(all_idx), dtype=np.int32); ranks[ord2]=np.arange(1,len(all_idx)+1)
            mp=dict(zip(all_idx.tolist(), ranks.tolist())); rp=np.fromiter((mp[int(x)] for x in ps), dtype=np.int32, count=m); rn=np.fromiter((mp[int(x)] for x in chosen), dtype=np.int32, count=m)
            dp=np.where(rp<=5, 1/np.log2(rp.astype(np.float32)+1), 0); dn=np.where(rn<=5, 1/np.log2(rn.astype(np.float32)+1), 0); idcg=float(disc5[:min(len(ps),5)].sum())
            if idcg > 0: weights=(1 + lambda_alpha*np.abs(dp-dn)/idcg).astype(np.float32)
        left.append(ps); right.append(chosen.astype(np.int64)); wts.append(weights)
    p=np.concatenate(left); n=np.concatenate(right); w=np.concatenate(wts); perm=rng.permutation(len(p)); return p[perm], n[perm], w[perm]


def multitask_loss(model, xb, ab, mb, obs, cens, cmask):
    losses=[]
    if model.n_aux > 0 and ab.shape[1] > 0:
        logits=model.aux_logits(xb); lm=torch.nn.functional.binary_cross_entropy_with_logits(logits, ab, reduction='none'); den=mb.sum()
        if den.item() > 0: losses.append((lm*mb).sum()/den)
    if model.use_cwm:
        den=cmask.sum()
        if den.item() > 0:
            mu=model.cwm_mu(xb); zobs=(obs-mu)/CWM_SIGMA; zdur=(obs.new_zeros(obs.shape)) # placeholder overwritten below outside? no duration not needed for uncens threshold only
            # censored records use threshold equal to observed duration proxy? obs is replaced by duration for censored below by caller not here
            log_pdf=0.5*zobs*zobs
            surv=torch.clamp(0.5*torch.special.erfc(zobs/np.sqrt(2.0)), min=1e-7)
            loss=torch.where(cens > 0.5, -torch.log(surv), log_pdf)
            losses.append((loss*cmask).sum()/den)
    return None if not losses else sum(losses)/len(losses)


def run(splits, data_dir, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=True):
    enc0, dim0 = encode(splits); enc, dim, aux, cwm, dlogs, active, has_play = augment(splits, enc0, dim0, data_dir)
    Xtr,ytr,utr=enc['train']; Xva,yva,uva=enc['valid']; dtr=dlogs['train']; dva=dlogs['valid']
    atr,mtr=aux['train']; obs,cens,cmask=cwm['train']
    # For censored samples, CWM likelihood is survival at duration, not at observed play.
    obs_thr = obs.copy(); obs_thr[cens > 0.5] = dtr[cens > 0.5]
    model=Model(dim, Xtr.shape[1], k=k, n_aux=len(active), use_cwm=has_play, seed=seed).to(device)
    decay=[model.V, model.W] + list(model.deep.parameters()); nodecay=[model.b, model.deep_scale, model.cwm_scale]
    if len(active)>0: decay += [model.aux_A, model.aux_W]; nodecay += [model.aux_b]
    if has_play: decay += [model.cwm_A, model.cwm_W]; nodecay += [model.cwm_b]
    opt=torch.optim.Adam([{'params':decay,'weight_decay':l2},{'params':nodecay,'weight_decay':0.0}], lr=lr)
    X_t=torch.from_numpy(Xtr.astype(np.int64)); y_t=torch.from_numpy(ytr.astype(np.float32)); d_t=torch.from_numpy(dtr.astype(np.float32))
    a_t=torch.from_numpy(atr.astype(np.float32)); m_t=torch.from_numpy(mtr.astype(np.float32)); o_t=torch.from_numpy(obs_thr.astype(np.float32)); c_t=torch.from_numpy(cens.astype(np.float32)); cm_t=torch.from_numpy(cmask.astype(np.float32))
    gs=groups(utr,ytr); rng=np.random.default_rng(seed); bce=torch.nn.BCEWithLogitsLoss(); best=-1; best_state=None; bad=0
    for ep in range(1, epochs+1):
        t0=time.time(); model.train(); losses=[]
        if ep <= 1:
            idx=rng.permutation(len(ytr))
            for i in range(0,len(idx),bs):
                sel=torch.from_numpy(idx[i:i+bs]); xb=X_t[sel].to(device); db=d_t[sel].to(device); yb=y_t[sel].to(device)
                opt.zero_grad(set_to_none=True); loss=bce(model(xb,db), yb)
                ml=multitask_loss(model, xb, a_t[sel].to(device), m_t[sel].to(device), o_t[sel].to(device), c_t[sel].to(device), cm_t[sel].to(device))
                if ml is not None: loss = loss + 0.05*ml
                loss.backward(); opt.step(); losses.append(loss.item())
        else:
            train_scores=model.predict(Xtr,dtr,device=device); pidx,nidx,w=make_pairs(gs,rng,train_scores=train_scores)
            for i in range(0,len(pidx),bs):
                ps_np=pidx[i:i+bs]; ns_np=nidx[i:i+bs]; ps=torch.from_numpy(ps_np); ns=torch.from_numpy(ns_np); wt=torch.from_numpy(w[i:i+bs]).to(device)
                xp=X_t[ps].to(device); xn=X_t[ns].to(device); dp=d_t[ps].to(device); dn=d_t[ns].to(device)
                opt.zero_grad(set_to_none=True); sp=model(xp,dp); sn=model(xn,dn)
                loss=(torch.nn.functional.softplus(-(sp-sn))*wt).sum()/(wt.sum()+1e-8) + 0.15*bce(torch.cat([sp,sn]), torch.cat([torch.ones_like(sp), torch.zeros_like(sn)]))
                both=torch.from_numpy(np.concatenate([ps_np,ns_np])); xb=X_t[both].to(device)
                ml=multitask_loss(model, xb, a_t[both].to(device), m_t[both].to(device), o_t[both].to(device), c_t[both].to(device), cm_t[both].to(device))
                if ml is not None: loss = loss + 0.05*ml
                loss.backward(); opt.step(); losses.append(loss.item())
        va=evaluate(uva,yva,model.predict(Xva,dva,device=device))
        if verbose: print(f"  epoch {ep:2d} cwm | scales d={float(model.deep_scale.detach().cpu()):+.3f} c={float(model.cwm_scale.detach().cpu()):+.3f} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5: best=va['primary']; bad=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state); return model, enc, dlogs


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir', default='./KuaiRand-Pure/data'); ap.add_argument('--split', default='valid', choices=['train','valid','test','dev']); ap.add_argument('--out', default=None); ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu', choices=['cpu','cuda']); a=ap.parse_args()
    torch.manual_seed(a.seed); print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f"fields={FIELDS}+time aux+CWM")
    model, enc, dlogs = run(splits, a.data_dir, seed=a.seed, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=a.out is None)
    X,y,u=enc[target]; scores=model.predict(X, dlogs[target], device=a.device)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        for sp in ('valid','test'):
            Xs,ys,us=enc[sp]; r=evaluate(us,ys,model.predict(Xs,dlogs[sp],device=a.device)); print(sp, r)
