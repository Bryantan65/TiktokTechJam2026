"""Sampled same-user softmax ranker on top of the node-20 feature/model stack.

Node 4's listwise attempt was a large negative; this is a diagnostic re-test of
that mechanism with the now-working stack and a safer sampled-softmax
formulation: each training example is one positive video contrasted against
several negatives from the same user, with the positive at class 0.
"""
import argparse, importlib.util, os, sys, time
from collections import defaultdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, '020_lambdamart_residual.py')
spec = importlib.util.spec_from_file_location('node20', PARENT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def group_pos_neg(users, y):
    p, n = defaultdict(list), defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        (p if yy > 0.5 else n)[u].append(i)
    out = []
    for u in p.keys():
        if u in n:
            out.append((np.asarray(p[u], dtype=np.int64), np.asarray(n[u], dtype=np.int64)))
    return out


def make_softmax_batches(gs, rng, train_scores=None, n_neg=6, semi_k=4, semi_frac=0.25, lambda_alpha=1.5):
    pos_all, neg_all, wt_all = [], [], []
    order = np.arange(len(gs)); rng.shuffle(order)
    disc5 = np.asarray([1/np.log2(i+2) for i in range(5)], dtype=np.float32)
    for gi in order:
        ps, ns = gs[gi]
        if len(ps) == 0 or len(ns) == 0:
            continue
        mpos = len(ps)
        negs = rng.choice(ns, size=(mpos, n_neg), replace=True).astype(np.int64)
        if train_scores is not None and semi_frac > 0 and semi_k > 1:
            mask = rng.random(mpos) < semi_frac
            hh = int(mask.sum())
            if hh:
                psel = ps[mask]
                cand = rng.choice(ns, size=(hh, semi_k), replace=True).astype(np.int64)
                cs = train_scores[cand]
                # Choose hard-but-not-higher negatives when available; otherwise leave uniform samples.
                ok = cs < train_scores[psel, None]
                any_ok = ok.any(1)
                if any_ok.any():
                    rows = np.where(any_ok)[0]
                    cols = np.argmax(np.where(ok, cs, -np.inf)[rows], axis=1)
                    tmp = negs[mask]
                    tmp[rows, 0] = cand[rows, cols]
                    negs[mask] = tmp
        weights = np.ones(mpos, dtype=np.float32)
        if train_scores is not None and lambda_alpha > 0:
            all_idx = np.concatenate([ps, ns])
            ord2 = np.argsort(-train_scores[all_idx], kind='mergesort')
            ranks = np.empty(len(all_idx), dtype=np.int32); ranks[ord2] = np.arange(1, len(all_idx)+1)
            rmap = dict(zip(all_idx.tolist(), ranks.tolist()))
            rp = np.fromiter((rmap[int(x)] for x in ps), dtype=np.int32, count=mpos)
            rn = np.fromiter((rmap[int(x)] for x in negs[:, 0]), dtype=np.int32, count=mpos)
            dp = np.where(rp <= 5, 1/np.log2(rp.astype(np.float32)+1), 0.0)
            dn = np.where(rn <= 5, 1/np.log2(rn.astype(np.float32)+1), 0.0)
            idcg = float(disc5[:min(len(ps), 5)].sum())
            if idcg > 0:
                weights = (1.0 + lambda_alpha * np.abs(dp - dn) / idcg).astype(np.float32)
        pos_all.append(ps); neg_all.append(negs); wt_all.append(weights)
    pos = np.concatenate(pos_all); neg = np.concatenate(neg_all); wt = np.concatenate(wt_all)
    perm = rng.permutation(len(pos))
    return pos[perm], neg[perm], wt[perm]


def run(splits, data_dir, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=2048, patience=4, device='cpu', verbose=True):
    enc0, dim0 = m.encode(splits)
    enc, dim, aux, cwm, dlogs, calfeats, active, has_play = m.augment(splits, enc0, dim0, data_dir)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    dtr = dlogs['train']; dva = dlogs['valid']
    atr, mtr = aux['train']; obs, cens, cmask = cwm['train']
    obs_thr = obs.copy(); obs_thr[cens > 0.5] = dtr[cens > 0.5]

    model = m.Model(dim, Xtr.shape[1], k=k, n_aux=len(active), use_cwm=has_play, seed=seed).to(device)
    decay = [model.V, model.W] + list(model.deep.parameters())
    nodecay = [model.b, model.deep_scale, model.cwm_scale]
    if len(active) > 0:
        decay += [model.aux_A, model.aux_W]; nodecay += [model.aux_b]
    if has_play:
        decay += [model.cwm_A, model.cwm_W]; nodecay += [model.cwm_b]
    opt = torch.optim.Adam([{'params': decay, 'weight_decay': l2}, {'params': nodecay, 'weight_decay': 0.0}], lr=lr)

    X_t = torch.from_numpy(Xtr.astype(np.int64)); y_t = torch.from_numpy(ytr.astype(np.float32)); d_t = torch.from_numpy(dtr.astype(np.float32))
    a_t = torch.from_numpy(atr.astype(np.float32)); ma_t = torch.from_numpy(mtr.astype(np.float32))
    o_t = torch.from_numpy(obs_thr.astype(np.float32)); c_t = torch.from_numpy(cens.astype(np.float32)); cm_t = torch.from_numpy(cmask.astype(np.float32))
    gs = group_pos_neg(utr, ytr)
    rng = np.random.default_rng(seed)
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1.0; best_state = None; bad = 0

    for ep in range(1, epochs+1):
        t0 = time.time(); model.train(); losses = []
        if ep <= 1:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), 8192):
                sel = torch.from_numpy(idx[i:i+8192])
                xb = X_t[sel].to(device); db = d_t[sel].to(device); yb = y_t[sel].to(device)
                opt.zero_grad(set_to_none=True)
                loss = bce(model(xb, db), yb)
                ml = m.multitask_loss(model, xb, a_t[sel].to(device), ma_t[sel].to(device), o_t[sel].to(device), c_t[sel].to(device), cm_t[sel].to(device))
                if ml is not None:
                    loss = loss + 0.05 * ml
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        else:
            train_scores = model.predict(Xtr, dtr, device=device)
            pidx, nmat, wt = make_softmax_batches(gs, rng, train_scores=train_scores, n_neg=6, semi_k=4, semi_frac=0.25, lambda_alpha=1.5)
            target = torch.zeros(bs, dtype=torch.long, device=device)
            for i in range(0, len(pidx), bs):
                ps_np = pidx[i:i+bs]; ns_np = nmat[i:i+bs]; w_np = wt[i:i+bs]
                cur = len(ps_np)
                cand_np = np.concatenate([ps_np[:, None], ns_np], axis=1)
                flat_np = cand_np.reshape(-1)
                xb = X_t[torch.from_numpy(flat_np)].to(device)
                db = d_t[torch.from_numpy(flat_np)].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb, db).reshape(cur, 1 + ns_np.shape[1])
                ce = torch.nn.functional.cross_entropy(logits, target[:cur], reduction='none')
                w = torch.from_numpy(w_np).to(device)
                loss = (ce * w).sum() / (w.sum() + 1e-8)
                # Small pointwise anchor over the sampled slate keeps the score scale calibrated.
                labs = torch.zeros_like(logits); labs[:, 0] = 1.0
                loss = loss + 0.10 * torch.nn.functional.binary_cross_entropy_with_logits(logits, labs)
                ml = m.multitask_loss(model, xb, a_t[torch.from_numpy(flat_np)].to(device), ma_t[torch.from_numpy(flat_np)].to(device), o_t[torch.from_numpy(flat_np)].to(device), c_t[torch.from_numpy(flat_np)].to(device), cm_t[torch.from_numpy(flat_np)].to(device))
                if ml is not None:
                    loss = loss + 0.05 * ml
                loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        va = m.evaluate(uva, yva, model.predict(Xva, dva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} sampled-softmax | scales d={float(model.deep_scale.detach().cpu()):+.3f} c={float(model.cwm_scale.detach().cpu()):+.3f} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    base_tr = model.predict(Xtr, dtr, device=device)
    resid = m.LambdaResidual(seed=seed, weight=0.35).fit(base_tr, ytr, utr, Xtr, calfeats['train'])
    return model, resid, enc, dlogs, calfeats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = m.load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={m.FIELDS}+time aux+CWM+sampled-softmax+LambdaMART-residual")
    model, resid, enc, dlogs, calfeats = run(splits, a.data_dir, seed=a.seed, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=a.out is None)
    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=a.device)
    scores = resid.predict(base, u, X, calfeats[target])
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        for sp in ('valid','test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                b = model.predict(Xs, dlogs[sp], device=a.device)
                print(sp, m.evaluate(us, ys, resid.predict(b, us, Xs, calfeats[sp])))
