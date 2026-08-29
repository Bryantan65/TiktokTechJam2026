"""Recency-weighted node-20 stack.

Node 10 added hour/day context, but all train days were still weighted equally.
This variant keeps the node-20 architecture/residual and gives later train dates
larger loss weight, testing train->valid temporal drift as a different time
mechanism rather than another feature tweak.
"""
import argparse, importlib.util, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, '020_lambdamart_residual.py')
spec = importlib.util.spec_from_file_location('node20', PARENT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def date_weights(train_rows, half_life_days=8.0):
    dates = np.asarray([int(r[0]) for r in train_rows], dtype=np.int64)
    uniq = np.sort(np.unique(dates))
    rank = {int(d): i for i, d in enumerate(uniq.tolist())}
    rr = np.asarray([rank[int(d)] for d in dates], dtype=np.float32)
    mx = float(rr.max()) if len(rr) else 0.0
    raw = np.exp((rr - mx) / float(half_life_days)).astype(np.float32)
    raw = np.clip(raw, 0.25, 3.0)
    raw /= float(raw.mean()) + 1e-8
    return raw.astype(np.float32)


def run(splits, data_dir, seed=0, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, device='cpu', verbose=True):
    enc0, dim0 = m.encode(splits)
    enc, dim, aux, cwm, dlogs, calfeats, active, has_play = m.augment(splits, enc0, dim0, data_dir)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    dtr = dlogs['train']; dva = dlogs['valid']
    atr, mtr = aux['train']
    obs, cens, cmask = cwm['train']
    obs_thr = obs.copy(); obs_thr[cens > 0.5] = dtr[cens > 0.5]
    rw_np = date_weights(splits['train'], half_life_days=8.0)
    if verbose:
        ds = np.asarray([int(r[0]) for r in splits['train']])
        print('recency weights:', [(int(d), float(rw_np[ds == d].mean())) for d in np.sort(np.unique(ds))])

    model = m.Model(dim, Xtr.shape[1], k=k, n_aux=len(active), use_cwm=has_play, seed=seed).to(device)
    decay = [model.V, model.W] + list(model.deep.parameters())
    nodecay = [model.b, model.deep_scale, model.cwm_scale]
    if len(active) > 0:
        decay += [model.aux_A, model.aux_W]; nodecay += [model.aux_b]
    if has_play:
        decay += [model.cwm_A, model.cwm_W]; nodecay += [model.cwm_b]
    opt = torch.optim.Adam([{'params': decay, 'weight_decay': l2}, {'params': nodecay, 'weight_decay': 0.0}], lr=lr)

    X_t = torch.from_numpy(Xtr.astype(np.int64)); y_t = torch.from_numpy(ytr.astype(np.float32)); d_t = torch.from_numpy(dtr.astype(np.float32))
    rw_t = torch.from_numpy(rw_np.astype(np.float32))
    a_t = torch.from_numpy(atr.astype(np.float32)); ma_t = torch.from_numpy(mtr.astype(np.float32))
    o_t = torch.from_numpy(obs_thr.astype(np.float32)); c_t = torch.from_numpy(cens.astype(np.float32)); cm_t = torch.from_numpy(cmask.astype(np.float32))
    gs = m.groups(utr, ytr); rng = np.random.default_rng(seed)
    bce_none = torch.nn.BCEWithLogitsLoss(reduction='none')
    bce = torch.nn.BCEWithLogitsLoss()
    best = -1.0; best_state = None; bad = 0

    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); losses = []
        if ep <= 1:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), bs):
                sel_np = idx[i:i+bs]; sel = torch.from_numpy(sel_np)
                xb = X_t[sel].to(device); db = d_t[sel].to(device); yb = y_t[sel].to(device); wb = rw_t[sel].to(device)
                opt.zero_grad(set_to_none=True)
                loss = (bce_none(model(xb, db), yb) * wb).sum() / (wb.sum() + 1e-8)
                ml = m.multitask_loss(model, xb, a_t[sel].to(device), ma_t[sel].to(device), o_t[sel].to(device), c_t[sel].to(device), cm_t[sel].to(device))
                if ml is not None: loss = loss + 0.05 * ml
                loss.backward(); opt.step(); losses.append(float(loss.item()))
        else:
            train_scores = model.predict(Xtr, dtr, device=device)
            pidx, nidx, w = m.make_pairs(gs, rng, train_scores=train_scores)
            w = (w.astype(np.float32) * rw_np[pidx].astype(np.float32))
            for i in range(0, len(pidx), bs):
                ps_np = pidx[i:i+bs]; ns_np = nidx[i:i+bs]
                ps = torch.from_numpy(ps_np); ns = torch.from_numpy(ns_np); wt = torch.from_numpy(w[i:i+bs]).to(device)
                xp = X_t[ps].to(device); xn = X_t[ns].to(device); dp = d_t[ps].to(device); dn = d_t[ns].to(device)
                opt.zero_grad(set_to_none=True)
                sp = model(xp, dp); sn = model(xn, dn)
                pair = (torch.nn.functional.softplus(-(sp - sn)) * wt).sum() / (wt.sum() + 1e-8)
                # Keep the parent BCE anchor unweighted on the balanced pair rows; only the ranking term tests recency.
                loss = pair + 0.15 * bce(torch.cat([sp, sn]), torch.cat([torch.ones_like(sp), torch.zeros_like(sn)]))
                both_np = np.concatenate([ps_np, ns_np]); both = torch.from_numpy(both_np); xb = X_t[both].to(device)
                ml = m.multitask_loss(model, xb, a_t[both].to(device), ma_t[both].to(device), o_t[both].to(device), c_t[both].to(device), cm_t[both].to(device))
                if ml is not None: loss = loss + 0.05 * ml
                loss.backward(); opt.step(); losses.append(float(loss.item()))
        va = m.evaluate(uva, yva, model.predict(Xva, dva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} recw | scales d={float(model.deep_scale.detach().cpu()):+.3f} c={float(model.cwm_scale.detach().cpu()):+.3f} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
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
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = m.load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, f"fields={m.FIELDS}+time aux+CWM+LambdaMART-residual+recency-weight")
    model, resid, enc, dlogs, calfeats = run(splits, a.data_dir, seed=a.seed, k=a.k, lr=a.lr, epochs=a.epochs, device=a.device, verbose=a.out is None)
    X, y, u = enc[target]
    base = model.predict(X, dlogs[target], device=a.device)
    scores = resid.predict(base, u, X, calfeats[target])
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                b = model.predict(Xs, dlogs[sp], device=a.device)
                print(sp, m.evaluate(us, ys, resid.predict(b, us, Xs, calfeats[sp])))
