"""Draft direction 3: auxiliary-feedback multi-task regularisation for the sequence FM.

Following MMoE/MTL recommender motivation (Google MMoE paper:
https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/),
keep the strong node22 ensemble unchanged and add a readable 30% member whose
main objective is same-user BPR but whose shared FM embeddings are also trained
with auxiliary feedback labels from the raw logs (like/follow/comment/forward
and a play-ratio task when available).  If raw aux columns are unavailable the
member falls back to ordinary BPR, so the script remains standalone.
"""
import argparse, csv, os, sys, time
from collections import defaultdict, deque
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'kuairand-starter-kit'))
import importlib.util
spec22 = importlib.util.spec_from_file_location('node22', os.path.join(HERE, '022_seq_context_fm.py'))
node22 = importlib.util.module_from_spec(spec22); spec22.loader.exec_module(node22)
node20 = node22.node20
from data import load, encode
from evaluate import evaluate


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, ntasks, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W_main = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b_main = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.ntasks = int(ntasks)
        if self.ntasks > 0:
            self.W_aux = torch.nn.Parameter(torch.zeros((self.ntasks, dim), dtype=torch.float32))
            self.b_aux = torch.nn.Parameter(torch.zeros(self.ntasks, dtype=torch.float32))
        else:
            self.W_aux = None; self.b_aux = None

    def shared_inter(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def forward_main(self, X):
        return self.b_main + self.W_main[X].sum(1) + self.shared_inter(X)

    def forward_aux(self, X):
        inter = self.shared_inter(X).reshape(-1, 1)
        lin = self.W_aux[:, X].sum(2).transpose(0, 1)
        return inter + lin + self.b_aux.reshape(1, -1)

    @torch.no_grad()
    def predict(self, X, bs=200000, device='cpu'):
        self.eval(); out=[]
        for i in range(0, len(X), bs):
            out.append(self.forward_main(torch.from_numpy(X[i:i+bs].astype(np.int64)).to(device)).cpu().numpy())
        return np.concatenate(out)


def make_pair_sampler(y, users):
    pos, neg = defaultdict(list), defaultdict(list)
    for i, (yy, u) in enumerate(zip(y, users)):
        (pos if yy > 0.5 else neg)[u].append(i)
    pidx=[]; pusers=[]; neg_by={}
    for u, ps in pos.items():
        ns = neg.get(u)
        if ns:
            neg_by[u] = np.asarray(ns, dtype=np.int64)
            pidx.extend(ps); pusers.extend([u] * len(ps))
    return np.asarray(pidx, dtype=np.int64), np.asarray(pusers, dtype=object), neg_by


def row_get(row, names, default=''):
    for n in names:
        if n in row and row[n] not in ('', 'None', 'nan'):
            return row[n]
    return default


def row_float(row, names, default=0.0):
    try: return float(row_get(row, names, ''))
    except Exception: return default


def row_bin(row, names):
    try: return 1.0 if float(row_get(row, names, '0')) > 0 else 0.0
    except Exception: return 0.0


def load_aux_targets(data_dir, splits, verbose=False):
    n = len(splits['train'])
    paths = node20.find_log_files(data_dir)
    if paths is None:
        if verbose: print('aux mtl: raw log csvs not found, no aux tasks')
        return np.zeros((n, 0), dtype=np.float32), []
    full_q, short_q = defaultdict(deque), defaultdict(deque)
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                full, short = node20.make_keys_from_csv(row)
                if full is not None: full_q[full].append(row)
                short_q[short].append(row)
    cols = [
        ('like', ['is_like','like']),
        ('follow', ['is_follow','follow']),
        ('comment', ['is_comment','comment']),
        ('forward', ['is_forward','forward','is_share','share']),
        ('profile', ['is_profile_enter','profile_enter','is_enter_profile']),
    ]
    vals = np.zeros((n, len(cols) + 1), dtype=np.float32)
    miss = 0; seen_cols = set()
    for i, r in enumerate(splits['train']):
        rec = None
        fq = full_q.get(node20.make_key_from_tuple(r, True))
        if fq: rec = fq.popleft()
        else:
            sq = short_q.get(node20.make_key_from_tuple(r, False))
            if sq: rec = sq.popleft()
        if rec is None:
            miss += 1; continue
        for j, (name, names) in enumerate(cols):
            if any(c in rec for c in names):
                seen_cols.add(name)
                vals[i, j] = row_bin(rec, names)
        play = row_float(rec, ['play_time_ms','play_time','watch_time_ms'], -1.0)
        try: dur = float(r[5])
        except Exception: dur = 0.0
        if play >= 0 and dur > 0:
            seen_cols.add('play50')
            vals[i, len(cols)] = 1.0 if (play / max(dur, 1.0)) >= 0.50 else 0.0
    names = [c[0] for c in cols] + ['play50']
    keep=[]; keep_names=[]
    for j, name in enumerate(names):
        m = float(vals[:, j].mean())
        # Drop absent/constant tasks; rare positive tasks are retained with pos_weight.
        if name in seen_cols and 0.00005 < m < 0.99995:
            keep.append(j); keep_names.append(name)
    if verbose:
        means = {names[j]: float(vals[:, j].mean()) for j in keep}
        print(f'aux mtl: missing={miss}/{n} tasks={keep_names} means={means}')
    if not keep:
        return np.zeros((n, 0), dtype=np.float32), []
    return vals[:, keep].astype(np.float32), keep_names


def train_mtl_member(enc, dim, aux, k=16, lr=0.001, l2=1e-6, epochs=50, bs=8192, patience=5, seed=0, device='cpu', verbose=True, nneg=3, aux_lambda=0.05):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    torch.manual_seed(seed)
    nt = aux.shape[1] if aux is not None else 0
    model = MultiTaskFM(dim, nt, k, seed).to(device)
    params = [{'params':[model.V, model.W_main], 'weight_decay':l2}, {'params':[model.b_main], 'weight_decay':0.0}]
    if nt > 0:
        params.append({'params':[model.W_aux], 'weight_decay':l2}); params.append({'params':[model.b_aux], 'weight_decay':0.0})
    opt = torch.optim.Adam(params, lr=lr)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    aux_t = None if nt == 0 else torch.from_numpy(aux.astype(np.float32))
    if nt > 0:
        means = np.clip(aux.mean(axis=0), 1e-5, 1-1e-5)
        posw = np.clip((1.0 - means) / means, 1.0, 50.0).astype(np.float32)
        posw_t = torch.from_numpy(posw).to(device)
    else:
        posw_t = None
    pos_idx, pos_users, neg_by = make_pair_sampler(ytr, utr)
    if verbose: print(f'mtl member seed={seed}: positives {len(pos_idx):,d}; aux_tasks={nt}; nneg={nneg}')
    rng = np.random.default_rng(seed); best = -1.0; best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(pos_idx)); model.train(); losses=[]; t0=time.time()
        for i in range(0, len(order), bs):
            sel = order[i:i+bs]; pidx = pos_idx[sel]; nidx = np.empty((len(sel), nneg), dtype=np.int64)
            for j, u in enumerate(pos_users[sel]):
                ns = neg_by[u]; nidx[j] = ns[rng.integers(len(ns), size=nneg)]
            xp = Xtr_t[torch.from_numpy(pidx)].to(device)
            xn = Xtr_t[torch.from_numpy(nidx.reshape(-1))].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model.forward_main(xp); sn = model.forward_main(xn).reshape(len(sel), nneg)
            loss = torch.nn.functional.softplus(-(sp.repeat_interleave(nneg) - sn.reshape(-1))).mean()
            if nt > 0 and aux_lambda > 0:
                # Use positives and sampled negatives for the auxiliary BCE so every update
                # regularises the same user-local comparisons as the ranking loss.
                aidx = np.concatenate([pidx, nidx.reshape(-1)])
                xa = Xtr_t[torch.from_numpy(aidx)].to(device)
                ya = aux_t[torch.from_numpy(aidx)].to(device)
                la = torch.nn.functional.binary_cross_entropy_with_logits(model.forward_aux(xa), ya, pos_weight=posw_t)
                loss = loss + aux_lambda * la
            loss.backward(); opt.step(); losses.append(float(loss.item()))
        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose: print(f'  mtl {seed} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va["primary"]:.4f} | {time.time()-t0:.1f}s')
        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model


def cached_mtl_predictions(enc, dim, aux, target, split_name, seed, device, verbose, k=16, lr=0.001, epochs=50):
    os.makedirs('pred_cache', exist_ok=True)
    # Cache name encodes formulation and lambda; change it if training changes.
    path = os.path.join('pred_cache', f'024_seq_mtl_aux005_member_seed{seed}_{split_name}_{target}.npy')
    if os.path.isfile(path):
        if verbose: print(f'loading cached member: {path}')
        return np.load(path)
    model = train_mtl_member(enc, dim, aux, k=k, lr=lr, epochs=epochs, seed=seed, device=device, verbose=verbose)
    preds = model.predict(enc[target][0], device=device).astype(np.float64)
    np.save(path, preds)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args(); torch.manual_seed(a.seed)
    print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'
    else:
        splits = load(a.data_dir); target = a.split
    print({k: len(v) for k, v in splits.items()}, 'fields=node22+seq_mtl_aux')
    base = node22.main_base if False else None
    # Reproduce node22 exactly for the incumbent prediction.
    base = node22.node20.run_solution(splits, target, a.split, a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, device=a.device, verbose=a.out is None)
    enc0, _ = encode(splits); users = enc0[target][2]
    senc, sdim = node22.build_seq_rows(splits)
    member_seeds = [a.seed + 1000 * m for m in range(3)]
    seq_members = []
    for ms in member_seeds:
        p = node22.node20.cached_predictions(senc, sdim, target, a.split, ms, a.device, a.out is None,
                                             '022_seq_context_bpr_member', 'bpr', None, 3, a.k, a.lr, a.epochs)
        seq_members.append(node22.per_user_zscore(p, users))
    seq = node22.per_user_zscore(np.mean(np.vstack(seq_members), axis=0), users)
    incumbent = 0.70 * node22.per_user_zscore(base, users) + 0.30 * seq

    aux, aux_names = load_aux_targets(a.data_dir, splits, verbose=a.out is None)
    mtl_members = []
    for ms in member_seeds:
        p = cached_mtl_predictions(senc, sdim, aux, target, a.split, ms, a.device, a.out is None, k=a.k, lr=a.lr, epochs=a.epochs)
        mtl_members.append(node22.per_user_zscore(p, users))
    mtl = node22.per_user_zscore(np.mean(np.vstack(mtl_members), axis=0), users)
    scores = 0.70 * node22.per_user_zscore(incumbent, users) + 0.30 * mtl
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(f'produced {len(scores):,d} predictions for split={a.split}; aux={aux_names}')

if __name__ == '__main__':
    main()
