import argparse, importlib.util, os, sys
from collections import defaultdict
import numpy as np
import torch

# Reuse the complete standalone training/fusion implementation from node 30.
# If caches are absent, its member_pred functions still train the requested members from scratch.
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '030_fixed_seed27_export.py')
spec = importlib.util.spec_from_file_location('base030', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def _clip_logit(p):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def stat_score(train_keys, y, target_keys, alpha=30.0):
    y = np.asarray(y, dtype=np.float64)
    gmean = float(y.mean()) if len(y) else 0.5
    cnt = defaultdict(int); pos = defaultdict(float)
    for k, yy in zip(train_keys, y):
        cnt[k] += 1; pos[k] += float(yy)
    g = _clip_logit(gmean)
    out = np.empty(len(target_keys), dtype=np.float64)
    for i, k in enumerate(target_keys):
        c = cnt.get(k, 0)
        if c:
            p = (pos[k] + alpha * gmean) / (c + alpha)
            out[i] = _clip_logit(p) - g
        else:
            out[i] = 0.0
    return out


def make_stat_member(enc, target, users):
    Xtr, ytr, _, _ = enc['train']
    Xta = enc[target][0]
    specs = [
        (lambda X: X[:, 1], 0.22, 40.0),                 # video
        (lambda X: X[:, 2], 0.18, 35.0),                 # author
        (lambda X: X[:, 3], 0.10, 80.0),                 # tab
        (lambda X: X[:, 4], 0.06, 80.0),                 # duration bucket
        (lambda X: X[:, 5], 0.05, 80.0),                 # date
        (lambda X: X[:, 7], 0.04, 80.0),                 # hour
        (lambda X: list(zip(X[:, 3], X[:, 7])), 0.09, 50.0),   # tab-hour
        (lambda X: list(zip(X[:, 0], X[:, 2])), 0.13, 20.0),   # user-author
        (lambda X: list(zip(X[:, 0], X[:, 3])), 0.08, 30.0),   # user-tab
        (lambda X: list(zip(X[:, 2], X[:, 3])), 0.05, 30.0),   # author-tab
    ]
    parts = []
    for fn, w, a in specs:
        s = stat_score(fn(Xtr), ytr, fn(Xta), alpha=a)
        parts.append(w * base.z_by_user(s, users))
    return base.z_by_user(np.sum(parts, axis=0), users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'; split_name = 'dev'
    else:
        from data import load
        splits = load(a.data_dir); target = a.split; split_name = a.split
    enc, dim = base.build_augmented(splits, a.data_dir)
    hist = base.build_hist(enc, K=20)
    users = enc[target][2]
    seeds5 = [2, 103, 204, 305, 406]
    old = [base.member_pred(enc, dim, target, s, split_name, a.device, 'bpr') for s in seeds5]
    bal = [base.member_pred(enc, dim, target, s, split_name, a.device, 'balanced') for s in seeds5]
    mt = [base.member_pred(enc, dim, target, s, split_name, a.device, 'mtbpr') for s in seeds5]
    parent = base.z_by_user(0.70 * base.z_by_user(0.65 * base.composite(old, users) + 0.35 * base.composite(bal, users), users) + 0.30 * base.composite(mt, users), users)
    din = [base.member_pred(enc, dim, target, s, split_name, a.device, 'din', hist) for s in seeds5]
    dinc = base.composite(din, users)
    hv, _ = hist[target]
    hlen = (hv >= 0).sum(1).astype(np.float64)
    alpha = 0.10 + 0.25 * np.minimum(hlen, 8.0) / 8.0
    strong = base.z_by_user((1.0 - alpha) * parent + alpha * dinc, users)
    stat = make_stat_member(enc, target, users)
    # Train-only empirical rates are mostly redundant with FM embeddings, so use them only as a small
    # tie-breaker after per-user normalization.
    scores = base.z_by_user(0.91 * strong + 0.09 * stat, users)
    print('fixed_seed27_plus_statprior stat_weight=0.09', flush=True)
    np.save(a.out, scores.astype(np.float64))


if __name__ == '__main__':
    main()
