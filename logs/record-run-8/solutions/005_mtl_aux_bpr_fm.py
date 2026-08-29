"""Multi-task FM: primary within-user BPR+BCE plus auxiliary feedback heads.

This drafts the multi-task direction from the current best BPR+BCE FM.  The
primary head is unchanged for prediction.  Additional heads share the FM
interaction embeddings and learn raw KuaiRand feedback labels (click/like/follow/
comment/forward) read from the original CSVs, acting as an auxiliary regularizer
for the shared representation.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


AUX_CANDIDATES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
LOG_FILES = [
    'log_standard_4_08_to_4_21_pure.csv',
    'log_standard_4_22_to_5_08_pure.csv',
]


def _first_key(row, names):
    for n in names:
        if n in row:
            return row[n]
    return None


def _to_int(x, default=0):
    try:
        if x is None or x == '':
            return default
        return int(float(x))
    except Exception:
        return default


def row_key_from_tuple(r):
    # data.load rows: (date, user_id, video_id, author_id, tab, duration_ms, label)
    return (_to_int(r[0]), str(r[1]), str(r[2]), str(r[3]), _to_int(r[4]), _to_int(r[5]))


def row_key_from_csv(row):
    date = _to_int(_first_key(row, ['date', 'request_date', 'day']))
    user = str(_first_key(row, ['user_id', 'userId', 'user']) or '')
    video = str(_first_key(row, ['video_id', 'photo_id', 'item_id', 'videoId']) or '')
    author = str(_first_key(row, ['author_id', 'author', 'authorId']) or '')
    tab = _to_int(_first_key(row, ['tab', 'tab_id']))
    dur = _to_int(_first_key(row, ['duration_ms', 'video_duration', 'duration', 'play_duration_ms']))
    return (date, user, video, author, tab, dur)


def load_auxiliary(data_dir, splits):
    """Return train auxiliary label matrix aligned to splits['train']."""
    rows_by_key = defaultdict(deque)
    fields_found = None
    for fn in LOG_FILES:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            # Some harnesses pass the KuaiRand-Pure root rather than its data dir.
            path = os.path.join(data_dir, 'data', fn)
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fields_found is None:
                fields_found = [c for c in AUX_CANDIDATES if c in (reader.fieldnames or [])]
            aux_cols = [c for c in AUX_CANDIDATES if c in (reader.fieldnames or [])]
            for row in reader:
                key = row_key_from_csv(row)
                vals = [_to_int(row.get(c, 0)) for c in aux_cols]
                rows_by_key[key].append(vals)

    aux_cols = fields_found or []
    if not aux_cols:
        return np.zeros((len(splits['train']), 0), dtype=np.float32), []

    aligned = {}
    misses = 0
    for sp in [s for s in ('train', 'valid', 'test') if s in splits]:
        arr = np.zeros((len(splits[sp]), len(aux_cols)), dtype=np.float32)
        for i, r in enumerate(splits[sp]):
            q = rows_by_key.get(row_key_from_tuple(r))
            if q:
                vals = q.popleft()
                # If a later CSV had a different available-column list, keep the
                # common leading fields from AUX_CANDIDATES; in the KuaiRand files
                # the schemas are normally identical.
                arr[i, :min(len(vals), len(aux_cols))] = vals[:len(aux_cols)]
            else:
                misses += 1
        aligned[sp] = arr
    if misses:
        print(f"auxiliary CSV alignment misses: {misses:,d}")
    return aligned['train'], aux_cols


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, num_tasks, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(num_tasks, dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        self.alpha = torch.nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))

    def interaction(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def forward_task(self, X, task=0):
        inter = self.interaction(X)
        return self.b[task] + self.W[task, X].sum(1) + self.alpha[task] * inter

    def forward_all(self, X):
        inter = self.interaction(X)
        # self.W[:, X] has shape (tasks, batch, fields).
        lin = self.W[:, X].sum(2).transpose(0, 1)
        return self.b.view(1, -1) + lin + inter.view(-1, 1) * self.alpha.view(1, -1)

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self.forward_task(xb, 0).cpu().numpy())
        return np.concatenate(out)


def build_user_groups(y, users):
    pos = defaultdict(list)
    neg = defaultdict(list)
    for i, (yy, uu) in enumerate(zip(y, users)):
        if yy > 0.5:
            pos[uu].append(i)
        else:
            neg[uu].append(i)
    groups = []
    for uu, p in pos.items():
        n = neg.get(uu)
        if n:
            groups.append((np.asarray(p, dtype=np.int64), np.asarray(n, dtype=np.int64)))
    return groups


def sample_pairs(groups, rng):
    pos_parts = []
    neg_parts = []
    for p, n in groups:
        pos_parts.append(p)
        neg_parts.append(rng.choice(n, size=len(p), replace=True))
    pi = np.concatenate(pos_parts)
    ni = np.concatenate(neg_parts)
    order = rng.permutation(len(pi))
    return pi[order], ni[order]


def run(splits, data_dir, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
        patience=4, seed=0, device='cpu', verbose=True, bce_weight=0.15,
        aux_weight=0.03):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    aux_train, aux_cols = load_auxiliary(data_dir, splits)
    num_aux = aux_train.shape[1]
    num_tasks = 1 + num_aux

    groups = build_user_groups(ytr, utr)
    if verbose:
        n_pairs = sum(len(p) for p, _ in groups)
        rates = aux_train.mean(0).round(4).tolist() if num_aux else []
        print(f"BPR eligible users={len(groups):,d}, sampled pairs/epoch={n_pairs:,d}, "
              f"bce_weight={bce_weight}, aux_weight={aux_weight}, aux={aux_cols}, rates={rates}")

    model = MultiTaskFM(dim, num_tasks=num_tasks, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W, model.alpha], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    aux_t = torch.from_numpy(aux_train.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        model.train()
        losses = []
        bprs = []
        bces = []
        auxes = []
        for i in range(0, len(pos_idx), bs):
            psel_np = pos_idx[i:i + bs]
            nsel_np = neg_idx[i:i + bs]
            psel = torch.from_numpy(psel_np)
            nsel = torch.from_numpy(nsel_np)
            xp = Xtr_t[psel].to(device)
            xn = Xtr_t[nsel].to(device)
            opt.zero_grad(set_to_none=True)
            sp = model.forward_task(xp, 0)
            sn = model.forward_task(xn, 0)
            bpr = -F.logsigmoid(sp - sn).mean()
            bce = 0.5 * (F.softplus(-sp).mean() + F.softplus(sn).mean())
            loss = bpr + bce_weight * bce
            aux_loss = torch.zeros((), device=device)
            if num_aux:
                both_idx = np.concatenate([psel_np, nsel_np])
                rows = torch.from_numpy(both_idx)
                xb = Xtr_t[rows].to(device)
                yaux = aux_t[rows].to(device)
                aux_logits = model.forward_all(xb)[:, 1:]
                aux_loss = F.binary_cross_entropy_with_logits(aux_logits, yaux)
                loss = loss + aux_weight * aux_loss
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bprs.append(bpr.item())
            bces.append(bce.item())
            auxes.append(aux_loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} bpr {np.mean(bprs):.4f} "
                  f"bce {np.mean(bces):.4f} aux {np.mean(auxes):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, enc


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
    ap.add_argument('--bce_weight', type=float, default=0.15)
    ap.add_argument('--aux_weight', type=float, default=0.03)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(f"loading {a.data_dir} ...")
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir)
        target = 'valid'
    else:
        splits = load(a.data_dir)
        target = a.split
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, data_dir=a.data_dir, k=a.k, lr=a.lr,
                     epochs=a.epochs, seed=a.seed, device=a.device,
                     verbose=a.out is None, bce_weight=a.bce_weight,
                     aux_weight=a.aux_weight)

    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== mtl_aux_bpr_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
