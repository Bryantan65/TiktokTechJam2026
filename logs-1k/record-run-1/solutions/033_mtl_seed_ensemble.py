"""Seed ensemble of the best raw-feedback MTL FM.

Improve node 12: average three independently initialized copies of the same MTL
FM.  Node 12 is strong but still has seed spread; averaging full-strength
members should reduce variance and may improve within-user ranking.  Member
predictions are cached because changing blend logic should not retrain them.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402

AUX_BIN_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']


class MultiTaskFM(torch.nn.Module):
    def __init__(self, dim, n_tasks, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = torch.nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = torch.nn.Parameter(torch.zeros(n_tasks, dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))

    def forward_all(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W[:, X].sum(2).transpose(0, 1)
        return inter[:, None] + lin + self.b[None, :]

    def forward(self, X):
        return self.forward_all(X)[:, 0]

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        outs = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            outs.append(self(xb).cpu().numpy())
        return np.concatenate(outs)


def _int_val(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def _float_val(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _find_col(rec, names):
    for n in names:
        if n in rec:
            return n
    return None


def read_aux_lookup(data_dir):
    paths = [os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv'),
             os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')]
    if not all(os.path.isfile(p) for p in paths):
        return None, AUX_BIN_COLS + ['watch_ratio']
    q = defaultdict(deque)
    cols = None
    bad_header = False
    for path in paths:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                if cols is None:
                    date_c = _find_col(rec, ['date'])
                    user_c = _find_col(rec, ['user_id', 'user'])
                    video_c = _find_col(rec, ['video_id', 'video'])
                    author_c = _find_col(rec, ['author_id', 'author'])
                    tab_c = _find_col(rec, ['tab'])
                    dur_c = _find_col(rec, ['duration_ms', 'video_duration', 'duration'])
                    play_c = _find_col(rec, ['play_time_ms', 'playtime_ms', 'play_time'])
                    aux_cols = [_find_col(rec, [c]) for c in AUX_BIN_COLS]
                    cols = (date_c, user_c, video_c, author_c, tab_c, dur_c, play_c, aux_cols)
                    bad_header = any(c is None for c in [date_c, user_c, video_c, author_c, tab_c, dur_c])
                date_c, user_c, video_c, author_c, tab_c, dur_c, play_c, aux_cols = cols
                if bad_header:
                    continue
                dur = max(_int_val(rec[dur_c]), 1)
                vals = []
                for c in aux_cols:
                    vals.append(np.nan if c is None else float(_int_val(rec[c]) > 0))
                if play_c is None:
                    vals.append(np.nan)
                else:
                    vals.append(float(np.clip(_float_val(rec[play_c]) / dur, 0.0, 1.0)))
                key = (_int_val(rec[date_c]), str(rec[user_c]), str(rec[video_c]),
                       str(rec[author_c]), _int_val(rec[tab_c]), dur)
                q[key].append(np.asarray(vals, dtype=np.float32))
    return q, AUX_BIN_COLS + ['watch_ratio']


def attach_aux(splits, data_dir):
    lookup, names = read_aux_lookup(data_dir)
    n_aux = len(names)
    out = {}
    if lookup is None:
        for sp, rows in splits.items():
            out[sp] = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        return out, names
    miss = 0
    hit = 0
    for sp, rows in splits.items():
        arr = np.full((len(rows), n_aux), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            key = (_int_val(row[0]), str(row[1]), str(row[2]), str(row[3]),
                   _int_val(row[4]), max(_int_val(row[5]), 1))
            if lookup.get(key):
                arr[i] = lookup[key].popleft()
                hit += 1
            else:
                miss += 1
        out[sp] = arr
    print(f"raw auxiliaries joined for MTL seed ensemble; hit={hit} missing={miss}; aux={names}")
    return out, names


def train_one(enc, aux, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
              patience=4, seed=0, device='cpu', verbose=True, aux_weight=0.15):
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Yaux = aux['train']
    dim = int(max(max(v[0].max() for v in enc.values()), 0)) + 1

    model = MultiTaskFM(dim, n_tasks=1 + Yaux.shape[1], k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    aux_t = torch.from_numpy(Yaux.astype(np.float32))

    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        model.train()
        losses = []
        for i in range(0, len(idx), bs):
            sel = torch.from_numpy(idx[i:i + bs])
            xb = Xtr_t[sel].to(device)
            yb = ytr_t[sel].to(device)
            ab = aux_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model.forward_all(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], yb)
            aux_loss = 0.0
            aux_count = 0
            for j in range(ab.shape[1]):
                tgt = ab[:, j]
                mask = torch.isfinite(tgt)
                if bool(mask.any()):
                    aux_loss = aux_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[mask, j + 1], tgt[mask])
                    aux_count += 1
            if aux_count:
                loss = loss + aux_weight * aux_loss / aux_count
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"    seed {seed} epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop member seed {seed} at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model


def cache_tag(data_dir, split, target, n_rows, seed, member_seed):
    base = os.path.basename(os.path.dirname(os.path.abspath(data_dir))) + '_' + os.path.basename(os.path.abspath(data_dir))
    return f"033_mtl_seedens_{base}_{split}_{target}_n{n_rows}_master{seed}_m{member_seed}.npy"


def predict_ensemble(splits, data_dir, split_name, target, master_seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    aux, _ = attach_aux(splits, data_dir)
    Xtar, _, _ = enc[target]
    os.makedirs('pred_cache', exist_ok=True)

    # Three independent members.  The seed offsets make each harness seed test a
    # different deterministic 3-member bag while still honoring --seed.
    member_seeds = [master_seed * 100 + 17, master_seed * 100 + 29, master_seed * 100 + 43]
    preds = []
    for ms in member_seeds:
        path = os.path.join('pred_cache', cache_tag(data_dir, split_name, target, len(Xtar), master_seed, ms))
        if os.path.isfile(path):
            p = np.load(path)
            print(f"loaded cached member predictions {path}")
        else:
            print(f"training MTL member seed={ms}")
            torch.manual_seed(ms)
            model = train_one(enc, aux, seed=ms, device=device, verbose=verbose)
            p = model.predict(Xtar, device=device).astype(np.float64)
            np.save(path, p)
            print(f"saved cached member predictions {path}")
        preds.append(p.astype(np.float64))
    return np.mean(np.vstack(preds), axis=0), enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}+raw_aux_mtl_seed_ensemble")

    scores, enc = predict_ensemble(splits, data_dir=a.data_dir, split_name=a.split,
                                   target=target, master_seed=a.seed,
                                   device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        X, y, users = enc[target]
        r = evaluate(users, y, scores)
        print(f"\n=== mtl_seed_ensemble (seed={a.seed}, device={a.device}) ===")
        print(f"  {target:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
