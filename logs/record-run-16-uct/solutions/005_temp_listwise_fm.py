"""FM trained with temperature-scaled same-user listwise softmax loss.

This refines the full same-user listwise objective by using a low softmax
temperature.  The denominator is still all impressions from the same user, but
softmax(s / T) concentrates gradient on high-scoring hard negatives, which is
closer to top-of-list metrics such as nDCG@5 than a uniform full-list CE.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kuairand-starter-kit'))
from data import load, encode, FIELDS          # noqa: E402
from evaluate import evaluate                  # noqa: E402


class TorchFM(torch.nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = torch.nn.Parameter(torch.from_numpy(V0))
        self.W = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000, device='cpu'):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            out.append(self(xb).cpu().numpy())
        return np.concatenate(out)


def make_user_groups(y, users):
    by_user = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, []).append(i)
    groups = []
    npos_total = 0
    for idxs in by_user.values():
        idx = np.asarray(idxs, dtype=np.int64)
        yy = y[idx]
        npos = int((yy > 0.5).sum())
        nneg = len(idx) - npos
        if npos > 0 and nneg > 0:
            pos_mask = (yy > 0.5).astype(np.float32)
            groups.append((idx, pos_mask, npos))
            npos_total += npos
    return groups, npos_total


def run(splits, k=16, lr=0.001, l2=1e-6, epochs=40, users_per_batch=256,
        temperature=0.5, patience=4, seed=0, device='cpu', verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    groups, npos_total = make_user_groups(ytr, utr)
    if len(groups) == 0:
        raise RuntimeError('no train users with both positive and negative rows')
    if verbose:
        rows_in_groups = sum(len(g[0]) for g in groups)
        print(f"Temp-listwise groups: {len(groups):,d} users, {rows_in_groups:,d} rows, "
              f"{npos_total:,d} positives, T={temperature}")

    model = TorchFM(dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam([{'params': [model.V, model.W], 'weight_decay': l2},
                            {'params': [model.b], 'weight_decay': 0.0}],
                           lr=lr, betas=(0.9, 0.999), eps=1e-8)

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    temp = float(temperature)

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(groups))
        t0 = time.time()
        model.train()
        losses = []
        for st in range(0, len(order), users_per_batch):
            batch_ids = order[st:st + users_per_batch]
            idx_parts = []
            pos_parts = []
            lengths = []
            npos_parts = []
            for gi in batch_ids:
                idx, pos_mask, npos = groups[int(gi)]
                idx_parts.append(idx)
                pos_parts.append(pos_mask)
                lengths.append(len(idx))
                npos_parts.append(npos)
            idx_cat = np.concatenate(idx_parts)
            pos_cat = np.concatenate(pos_parts)
            xb = Xtr_t[torch.from_numpy(idx_cat)].to(device)
            pos_t = torch.from_numpy(pos_cat).to(device)

            opt.zero_grad(set_to_none=True)
            scores = model(xb)
            loss_num = scores.new_tensor(0.0)
            off = 0
            for ln, npos in zip(lengths, npos_parts):
                s = scores[off:off + ln]
                pm = pos_t[off:off + ln]
                # T * logsumexp(s/T) keeps the loss in score units while
                # concentrating negative gradients on high-scoring items.
                loss_num = loss_num + float(npos) * temp * torch.logsumexp(s / temp, dim=0) - (s * pm).sum()
                off += ln
            loss = loss_num / float(sum(npos_parts))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva, device=device))
        if verbose:
            print(f"  epoch {ep:2d} | tempListCE {np.mean(losses):.4f} | valid "
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
    ap.add_argument('--users_per_batch', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.5)
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
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    model, enc = run(splits, k=a.k, lr=a.lr, epochs=a.epochs,
                     users_per_batch=a.users_per_batch,
                     temperature=a.temperature, seed=a.seed,
                     device=a.device, verbose=a.out is None)
    X, y, users = enc[target]
    scores = model.predict(X, device=a.device)

    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f"wrote {len(scores):,d} predictions for split={a.split}")
    else:
        print(f"\n=== temp_listwise_fm (seed={a.seed}, device={a.device}) ===")
        for sp in ('valid', 'test'):
            if sp in enc:
                Xs, ys, us = enc[sp]
                r = evaluate(us, ys, model.predict(Xs, device=a.device))
                print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
                      f"| primary {r['primary']:.4f}")
