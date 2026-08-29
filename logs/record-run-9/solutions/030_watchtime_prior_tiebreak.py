"""Draft watch-time modelling as a post-hoc train-only prior.

This keeps the current best fixed-seed node-28 model unchanged, and adds a small
within-user rank tie-breaker based on train-only smoothed watch-time/completion
histories read from the raw logs.  No valid/test play_time values are used; raw
logs are only used to obtain auxiliary watch-time targets for train rows.
"""
import argparse
import csv
import importlib.util
import os
from collections import defaultdict, deque
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('node28_impl', os.path.join(_here, '028_seed0_history_tiebreak.py'))
impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(impl)


def _num(rec, names, default=0.0):
    for n in names:
        if n in rec and rec[n] not in (None, ''):
            try:
                return float(rec[n])
            except Exception:
                pass
    return default


def read_raw_aux_queues(data_dir):
    """Queues raw auxiliary watch time in original file order, keyed by tuple fields."""
    qs = defaultdict(deque)
    for fn in ['log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv']:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                date = int(_num(rec, ['date']))
                u = int(_num(rec, ['user_id', 'user']))
                v = int(_num(rec, ['video_id', 'photo_id', 'video']))
                a = int(_num(rec, ['author_id']))
                tab = int(_num(rec, ['tab']))
                dur = int(_num(rec, ['duration_ms', 'duration']))
                play = _num(rec, ['play_time_ms', 'play_time', 'watch_time_ms', 'time_ms'], 0.0)
                key = (date, u, v, a, tab, dur)
                qs[key].append(play)
    return qs


def _mean(sumv, cnt, prior=0.38, alpha=5.0):
    return (sumv + alpha * prior) / (cnt + alpha)


def _cap_ratio(play_ms, dur_ms):
    try:
        d = float(dur_ms)
    except Exception:
        d = 0.0
    if d <= 0:
        return 0.0
    # Cap replays/idle playback; completion-like ratio is usually the least noisy watch target.
    return float(max(0.0, min(2.0, play_ms / d))) / 2.0


def watchtime_prior(splits, data_dir, split):
    qs = read_raw_aux_queues(data_dir)
    # Sum of capped watch ratios from train only.
    sv = defaultdict(float); cv = defaultdict(int)
    sa = defaultdict(float); ca = defaultdict(int)
    su = defaultdict(float); cu = defaultdict(int)
    suv = defaultdict(float); cuv = defaultdict(int)
    sua = defaultdict(float); cua = defaultdict(int)
    stab = defaultdict(float); ctab = defaultdict(int)
    sdur = defaultdict(float); cdur = defaultdict(int)
    out = None
    for sp in ('train', 'valid', 'test'):
        vals = []
        for row in splits[sp]:
            date, u, v, a, tab, dur, y = row
            key = (int(date), int(u), int(v), int(a), int(tab), int(dur))
            play = qs[key].popleft() if key in qs and len(qs[key]) else 0.0
            d = int(dur) // 10000
            kv = (u, v); ka = (u, a); kt = (u, tab); kd = (u, d)
            # Predict completion/watch satisfaction using user-specific histories, backed off to
            # global video/author/user/context watch-time means learned from train rows only.
            ruv = _mean(suv[kv], cuv[kv], 0.38, 4.0)
            rua = _mean(sua[ka], cua[ka], 0.38, 4.0)
            ru = _mean(su[u], cu[u], 0.38, 8.0)
            rv = _mean(sv[v], cv[v], 0.38, 10.0)
            ra = _mean(sa[a], ca[a], 0.38, 10.0)
            rt = _mean(stab[kt], ctab[kt], 0.38, 6.0)
            rd = _mean(sdur[kd], cdur[kd], 0.38, 6.0)
            vals.append(0.24 * rua + 0.18 * ruv + 0.16 * ru + 0.14 * ra + 0.12 * rv + 0.09 * rt + 0.07 * rd)
            if sp == 'train':
                wr = _cap_ratio(play, dur)
                sv[v] += wr; cv[v] += 1
                sa[a] += wr; ca[a] += 1
                su[u] += wr; cu[u] += 1
                suv[kv] += wr; cuv[kv] += 1
                sua[ka] += wr; cua[ka] += 1
                stab[kt] += wr; ctab[kt] += 1
                sdur[kd] += wr; cdur[kd] += 1
        if sp == split:
            out = np.asarray(vals, dtype=np.float64)
    return out


def run_predict(splits, data_dir, split='valid', seed=0, device='cpu', verbose=False):
    base = impl.run_predict(splits, data_dir, split=split, seed=seed, device=device, verbose=verbose).astype(np.float64)
    users = np.asarray([r[1] for r in splits[split]], dtype=np.int64)
    br = impl.impl.within_user_ranks(base, users)
    wr = impl.impl.within_user_ranks(watchtime_prior(splits, data_dir, split), users)
    return 0.95 * br + 0.05 * wr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    print(f'loading {a.data_dir} ...')
    splits = impl.impl.load(a.data_dir)
    print({k: len(v) for k, v in splits.items()}, 'fields=node28_plus_train_watchtime_prior')
    scores = run_predict(splits, a.data_dir, split=a.split, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
