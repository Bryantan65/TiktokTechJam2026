"""IPS-debiased historical-rate member blended with node 29.

New mechanism: use the label-free random log from the same target period only to
estimate exposure propensities for video/author/tab.  Training labels still come
only from data.load(); random labels are ignored.  Historical rates are then
computed with inverse exposure weights, and blended at 30% with the unchanged
node-29 seed-2 ensemble so the signal is readable.
"""
import argparse, csv, glob, importlib.util, os, sys
from collections import Counter, defaultdict
import numpy as np
import torch

SOL_DIR = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(SOL_DIR, '030_seed2_topcf.py')
spec = importlib.util.spec_from_file_location('seed2topcf030', HELPER)
seed2topcf030 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed2topcf030)

sys.path.insert(0, os.path.join(SOL_DIR, '..', 'kuairand-starter-kit'))
from data import load, FIELDS  # noqa


def find_random_log(data_dir):
    pats = [
        os.path.join(data_dir, 'log_random_4_22_to_5_08_1k.csv'),
        os.path.join(data_dir, 'log_random_4_22_to_5_08*.csv'),
        os.path.join(data_dir, '*random*1k.csv'),
    ]
    for pat in pats:
        fs = glob.glob(pat)
        fs = [f for f in fs if '_pure' not in os.path.basename(f)] or fs
        if fs:
            return sorted(fs)[0]
    return None


def get_col(row, names):
    for n in names:
        if n in row and row[n] != '':
            return row[n]
    low = {k.lower(): k for k in row.keys()}
    for n in names:
        k = low.get(n.lower())
        if k is not None and row[k] != '':
            return row[k]
    return ''


def per_user_percentile(p, users):
    p = np.asarray(p, dtype=np.float64)
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    su = users[order]
    bounds = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1, len(su)]
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        sidx = idx[np.argsort(p[idx], kind='stable')]
        n = len(sidx)
        out[sidx] = 0.0 if n <= 1 else np.arange(n, dtype=np.float64) / (n - 1.0)
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def count_standard(rows):
    cv, ca, ct = Counter(), Counter(), Counter()
    for r in rows:
        cv[str(r[2])] += 1
        ca[str(r[3])] += 1
        ct[str(r[4])] += 1
    return cv, ca, ct


def count_random(data_dir):
    path = find_random_log(data_dir)
    cv, ca, ct = Counter(), Counter(), Counter()
    if path is None:
        print('random log not found; IPS member falls back to unweighted stats')
        return cv, ca, ct, 0
    print('reading label-free random exposure log from', path)
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        n = 0
        for row in rdr:
            v = get_col(row, ['video_id', 'videoid', 'photo_id', 'item_id'])
            au = get_col(row, ['author_id', 'authorid'])
            tab = get_col(row, ['tab', 'tab_id'])
            if v != '': cv[str(v)] += 1
            if au != '': ca[str(au)] += 1
            if tab != '': ct[str(tab)] += 1
            n += 1
    print('random exposure rows', n, 'unique video/author/tab', len(cv), len(ca), len(ct))
    return cv, ca, ct, n


def make_ratio(std_c, rnd_c, std_n, rnd_n, alpha=20.0):
    # ratio ~= p_random(key) / p_standard(key).  Missing random exposure should
    # not zero out a row, so smooth and clip aggressively for variance control.
    keys = set(std_c.keys()) | set(rnd_c.keys())
    kden = max(1, len(keys))
    out = {}
    for k in keys:
        ps = (std_c.get(k, 0.0) + alpha) / (std_n + alpha * kden)
        pr = (rnd_c.get(k, 0.0) + alpha) / (rnd_n + alpha * kden) if rnd_n > 0 else ps
        out[k] = float(np.clip(pr / ps, 0.20, 5.0))
    return out


def weighted_rate_maps(train_rows, rv, ra, rt):
    sums = defaultdict(float); cnts = defaultdict(float)
    gm_s = 0.0; gm_c = 0.0
    for r in train_rows:
        u = str(r[1]); v = str(r[2]); au = str(r[3]); tab = str(r[4]); dur = int(r[5]) // 10000
        y = float(r[6])
        # Geometric-ish mixture: item propensity matters most, author and tab add
        # lower-variance corrections.  The exponent keeps IPS from exploding.
        w = (rv.get(v, 1.0) ** 0.55) * (ra.get(au, 1.0) ** 0.30) * (rt.get(tab, 1.0) ** 0.15)
        w = float(np.clip(w, 0.25, 4.0))
        gm_s += w * y; gm_c += w
        feats = [
            ('v', v), ('a', au), ('t', tab), ('d', str(dur)),
            ('uv', u + '\x1f' + v), ('ua', u + '\x1f' + au), ('ut', u + '\x1f' + tab),
            ('vt', v + '\x1f' + tab), ('at', au + '\x1f' + tab),
        ]
        for key in feats:
            cnts[key] += w
            sums[key] += w * y
    gm = gm_s / max(gm_c, 1e-9)
    print('IPS weighted global mean', gm, 'effective weight mean', gm_c / max(1, len(train_rows)))
    return sums, cnts, gm


def ips_stats_member(splits, target, data_dir):
    cache_dir = 'pred_cache'; os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'035_ips_stats_{target}.npy')
    upath = os.path.join(cache_dir, f'035_ips_stats_{target}_users.npy')
    if os.path.isfile(path) and os.path.isfile(upath):
        p = np.load(path, allow_pickle=False); u = np.load(upath, allow_pickle=True)
        if len(p) == len(splits[target]):
            print('loaded cache', path)
            return p, u

    tr = splits['train']; tg = splits[target]
    sv, sa, st = count_standard(tr)
    rv_c, ra_c, rt_c, rn = count_random(data_dir)
    rv = make_ratio(sv, rv_c, len(tr), rn, alpha=30.0)
    ra = make_ratio(sa, ra_c, len(tr), rn, alpha=50.0)
    rt = make_ratio(st, rt_c, len(tr), rn, alpha=200.0)
    print('ratio examples tab', {k: round(rt[k], 3) for k in list(rt)[:8]})
    sums, cnts, gm = weighted_rate_maps(tr, rv, ra, rt)

    # Larger alpha for sparse user crosses, smaller for item/author.  Return a
    # logit-rate ensemble analogous to the strong historical-stat member, but
    # trained under inverse exposure weights.
    spec = [
        (1.25, 35.0, 'v'), (1.05, 35.0, 'a'), (0.95, 90.0, 't'), (0.35, 100.0, 'd'),
        (1.55, 18.0, 'uv'), (1.25, 28.0, 'ua'), (0.85, 45.0, 'ut'),
        (0.55, 45.0, 'vt'), (0.45, 45.0, 'at'),
    ]
    score = np.zeros(len(tg), dtype=np.float64); wsum = 0.0
    users = np.empty(len(tg), dtype=object)
    for i, r in enumerate(tg):
        u = str(r[1]); v = str(r[2]); au = str(r[3]); tab = str(r[4]); dur = int(r[5]) // 10000
        users[i] = r[1]
        vals = {
            'v': v, 'a': au, 't': tab, 'd': str(dur),
            'uv': u + '\x1f' + v, 'ua': u + '\x1f' + au, 'ut': u + '\x1f' + tab,
            'vt': v + '\x1f' + tab, 'at': au + '\x1f' + tab,
        }
        s = 0.0; ww = 0.0
        for w, alpha, name in spec:
            key = (name, vals[name])
            c = cnts.get(key, 0.0); sm = sums.get(key, 0.0)
            rate = (sm + alpha * gm) / (c + alpha)
            s += w * logit(rate); ww += w
        score[i] = s / ww
    p = per_user_percentile(score, users)
    np.save(path, p.astype(np.float64)); np.save(upath, users)
    return p, users


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test', 'dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    torch.manual_seed(2)
    print(f'loading {a.data_dir} ...')
    if a.split == 'dev':
        from devdata import load as load_dev
        splits = load_dev(a.data_dir); target = 'valid'; cache_split = 'dev'
    else:
        splits = load(a.data_dir); target = a.split; cache_split = a.split
    print({k: len(v) for k, v in splits.items()}, f'fields={FIELDS}')
    n = len(splits[target])

    rm, raux, rh, rcf, users = seed2topcf030.components(splits, target, cache_split, 2, a.data_dir, a.device, n)
    base = 0.50 * rm + 0.50 * rh
    top_aux = base + 0.35 * np.power(np.clip(base, 0.0, 1.0), 16.0) * (raux - base)
    wcf = 0.20 + 0.25 * np.power(np.clip(base, 0.0, 1.0), 8.0)
    node29 = per_user_percentile((1.0 - wcf) * top_aux + wcf * rcf, users)

    ips, u2 = ips_stats_member(splits, target, a.data_dir)
    # Align to the encoded users returned by node29; row order is the same, but
    # percentile-normalize once more after the blend.
    preds = per_user_percentile(0.70 * node29 + 0.30 * ips, users)
    if a.out:
        np.save(a.out, preds.astype(np.float64))
        print(f'wrote {len(preds):,d} IPS-debiased blend predictions for split={a.split}')
    else:
        print(preds[:10])
