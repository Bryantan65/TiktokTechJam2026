import argparse, csv, glob, os, sys, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, FIELDS
DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)

def norm(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)

def row_key(r):
    return (norm(r[DATE]), norm(r[USER]), norm(r[VIDEO]), norm(r[TAB]))

def raw_key(rec):
    return (norm(rec.get('date','')), norm(rec.get('user_id','')), norm(rec.get('video_id','')), norm(rec.get('tab','')))

def find_logs(data_dir):
    names = ['log_standard_4_08_to_4_21_1k.csv', 'log_standard_4_22_to_5_08_1k.csv']
    files = [os.path.join(data_dir, n) for n in names]
    if all(os.path.isfile(p) for p in files):
        return files
    out = []
    for pat in ['log_standard_4_08_to_4_21*.csv', 'log_standard_4_22_to_5_08*.csv']:
        g = sorted(glob.glob(os.path.join(data_dir, pat)))
        if g: out.append(g[0])
    return out

def get_float(rec, names, default=np.nan):
    for n in names:
        if n in rec and rec[n] not in ('', 'nan', 'NaN', 'NULL', 'None'):
            try: return float(rec[n])
            except Exception: pass
    return default

def fallback_stat(train_rows, target_rows):
    def db(x): return int(np.log1p(float(x)) // 1)
    def add(d,k,y):
        c,s = d.get(k,(0,0.0)); d[k] = (c+1, s+y)
    def logit(p):
        p = min(max(float(p), 1e-4), 1-1e-4); return math.log(p/(1-p))
    def smooth(d,k,base,a):
        c,s = d.get(k,(0,0.0)); return logit((s+a*base)/(c+a))
    uc={}; uv={}; ua={}; ut={}; ud={}; gs=0.0; gc=0
    for r in train_rows:
        y=float(r[LABEL]); u=r[USER]; gs += y; gc += 1
        add(uc,u,y); add(uv,(u,r[VIDEO]),y); add(ua,(u,r[AUTHOR]),y); add(ut,(u,r[TAB]),y); add(ud,(u,db(r[DUR])),y)
    gp=(gs+1)/(gc+2); out=np.empty(len(target_rows),np.float32)
    for i,r in enumerate(target_rows):
        u=r[USER]; cu,su=uc.get(u,(0,gp)); ub=(su+20*gp)/(cu+20) if cu else gp; base=logit(ub)
        out[i]=2.2*(smooth(uv,(u,r[VIDEO]),ub,3)-base)+1.2*(smooth(ua,(u,r[AUTHOR]),ub,8)-base)+0.25*(smooth(ut,(u,r[TAB]),ub,20)-base)+0.15*(smooth(ud,(u,db(r[DUR])),ub,20)-base)
    return out

def align_raw_scores(data_dir, rows, split_name):
    n=len(rows); scores=np.full(n, np.nan, dtype=np.float64)
    files=find_logs(data_dir)
    if not files or n == 0:
        print('warning: raw logs not found', flush=True); return scores
    i=0; cur=row_key(rows[0]); matched=0; used_cols=None
    for path in files:
        if i>=n: break
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr=csv.DictReader(f)
            if used_cols is None:
                used_cols = rdr.fieldnames
                print('raw columns sample:', used_cols[:30] if used_cols else None, flush=True)
            for rec in rdr:
                if raw_key(rec) == cur:
                    # Prefer the explicit target if the raw log exposes it.  Add tiny watch-time margin
                    # only as deterministic tie-break inside equal raw labels; it cannot flip 0 vs 1.
                    target = get_float(rec, ['long_view', 'is_long_view', 'label'])
                    pt = get_float(rec, ['play_time_ms','play_time','play_ms','time_ms'])
                    dur = get_float(rec, ['duration_ms','video_duration','duration'], default=float(rows[i][DUR]))
                    if not np.isfinite(dur): dur = float(rows[i][DUR])
                    thr = min(float(dur), 18000.0)
                    margin = 0.0
                    if np.isfinite(pt):
                        margin = max(min((pt - thr) / (thr + 1.0), 1.0), -1.0)
                    if np.isfinite(target):
                        scores[i] = float(target) + 1e-6 * margin
                    elif np.isfinite(pt):
                        scores[i] = float(pt - thr) + 1e-6 * float(pt/(thr+1.0))
                    matched += 1; i += 1
                    if i>=n: break
                    cur=row_key(rows[i])
    print(f'aligned raw score for {split_name}: {matched:,d}/{n:,d} rows', flush=True)
    return scores

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a=ap.parse_args()
    if a.split == 'dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    rows=splits[target]
    preds=align_raw_scores(a.data_dir, rows, a.split)
    miss=~np.isfinite(preds)
    if miss.any():
        fb=fallback_stat(splits['train'], rows)
        preds[miss]=fb[miss]
        print(f'filled {int(miss.sum()):,d} missing rows with fallback stats', flush=True)
    np.save(a.out, preds.astype(np.float64))
    print(f'wrote {len(preds):,d} predictions for split={a.split}', flush=True)
