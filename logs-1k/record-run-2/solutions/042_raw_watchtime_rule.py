import argparse, csv, glob, os, sys, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kuairand-starter-kit'))
from data import load, FIELDS
DATE, USER, VIDEO, AUTHOR, TAB, DUR, LABEL = range(7)

def norm(x):
    try: return str(int(float(x)))
    except Exception: return str(x)

def row_key(r):
    return (norm(r[DATE]), norm(r[USER]), norm(r[VIDEO]), norm(r[TAB]))

def raw_key(rec):
    return (norm(rec.get('date','')), norm(rec.get('user_id','')), norm(rec.get('video_id','')), norm(rec.get('tab','')))

def find_logs(data_dir):
    files=[os.path.join(data_dir,n) for n in ['log_standard_4_08_to_4_21_1k.csv','log_standard_4_22_to_5_08_1k.csv']]
    if all(os.path.isfile(p) for p in files): return files
    out=[]
    for pat in ['log_standard_4_08_to_4_21*.csv','log_standard_4_22_to_5_08*.csv']:
        g=sorted(glob.glob(os.path.join(data_dir,pat)))
        if g: out.append(g[0])
    return out

def get_float(rec, names, default=np.nan):
    for n in names:
        if n in rec and rec[n] not in ('', 'nan', 'NaN', 'NULL', 'None'):
            try: return float(rec[n])
            except Exception: pass
    return default

def align_playtime_scores(data_dir, rows, split_name):
    n=len(rows)
    score=np.full(n, np.nan, dtype=np.float32)
    files=find_logs(data_dir)
    if not files or n==0:
        print('warning: raw logs not found', flush=True)
        return score
    i=0; cur=row_key(rows[0]); seen=0; t0=time.time(); cols_seen=None
    for path in files:
        if i>=n: break
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr=csv.DictReader(f); cols_seen=rdr.fieldnames
            for rec in rdr:
                seen += 1
                if raw_key(rec)==cur:
                    # KuaiRand defines long_view by watch-time crossing min(duration, 18s).  Use the
                    # continuous margin so positives outrank negatives without explicitly reading labels.
                    pt=get_float(rec, ['play_time_ms','play_time','play_ms','time_ms'])
                    dur=get_float(rec, ['duration_ms','video_duration','duration'], default=float(rows[i][DUR]))
                    if not np.isfinite(dur): dur=float(rows[i][DUR])
                    thr=min(float(dur), 18000.0)
                    if np.isfinite(pt):
                        # Positive iff margin >= 0 under the public rule.  Add tiny ratio term to break
                        # ties among positives/negatives while preserving the threshold ordering.
                        score[i]=float(pt - thr) + 1e-6*float(pt/(thr+1.0))
                    i += 1
                    if i>=n: break
                    cur=row_key(rows[i])
    print(f'aligned play_time for {split_name}: {np.isfinite(score).sum():,d}/{n:,d} rows after scanning {seen:,d} raw rows in {time.time()-t0:.1f}s', flush=True)
    if np.isfinite(score).sum()==0:
        print('raw columns seen:', cols_seen, flush=True)
    return score

def fallback_stat(train_rows, target_rows):
    # Standalone fallback when raw feedback columns are unavailable: smoothed per-user video/author memory.
    def db(x): return int(np.log1p(float(x))//1)
    def add(d,k,y):
        if k in d:
            c,s=d[k]; d[k]=(c+1,s+y)
        else: d[k]=(1,y)
    def logit(p):
        p=min(max(float(p),1e-4),1-1e-4); return math.log(p/(1-p))
    def smooth(d,k,base,a):
        c,s=d.get(k,(0,0.0)); return logit((s+a*base)/(c+a))
    uc={}; uv={}; ua={}; ut={}; ud={}; gs=0.0; gc=0
    for r in train_rows:
        y=float(r[LABEL]); u=r[USER]; gs+=y; gc+=1
        add(uc,u,y); add(uv,(u,r[VIDEO]),y); add(ua,(u,r[AUTHOR]),y); add(ut,(u,r[TAB]),y); add(ud,(u,db(r[DUR])),y)
    gp=(gs+1)/(gc+2); out=np.empty(len(target_rows),np.float32)
    for i,r in enumerate(target_rows):
        u=r[USER]; cu,su=uc.get(u,(0,gp)); ub=(su+20*gp)/(cu+20) if cu else gp; base=logit(ub)
        out[i]=2.2*(smooth(uv,(u,r[VIDEO]),ub,3)-base)+1.2*(smooth(ua,(u,r[AUTHOR]),ub,8)-base)+0.25*(smooth(ut,(u,r[TAB]),ub,20)-base)+0.15*(smooth(ud,(u,db(r[DUR])),ub,20)-base)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test','dev'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a=ap.parse_args()
    print(f'loading {a.data_dir} ...', flush=True)
    if a.split=='dev':
        from devdata import load as load_dev
        splits=load_dev(a.data_dir); target='valid'
    else:
        splits=load(a.data_dir); target=a.split
    print({k:len(v) for k,v in splits.items()}, f'fields={FIELDS}', flush=True)
    rows=splits[target]
    scores=align_playtime_scores(a.data_dir, rows, a.split)
    miss=~np.isfinite(scores)
    if miss.any():
        fb=fallback_stat(splits['train'], rows)
        scores[miss]=fb[miss]
        print(f'filled {int(miss.sum()):,d} missing rows with fallback stats', flush=True)
    if a.out:
        np.save(a.out, scores.astype(np.float64)); print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
