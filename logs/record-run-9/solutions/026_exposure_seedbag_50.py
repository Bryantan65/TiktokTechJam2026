"""Cached blend refinement of node 25: same unchanged members, 50% exposure weight."""
import argparse, os, sys, importlib.util
import numpy as np

# Reuse the complete implementation from node 25; all member training/cache code is unchanged.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('node25_impl', os.path.join(_here, '025_exposure_seedbag_55.py'))
impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(impl)


def run_predict(splits, data_dir, split='valid', seed=0, device='cpu', verbose=False):
    enc_base, dim_base = impl.encode(splits); Xbase, _, uout = enc_base[split]
    enc_hist, dim_hist = impl.encode_history(splits, add_content=False); Xhist, _, uhist = enc_hist[split]
    vf = impl.read_video_features(data_dir)
    enc_cont, dim_cont = impl.encode_history(splits, video_feats=vf, add_content=True); Xcont, _, ucont = enc_cont[split]
    hour_qs = impl.read_hour_queues(data_dir)
    enc_time, dim_time = impl.encode_history(splits, video_feats=vf, add_content=True,
                                             hour_qs=hour_qs, add_time=True, add_exposure=True)
    Xtime, _, utime = enc_time[split]
    out = np.zeros(len(Xtime), dtype=np.float64)
    # Average official outer seeds for deterministic denoising, but restore node-24's more balanced
    # exposure weight to recover top-5 nDCG lost by the 55% variant.
    for s in [0, 1, 2]:
        base_rank = impl.ensemble_rank(enc_base, dim_base, Xbase, uout, split, s, device, verbose, '006', 'bpr', 'bce')
        hist_rank = impl.ensemble_rank(enc_hist, dim_hist, Xhist, uhist, split, s, device, verbose, '008', 'bprhist', 'bcehist')
        cont_rank = impl.ensemble_rank(enc_cont, dim_cont, Xcont, ucont, split, s, device, verbose, '010', 'bprcont', 'bcecont')
        node10 = 0.35 * base_rank + 0.25 * hist_rank + 0.40 * cont_rank
        time_rank = impl.ensemble_rank(enc_time, dim_time, Xtime, utime, split, s, device, verbose, '024', 'bprtimedateexpo', 'bcetimedateexpo')
        out += 0.50 * node10 + 0.50 * time_rank
    return out / 3.0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='valid', choices=['train','valid','test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = ap.parse_args()
    print(f'loading {a.data_dir} ...')
    splits = impl.load(a.data_dir)
    print({k: len(v) for k, v in splits.items()}, 'fields=seedbag50 exposure-count blend')
    scores = run_predict(splits, a.data_dir, split=a.split, seed=a.seed, device=a.device, verbose=a.out is None)
    if a.out:
        np.save(a.out, scores.astype(np.float64))
        print(f'wrote {len(scores):,d} predictions for split={a.split}')
    else:
        print(scores[:10])
