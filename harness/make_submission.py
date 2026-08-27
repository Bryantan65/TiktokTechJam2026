"""Turn a solution into a submission CSV, and validate it. HUMAN TOOL ONLY.

    python harness/make_submission.py 008_time_features_two_bpr_bce.py \
        --split valid --out submission_valid.csv
    python harness/make_submission.py 008_time_features_two_bpr_bce.py \
        --split test  --out submission.csv

This is the one thing in the repo that may touch `test`, which is exactly why it
is not reachable by the agent: it lives outside `solutions/`, it is not in
TOOL_DISPATCH, and `harness/run.py` still refuses any split but valid. Scoring
on test is a decision a person makes, at most two or three times for the whole
competition.

Writing and checking both go through the official `submit.py` unmodified, so a
file this produces is validated by the organisers' own code rather than ours.

`--seeds N` averages predictions across N seeds. That is a real modelling choice,
not a formality: the ledger reports the MEAN OF METRICS across seeds, but a
submission is one set of numbers. Submitting seed 0 alone draws one sample from
that distribution - it can land either side of the mean. Averaging predictions
instead submits an N-model ensemble, which is a different (larger) model than the
one that was validated, and usually a slightly better one. Measure both on valid
and submit whichever you can defend.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'kuairand-starter-kit')
sys.path.insert(0, KIT)

from data import load                                  # noqa: E402  official
from evaluate import evaluate                          # noqa: E402  official
from submit import write_submission, read_submission   # noqa: E402  official

DATA_DIR = os.path.join(ROOT, 'rec_datasets', 'KuaiRand-Pure', 'data')


def predict(solution, split, data_dir, seed, n_rows):
    """One run of a solution. Returns its scores, or raises with the stderr."""
    fd, out = tempfile.mkstemp(suffix='.npy')
    os.close(fd)
    try:
        # Solutions locate the starter kit relative to their own file
        # ("../kuairand-starter-kit"), which stops resolving the moment a run is
        # archived into logs/<run>/solutions/. Putting the kit on PYTHONPATH
        # makes the import work wherever the file lives, so an archived run
        # stays reproducible - which is most of the point of archiving it.
        env = dict(os.environ)
        env['PYTHONPATH'] = KIT + os.pathsep + env.get('PYTHONPATH', '')
        proc = subprocess.run(
            [sys.executable, solution, '--data_dir', data_dir,
             '--split', split, '--out', out, '--seed', str(seed)],
            cwd=ROOT, capture_output=True, text=True, timeout=1800, env=env)
        if proc.returncode != 0:
            raise RuntimeError('solution exited %d on seed %d:\n%s'
                               % (proc.returncode, seed,
                                  '\n'.join(proc.stderr.strip().splitlines()[-20:])))
        s = np.asarray(np.load(out), dtype=np.float64).ravel()
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    if len(s) != n_rows:
        raise RuntimeError('wrote %d scores, %s has %d rows'
                           % (len(s), split, n_rows))
    if not np.isfinite(s).all():
        raise RuntimeError('predictions contain NaN or Inf (seed %d)' % seed)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('solution', help='filename inside solutions/, or a path')
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=1,
                    help='average predictions over this many seeds (default 1)')
    ap.add_argument('--data_dir', default=DATA_DIR)
    a = ap.parse_args()

    sol = a.solution
    if not os.path.isfile(sol):
        sol = os.path.join(ROOT, 'solutions', a.solution)
    if not os.path.isfile(sol):
        for d in sorted(os.listdir(os.path.join(ROOT, 'logs'))):
            cand = os.path.join(ROOT, 'logs', d, 'solutions', a.solution)
            if os.path.isfile(cand):
                sol = cand
                break
    if not os.path.isfile(sol):
        sys.exit('no such solution: %s' % a.solution)

    rows = load(a.data_dir)[a.split]
    print('solution : %s' % os.path.relpath(sol, ROOT).replace('\\', '/'))
    print('split    : %s (%d rows)' % (a.split, len(rows)))

    per_seed = []
    for seed in range(a.seeds):
        s = predict(sol, a.split, a.data_dir, seed, len(rows))
        per_seed.append(s)
        print('  seed %d done' % seed)

    # Rank-average rather than mean the raw scores: different seeds can put
    # their logits on different scales, and one wide-ranged run would otherwise
    # dominate the average. Only relative order is scored, so ranks lose
    # nothing.
    if len(per_seed) == 1:
        scores = per_seed[0]
    else:
        ranks = [np.argsort(np.argsort(s)).astype(np.float64) for s in per_seed]
        scores = np.mean(ranks, axis=0)
        print('  averaged %d seeds by rank' % len(per_seed))

    write_submission(a.out, rows, scores)
    print('wrote    : %s' % a.out)

    # Validate with the organisers' own reader, which is stricter than anything
    # we would write: header, row_id continuity, per-row (user_id, video_id)
    # alignment, and non-finite scores.
    check = read_submission(a.out, rows)
    print('check    : OK, %d rows aligned' % len(check))

    if a.split == 'valid':
        res = evaluate([r[1] for r in rows], [r[6] for r in rows],
                       np.asarray(check, dtype=np.float64))
        print('valid    : GAUC %.6f  nDCG@5 %.6f  primary %.6f'
              % (res['GAUC'], res['nDCG@5'], res['primary']))
    else:
        print('test     : not scored here. Submit the file; the score is the '
              'organisers\' to compute.')


if __name__ == '__main__':
    main()
