"""Run one solution and score it. No LLM, no judgement, never raises.

The contract a solution must satisfy:

    python solutions/NNN_name.py --data_dir DIR --split valid --out FILE.npy

writes a float array of len(split) scores, one per row, in the split's row
order, then exits 0.

Solutions emit *predictions*, never metrics. The harness owns the labels and
the scoring. That is deliberate: an agent that computes its own metric will
eventually report a number it did not earn, and the search then optimises
toward a fiction. (Documented failure mode - see the MLE-bench lessons: "never
hard-code evaluation metrics".)

Scoring uses kuairand-starter-kit/evaluate.py unmodified. It is the sole
authority.
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'kuairand-starter-kit')
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load                     # noqa: E402  official, unmodified
from evaluate import evaluate             # noqa: E402  official, unmodified
import ledger                             # noqa: E402

DATA_DIR = os.path.join(ROOT, 'rec_datasets', 'KuaiRand-Pure', 'data')
TIMEOUT = 900          # 15 min; a correct run takes ~30s, so this is a hang
_SPLIT_CACHE = {}

# How many seeds each experiment is scored on. Every ledger row used to be a
# single draw: measured 2026-08-27, one solution scored 0.603999 / 0.603210 /
# 0.602734 on seeds 0/1/2 - a 0.0013 swing from changing nothing. The agent was
# meanwhile deciding between experiments that differed by 0.0003, so five
# consecutive iterations resolved nothing at all. Averaging n seeds shrinks the
# error on the mean by sqrt(n), which is what makes a small margin readable.
#
# Cost is linear: ~40 s per seed on one core. Set HARNESS_SEEDS=1 to go back to
# single-seed scoring (faster, and what the organisers' epsilon assumes).
N_SEEDS = int(os.environ.get('HARNESS_SEEDS', 3))


def _splits(data_dir):
    if data_dir not in _SPLIT_CACHE:
        _SPLIT_CACHE[data_dir] = load(data_dir)
    return _SPLIT_CACHE[data_dir]


def syntax_check(path):
    """Parse without executing. Milliseconds, and gives a line number."""
    try:
        with open(path, encoding='utf-8') as fh:
            ast.parse(fh.read(), filename=path)
        return None
    except SyntaxError as exc:
        return 'SyntaxError: %s (line %s)' % (exc.msg, exc.lineno)
    except OSError as exc:
        return 'cannot read solution: %s' % exc


def _run_one(solution, split, data_dir, seed, timeout, n_rows):
    """Execute the solution once. Returns (scores, error, stdout, stderr).

    scores is None whenever error is set. Every failure mode a solution can
    reach is converted to a string here, so the caller never has to catch.
    """
    out_fd, out_path = tempfile.mkstemp(suffix='.npy')
    os.close(out_fd)
    try:
        cmd = [sys.executable, solution, '--data_dir', data_dir,
               '--split', split, '--out', out_path, '--seed', str(seed)]
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, 'timeout after %ds (seed %d)' % (timeout, seed), '', ''

        stdout = '\n'.join(proc.stdout.strip().splitlines()[-15:])
        stderr = '\n'.join(proc.stderr.strip().splitlines()[-25:])
        if proc.returncode != 0:
            return None, 'exited %d (seed %d)' % (proc.returncode, seed), \
                   stdout, stderr

        try:
            scores = np.load(out_path)
        except Exception as exc:                       # noqa: BLE001
            return None, 'could not read predictions (seed %d): %s: %s' % (
                seed, type(exc).__name__, exc), stdout, stderr

        scores = np.asarray(scores, dtype=np.float64).ravel()
        if len(scores) != n_rows:
            return None, ('wrote %d scores, split has %d rows (seed %d)'
                          % (len(scores), n_rows, seed)), stdout, stderr
        if not np.isfinite(scores).all():
            return None, 'predictions contain NaN or Inf (seed %d)' % seed, \
                   stdout, stderr
        return scores, None, stdout, stderr
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def run_experiment(solution, hypothesis='', parent=None, by='agent',
                   split='valid', data_dir=DATA_DIR, seed=0, timeout=TIMEOUT,
                   n_seeds=None):
    """Run one solution, score it, log it. Always returns a dict; never raises.

    `split` is chosen by the harness, never by the solution - a solution cannot
    ask to be scored on test.
    """
    t0 = time.time()
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    seeds = list(range(seed, seed + n_seeds))
    rec = {
        'iteration': ledger.next_iteration(),
        'solution': os.path.relpath(solution, ROOT).replace('\\', '/'),
        'hypothesis': hypothesis,
        'parent': parent,
        'by': by,
        'split': split,
        'seed': seed,
        'seeds': seeds,
        'timestamp': ledger.stamp(),
        'status': 'error',
        'error': None,
        'valid_primary': None,
        'GAUC': None,
        'nDCG@5': None,
    }

    if split != 'valid':
        rec['error'] = ("refusing to score on %r; the agent develops on valid "
                        "only" % split)
        ledger.write(rec)
        return rec

    solutions_dir = os.path.join(ROOT, 'solutions')
    real_solution = os.path.realpath(solution)
    if not real_solution.startswith(os.path.realpath(solutions_dir) + os.sep):
        rec['error'] = ("refusing to run %r; solutions must live inside "
                        "solutions/" % solution)
        ledger.write(rec)
        return rec

    if not os.path.isfile(solution):
        rec['error'] = 'no such solution: %s' % solution
        ledger.write(rec)
        return rec

    rec['source_hash'] = ledger.source_hash(solution)
    prior = ledger.find_by_hash(rec['source_hash'])
    if prior is not None:
        rec.update({'status': 'duplicate',
                    'error': 'identical source already run at iteration %d '
                             '(valid_primary %s)' % (prior['iteration'],
                                                     prior.get('valid_primary'))})
        ledger.write(rec)
        return rec

    err = syntax_check(solution)
    if err:
        rec['error'] = err
        rec['seconds'] = round(time.time() - t0, 1)
        ledger.write(rec)
        return rec

    rows = _splits(data_dir)[split]
    uids = [r[1] for r in rows]
    labels = [r[6] for r in rows]

    per_seed = []
    for s in seeds:
        scores, err, stdout, stderr = _run_one(solution, split, data_dir, s,
                                               timeout, len(rows))
        if stdout:
            rec['stdout_tail'] = stdout
        if err:
            # One seed failing means the solution is broken, not unlucky.
            # Report it immediately rather than averaging over the survivors:
            # a mean of the runs that happened to work is not a measurement of
            # anything, and it would hide a solution that fails 1 time in 3.
            rec['error'] = err
            if stderr:
                rec['stderr_tail'] = stderr
            rec['seeds_completed'] = len(per_seed)
            rec['seconds'] = round(time.time() - t0, 1)
            # The run log has to carry this too. A solution crash is an "error
            # event" under the run-log requirements, but events.jsonl only ever
            # saw API-level failures, so a run whose best robustness moment was
            # crash -> diagnose -> fix showed no trace of it there.
            ledger.log_event('solution_error', rec['error'],
                             iteration=rec['iteration'],
                             solution=os.path.basename(solution),
                             stderr_tail=(stderr or '')[-600:])
            ledger.write(rec)
            return rec

        res = evaluate(uids, labels, scores)
        per_seed.append({'seed': s,
                         'GAUC': round(res['GAUC'], 6),
                         'nDCG@5': round(res['nDCG@5'], 6),
                         'primary': round(res['primary'], 6)})
        if s == seeds[0]:
            # Fingerprint the first seed's predictions. Identical output from
            # different code means nothing was actually tested - the change was
            # computed and then discarded. Scoring it anyway records a no-op as
            # evidence about the technique.
            rec['predictions_hash'] = hashlib.sha256(
                np.ascontiguousarray(scores.astype(np.float64))).hexdigest()[:12]
            rec.update({'users': res['users'], 'rows': res['rows']})

    def _mean(key):
        return round(sum(p[key] for p in per_seed) / len(per_seed), 6)

    def _std(key):
        if len(per_seed) < 2:
            return None
        m = sum(p[key] for p in per_seed) / len(per_seed)
        var = sum((p[key] - m) ** 2 for p in per_seed) / (len(per_seed) - 1)
        return round(var ** 0.5, 6)

    # The headline numbers are means across seeds, so verdict() and converged()
    # act on a measurement rather than on one draw. primary_std is what says
    # whether a margin is readable at all: a gap smaller than it is not a
    # result, and the agent is shown both.
    rec.update({'status': 'ok', 'error': None,
                'GAUC': _mean('GAUC'),
                'nDCG@5': _mean('nDCG@5'),
                'valid_primary': _mean('primary'),
                'primary_std': _std('primary'),
                'per_seed': per_seed})

    twin, why = ledger.find_twin(rec['predictions_hash'], rec,
                                 exclude_iteration=rec['iteration'])
    if twin is not None:
        rec['status'] = 'no-op'
        rec['no_op_twin'] = twin['iteration']
        rec['error'] = (
            'no-op: %s as iteration %d (%s). Different code, same model - '
            'nothing was actually tested. Common cause: the new method ran '
            'but a "keep the best checkpoint" rule discarded its result, so '
            'the final model is the parent. Check that your change reaches '
            'the model that gets saved. This is NOT evidence about the '
            'technique.'
            % (why, twin['iteration'],
               os.path.basename(twin.get('solution', '?'))))

    # Recovery is the half that gets scored. An experiment that scores while its
    # parent did not is the agent routing around a failure, which is the exact
    # behaviour the Robustness criterion asks about - and it was previously only
    # inferable by reading the parent chain by hand.
    if rec['status'] == 'ok' and parent is not None:
        try:
            prior = json.load(open(os.path.join(
                ledger.LOG_DIR, '%04d.json' % int(parent))))
        except (ValueError, OSError, TypeError):
            prior = None
        if prior is not None and prior.get('status') not in ('ok', None):
            ledger.log_event(
                'solution_recovered',
                'iteration %s scored %.6f after its parent %s failed (%s)'
                % (rec['iteration'], rec['valid_primary'], parent,
                   (prior.get('error') or '')[:120]),
                iteration=rec['iteration'], recovered_from=int(parent))

    rec['seconds'] = round(time.time() - t0, 1)
    rec['verdict'] = ledger.verdict(rec['valid_primary'], rec['status'])
    rec['delta'] = (round(rec['valid_primary'] - ledger.BASELINE_VALID, 6)
                    if rec['valid_primary'] is not None else None)
    ledger.write(rec)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run and score one solution')
    ap.add_argument('solution')
    ap.add_argument('--hypothesis', default='')
    ap.add_argument('--parent', default=None)
    ap.add_argument('--by', default='human')
    ap.add_argument('--seed', type=int, default=0, help='first seed')
    ap.add_argument('--n_seeds', type=int, default=None,
                    help='seeds to average over (default: HARNESS_SEEDS or 3)')
    ap.add_argument('--data_dir', default=DATA_DIR)
    a = ap.parse_args()
    print(json.dumps(run_experiment(a.solution, hypothesis=a.hypothesis,
                                    parent=a.parent, by=a.by, seed=a.seed,
                                    n_seeds=a.n_seeds,
                                    data_dir=a.data_dir), indent=2))
