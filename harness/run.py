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


def run_experiment(solution, hypothesis='', parent=None, by='agent',
                   split='valid', data_dir=DATA_DIR, seed=0, timeout=TIMEOUT):
    """Run one solution, score it, log it. Always returns a dict; never raises.

    `split` is chosen by the harness, never by the solution - a solution cannot
    ask to be scored on test.
    """
    t0 = time.time()
    rec = {
        'iteration': ledger.next_iteration(),
        'solution': os.path.relpath(solution, ROOT).replace('\\', '/'),
        'hypothesis': hypothesis,
        'parent': parent,
        'by': by,
        'split': split,
        'seed': seed,
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

    out_fd, out_path = tempfile.mkstemp(suffix='.npy')
    os.close(out_fd)
    try:
        cmd = [sys.executable, solution, '--data_dir', data_dir,
               '--split', split, '--out', out_path, '--seed', str(seed)]
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            rec['error'] = 'timeout after %ds' % timeout
            rec['seconds'] = round(time.time() - t0, 1)
            ledger.write(rec)
            return rec

        rec['stdout_tail'] = '\n'.join(proc.stdout.strip().splitlines()[-15:])
        if proc.returncode != 0:
            rec['error'] = 'exited %d' % proc.returncode
            rec['stderr_tail'] = '\n'.join(proc.stderr.strip().splitlines()[-25:])
            rec['seconds'] = round(time.time() - t0, 1)
            ledger.write(rec)
            return rec

        rows = _splits(data_dir)[split]
        try:
            scores = np.load(out_path)
        except Exception as exc:                       # noqa: BLE001
            rec['error'] = 'could not read predictions: %s: %s' % (
                type(exc).__name__, exc)
            rec['seconds'] = round(time.time() - t0, 1)
            ledger.write(rec)
            return rec

        scores = np.asarray(scores, dtype=np.float64).ravel()
        if len(scores) != len(rows):
            rec['error'] = ('wrote %d scores, split has %d rows'
                            % (len(scores), len(rows)))
            rec['seconds'] = round(time.time() - t0, 1)
            ledger.write(rec)
            return rec
        if not np.isfinite(scores).all():
            rec['error'] = 'predictions contain NaN or Inf'
            rec['seconds'] = round(time.time() - t0, 1)
            ledger.write(rec)
            return rec

        res = evaluate([r[1] for r in rows], [r[6] for r in rows], scores)
        rec.update({'status': 'ok', 'error': None,
                    'GAUC': round(res['GAUC'], 6),
                    'nDCG@5': round(res['nDCG@5'], 6),
                    'valid_primary': round(res['primary'], 6),
                    'users': res['users'], 'rows': res['rows']})
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

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
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_dir', default=DATA_DIR)
    a = ap.parse_args()
    print(json.dumps(run_experiment(a.solution, hypothesis=a.hypothesis,
                                    parent=a.parent, by=a.by, seed=a.seed,
                                    data_dir=a.data_dir), indent=2))
