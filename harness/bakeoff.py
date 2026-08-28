"""Run the same agent on several models and compare what they do with it.

Every record run so far - 3, 4, 5, 6 - used gpt-5.5, and a day of work has gone
into the scaffolding around it. MLE-bench's own conclusion is that "while
scaffolding design matters, the underlying base model is the ultimate
bottleneck" (arXiv 2410.07095), so the one variable nobody has moved is the one
the literature points at.

Swapping the model injects no knowledge about the task, which makes it the
cleanest change available on the autonomy line: same prompt, same tools, same
harness, same starting solution. Only the reasoning engine differs.

    python harness/bakeoff.py --iters 10
    python harness/bakeoff.py --models gpt-5.5,gpt-5.6-sol --iters 8
    python harness/bakeoff.py --report            # re-print without re-running

THESE ARE NOT RECORD RUNS. They are capped by --iters, well below the
convergence floor, so none of them can converge and none is a valid autonomy
result. They exist to choose a model; the chosen model then gets a real run.

Runs are sequential on purpose: two agents at once would contend for the same
cores and neither timing would mean anything.

A caveat that has to be read with the output. This is n=1 per model against a
process whose run-to-run spread on the final number is ~0.0005. A 0.001 gap
between two models here is one draw, not a ranking. What is worth reading is
the qualitative column: does the model use the capabilities it was given -
does it screen on dev, reach for a library, refine rather than restart, recover
from its own crashes.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
DEFAULT_MODELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']
BASELINE_VALID = 0.6015

# The gpt-5.6 family refuses function tools on /v1/chat/completions unless
# reasoning is switched off:
#   "Function tools with reasoning_effort are not supported ... use
#    /v1/responses or set reasoning_effort to 'none'."
# Verified on all three variants: with reasoning_effort='none' they call tools
# correctly; without it they 400.
#
# So this compares 5.5 WITH reasoning against 5.6 WITHOUT it. That is a
# handicap, not a fair fight, and the result is a FLOOR for the 5.6 family
# rather than a measurement of it. It is still the right cheap experiment: if
# 5.6 keeps up while thinking less, porting the loop to /v1/responses (where it
# keeps its reasoning) is worth the risk. If it does not, we learned that for
# the price of a few iterations instead of by rewriting the loop two days
# before the deadline.
MODEL_ENV = {m: {'AGENT_REASONING_EFFORT': 'none'}
             for m in ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra')}


def _slug(model):
    return 'bakeoff-' + model.replace('.', '').replace('-', '')


def _load(run_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(run_dir, '0*.json'))):
        try:
            with open(f, encoding='utf-8') as fh:
                recs.append(json.load(fh))
        except (ValueError, OSError):
            pass
    return recs


def _events(run_dir):
    path = os.path.join(run_dir, 'events.jsonl')
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def summarise(model, run_dir):
    """One row of the comparison. Everything here is read back from the run's
    own artifacts, not from anything the runner remembered."""
    recs, evs = _load(run_dir), _events(run_dir)
    ok = [r for r in recs if r.get('status') == 'ok'
          and r.get('split', 'valid') == 'valid']
    kinds = collections.Counter(e.get('kind') for e in evs)
    end = next((e for e in evs if e.get('kind') == 'run_end'), {})

    best = max((r['valid_primary'] for r in ok), default=None)
    # Iterations spent before first clearing the organisers' epsilon. The
    # interesting number is not the peak but how quickly it got somewhere.
    to_bar, running = None, 0.0
    for r in recs:
        if r.get('status') == 'ok' and r.get('split', 'valid') == 'valid':
            running = max(running, r['valid_primary'])
            if to_bar is None and running - BASELINE_VALID >= 0.002:
                to_bar = r['iteration']

    srcs = ' '.join(
        open(p, encoding='utf-8', errors='ignore').read()
        for p in glob.glob(os.path.join(run_dir, 'solutions', '*.py')))
    return {
        'model': model,
        'records': len(recs),
        'scored': len(ok),
        'screens': sum(1 for r in recs if r.get('split') == 'dev'),
        'best': best,
        'delta': None if best is None else best - BASELINE_VALID,
        'to_bar': to_bar,
        'compute_min': sum(r.get('seconds') or 0 for r in recs) / 60,
        'tokens_in': end.get('tokens_in'),
        'tokens_out': end.get('tokens_out'),
        'cost': end.get('cost_usd'),
        'searches': kinds.get('web_search', 0),
        'errors': kinds.get('solution_error', 0),
        'recovered': kinds.get('solution_recovered', 0),
        'gbdt': sum(x in srcs for x in ('lightgbm', 'xgboost')),
        'stop': end.get('detail'),
    }


def report(models):
    rows = [summarise(m, os.path.join(LOGS, _slug(m)))
            for m in models
            if os.path.isdir(os.path.join(LOGS, _slug(m)))]
    if not rows:
        print('no bake-off runs found under logs/')
        return rows

    print('\n%-14s %6s %6s %7s %9s %8s %6s %7s %7s %6s %5s'
          % ('model', 'recs', 'scored', 'screens', 'best', 'delta',
             'to bar', 'min', 'tok out', 'search', 'err'))
    print('-' * 100)
    for r in sorted(rows, key=lambda x: -(x['best'] or 0)):
        print('%-14s %6d %6d %7d %9s %8s %6s %7.0f %7s %6d %5d' % (
            r['model'], r['records'], r['scored'], r['screens'],
            ('%.6f' % r['best']) if r['best'] else '--',
            ('%+.4f' % r['delta']) if r['delta'] is not None else '--',
            r['to_bar'] if r['to_bar'] else '--',
            r['compute_min'],
            '{:,}'.format(r['tokens_out']) if r['tokens_out'] else '--',
            r['searches'], r['errors']))
    print('\nqualitative - did the model use what it was given:')
    for r in rows:
        print('   %-14s dev screens %-3d  GBDT solutions %-3d  '
              'crashes recovered %d/%d  stopped: %s'
              % (r['model'], r['screens'], r['gbdt'],
                 r['recovered'], r['errors'], r['stop'] or 'still running'))
    print('\nn=1 per model against ~0.0005 run-to-run spread: a gap of that '
          'size is a draw, not a ranking.')
    print('gpt-5.6 rows ran with reasoning_effort=none (tools are refused '
          'otherwise on chat.completions), so they are a FLOOR for that '
          'family, not a measurement of it.')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS))
    ap.add_argument('--iters', type=int, default=10,
                    help='iterations per model. Well below the convergence '
                         'floor on purpose: this picks a model, it does not '
                         'produce a result.')
    ap.add_argument('--report', action='store_true',
                    help='re-print the comparison without running anything')
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(',') if m.strip()]

    if a.report:
        report(models)
        return

    print('bake-off: %d models x %d iterations, sequential' % (len(models), a.iters))
    print('models: %s\n' % ', '.join(models))
    for model in models:
        run_id = _slug(model)
        if os.path.isdir(os.path.join(LOGS, run_id)):
            print('== %-14s SKIP, logs/%s already exists' % (model, run_id))
            continue
        env = dict(os.environ)
        env['AGENT_MODEL'] = model
        extra = MODEL_ENV.get(model, {})
        env.update(extra)
        print('== %-14s -> logs/%s%s'
              % (model, run_id,
                 ('  [%s]' % ', '.join('%s=%s' % kv for kv in extra.items()))
                 if extra else ''))
        t0 = time.time()
        rc = subprocess.call(
            [sys.executable, '-m', 'agent', '--run-id', run_id,
             '--max-iter', str(a.iters)],
            cwd=ROOT, env=env)
        print('   exit %d after %.0f min\n' % (rc, (time.time() - t0) / 60))
    report(models)


if __name__ == '__main__':
    main()
