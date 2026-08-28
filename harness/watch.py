"""Emit one line per experiment and per notable run event, as they happen.

Run alongside the agent to follow a long unattended run:

    python harness/watch.py

Prints to stdout, line-buffered, so it works as the event source for a
monitor or just as a second terminal. Exits when the run logs `run_end`.

Emits failures as loudly as successes on purpose: a watcher that only prints
good news is silent through a crash loop, and silence looks exactly like
"still running".
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLL = 15


def newest_run_dir():
    """The most recently modified logs/<name>-N/ folder, else logs/iterations/.

    Runs write into their own folder now (ledger.init_run_dir), so a fixed path
    would watch the wrong place. Picking by mtime rather than by highest number
    means this follows whichever run is actually live, including a re-run of an
    earlier name.
    """
    logs = os.path.join(ROOT, 'logs')
    cands = []
    for name in os.listdir(logs):
        d = os.path.join(logs, name)
        if os.path.isdir(d) and re.match(r'^[a-z-]+-\d+$', name):
            cands.append((os.path.getmtime(d), d))
    if not cands:
        return os.path.join(logs, 'iterations')
    return max(cands)[1]


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run-dir', default=None,
                help='folder to watch (default: the most recently active run)')
_args = ap.parse_args()
LOG_DIR = _args.run_dir or newest_run_dir()
EVENTS = os.path.join(LOG_DIR, 'events.jsonl')
if not os.path.isfile(EVENTS):          # pre-run-folder layout
    alt = os.path.join(ROOT, 'logs', 'events.jsonl')
    if os.path.isfile(alt):
        EVENTS = alt

# Everything worth interrupting for. Deliberately includes every terminal
# state, not just the happy one.
LOUD = {'run_start', 'api_retry', 'api_recovered', 'api_gave_up', 'api_fatal',
        'crash', 'interrupted', 'converged', 'budget_stop', 'corrupt_record',
        'run_end', 'web_search'}


def emit(s):
    print(s, flush=True)


def read(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None          # mid-write; it will be picked up next poll


def describe(r):
    n, status = r.get('iteration'), r.get('status')
    sol = os.path.basename(r.get('solution') or '?')
    if status != 'ok':
        return '#%-3d %-34s %-9s %s' % (
            n, sol, status.upper(), (r.get('error') or '')[:100])
    p, sd = r.get('valid_primary'), r.get('primary_std')
    delta = p - 0.6015
    # A delta smaller than the spread has not been measured, so say so here
    # rather than leaving a tidy-looking number to be misread later.
    readable = '' if sd is None else ('' if abs(delta) > sd else '  [< spread]')
    return '#%-3d %-34s %.6f +/-%.6f  %+.4f  %-8s%s' % (
        n, sol, p, sd or 0.0, delta, r.get('verdict', '?'), readable)


def main():
    seen = set()
    if os.path.isdir(LOG_DIR):
        seen = {f for f in os.listdir(LOG_DIR) if f.endswith('.json')}
    pos = os.path.getsize(EVENTS) if os.path.isfile(EVENTS) else 0
    emit('watching %s: %d experiments already logged'
         % (os.path.relpath(LOG_DIR, ROOT).replace(os.sep, '/'), len(seen)))

    while True:
        if os.path.isdir(LOG_DIR):
            for name in sorted(os.listdir(LOG_DIR)):
                if name.endswith('.json') and name not in seen:
                    rec = read(os.path.join(LOG_DIR, name))
                    if rec is None:
                        continue
                    seen.add(name)
                    emit(describe(rec))

        if os.path.isfile(EVENTS):
            size = os.path.getsize(EVENTS)
            if size > pos:
                with open(EVENTS, encoding='utf-8') as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                for line in chunk.splitlines():
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue          # torn final line; next poll gets it
                    if ev.get('kind') not in LOUD:
                        continue
                    emit('[%s] %s' % (ev['kind'], ev.get('detail', '')[:180]))
                    if ev.get('kind') == 'run_end':
                        emit('run finished: %s' % ev.get('detail'))
                        return 0
        time.sleep(POLL)


if __name__ == '__main__':
    sys.exit(main())
