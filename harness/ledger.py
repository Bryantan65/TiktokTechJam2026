"""Durable memory for the agent loop.

Two tiers, both on disk so a restarted agent resumes with everything it knew:
  logs/iterations/NNN.json   full record of one experiment
  LEDGER.md                  one line per experiment, read in full every turn

The ledger stays small enough to keep entirely in context. No retrieval: an
agent that "misses" a past experiment re-runs it and records the same
conclusion twice. For an experiment log, completeness beats relevance.
"""
import hashlib
import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'logs', 'iterations')
LEDGER = os.path.join(ROOT, 'LEDGER.md')

# Official baseline, reproduced on this machine (see CLAUDE.md).
BASELINE_VALID = 0.6015
EPSILON = 0.002          # official convergence threshold; also the accept gate
N_CONVERGE = 3           # official: 3 consecutive iterations below epsilon

HEADER = """# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`delta` is validation primary against the reproduced official baseline
(**0.6015**). A result counts only at **>= +0.002** (the official epsilon).

| # | parent | hypothesis | valid | delta | verdict | by |
|---|---|---|---|---|---|---|
"""


def next_iteration():
    os.makedirs(LOG_DIR, exist_ok=True)
    used = [int(f[:4]) for f in os.listdir(LOG_DIR)
            if f.endswith('.json') and f[:4].isdigit()]
    return max(used) + 1 if used else 1


def source_hash(path):
    """Fingerprint of a solution's source, so an identical rerun is detectable."""
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def find_by_hash(h):
    """A previous record with this exact source, or None."""
    if not os.path.isdir(LOG_DIR):
        return None
    for name in sorted(os.listdir(LOG_DIR)):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(LOG_DIR, name)) as fh:
                rec = json.load(fh)
        except (ValueError, OSError):
            continue
        if rec.get('source_hash') == h:
            return rec
    return None


def verdict(primary, status):
    if status != 'ok' or primary is None:
        return 'failed'
    delta = primary - BASELINE_VALID
    if delta >= EPSILON:
        return 'KEPT'
    if delta <= -EPSILON:
        return 'worse'
    return 'noise'


def converged(n=N_CONVERGE):
    """True when the last n verdicts all failed to clear epsilon.

    Uses the official rule: n consecutive iterations without an improvement
    greater than epsilon.
    """
    recs = recent(n)
    if len(recs) < n:
        return False
    return all(r.get('verdict') not in ('KEPT',) for r in recs)


def write(record):
    # Set here rather than at each call site so an early return cannot leave a
    # record without a verdict.
    record.setdefault('verdict',
                      verdict(record.get('valid_primary'), record.get('status')))
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, '%04d.json' % record['iteration'])
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=2)

    if not os.path.exists(LEDGER):
        with open(LEDGER, 'w') as fh:
            fh.write(HEADER)

    p = record.get('valid_primary')
    line = '| %d | %s | %s | %s | %s | %s | %s |\n' % (
        record['iteration'],
        record.get('parent') or '-',
        (record.get('hypothesis') or '').replace('|', '/')[:90],
        ('%.4f' % p) if p is not None else '--',
        ('%+.4f' % (p - BASELINE_VALID)) if p is not None else '--',
        record.get('verdict', '?'),
        record.get('by', 'agent'),
    )
    with open(LEDGER, 'a') as fh:
        fh.write(line)
    return path


def recent(n=3):
    """Full records for the last n experiments. Older ones are represented by
    their ledger line only, which keeps context constant-size."""
    if not os.path.isdir(LOG_DIR):
        return []
    names = sorted(f for f in os.listdir(LOG_DIR) if f.endswith('.json'))
    out = []
    for name in names[-n:]:
        try:
            with open(os.path.join(LOG_DIR, name)) as fh:
                out.append(json.load(fh))
        except (ValueError, OSError):
            continue
    return out


def best():
    """The highest-scoring successful record, or None. This is the node a new
    experiment branches from."""
    if not os.path.isdir(LOG_DIR):
        return None
    top = None
    for name in sorted(os.listdir(LOG_DIR)):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(LOG_DIR, name)) as fh:
                rec = json.load(fh)
        except (ValueError, OSError):
            continue
        if rec.get('status') != 'ok' or rec.get('valid_primary') is None:
            continue
        if top is None or rec['valid_primary'] > top['valid_primary']:
            top = rec
    return top


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')
