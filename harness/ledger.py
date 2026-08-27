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
import sys
import tempfile
from datetime import datetime

if sys.platform == 'win32':
    import msvcrt

    def _lock(fh):
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(fh):
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fh):
        fcntl.flock(fh, fcntl.LOCK_EX)

    def _unlock(fh):
        fcntl.flock(fh, fcntl.LOCK_UN)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'logs', 'iterations')
LEDGER = os.path.join(ROOT, 'LEDGER.md')
EVENTS = os.path.join(ROOT, 'logs', 'events.jsonl')

# Official baseline, reproduced on this machine (see CLAUDE.md).
BASELINE_VALID = 0.6015
EPSILON = 0.002          # official convergence threshold; also the accept gate
N_CONVERGE = 3           # official: 3 consecutive iterations below epsilon

HEADER = """# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`valid` is the **mean** validation primary across seeds; `+/-` is its standard
deviation across those seeds. `delta` is against the reproduced official
baseline (**0.6015**); a result counts only at **>= +0.002** (the official
epsilon). Two rows closer together than their `+/-` have not been told apart.

| # | parent | hypothesis | valid | +/- | delta | verdict | by |
|---|---|---|---|---|---|---|---|
"""


def log_event(kind, detail, **extra):
    """Append one line to the chronological run log.

    The experiment records answer "what was tried and what did it score". They
    cannot answer "did anything go wrong, and did the run carry on" - a failure
    that happens between experiments (a 500 from the API, a retry, a corrupt
    record) touches no experiment and so appears nowhere. Robustness is scored
    on exactly that: "how it handles one - recovering, retrying, or routing
    around a failed step", and section 2.4 requires each iteration to record
    "any error / recovery events". Printing them loses them when the terminal
    closes.

    Append-only JSONL, one event per line, so a crash mid-write costs at most
    the last line and never corrupts what came before.
    """
    rec = {'ts': stamp(), 'kind': kind, 'detail': str(detail)[:1000]}
    rec.update(extra)
    os.makedirs(os.path.dirname(EVENTS), exist_ok=True)
    try:
        with open(EVENTS, 'a', encoding='utf-8') as fh:
            _lock(fh)
            try:
                fh.write(json.dumps(rec) + '\n')
            finally:
                _unlock(fh)
    except OSError:
        pass                 # logging must never be the thing that kills a run
    return rec


def events(kinds=None, since_iteration=None):
    """Read back the run log, optionally filtered. Used for the writeup."""
    if not os.path.isfile(EVENTS):
        return []
    out = []
    with open(EVENTS, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue     # a torn final line from a hard kill; skip it
            if kinds and rec.get('kind') not in kinds:
                continue
            if (since_iteration is not None
                    and (rec.get('iteration') or 0) < since_iteration):
                continue
            out.append(rec)
    return out


def _write_json_atomic(path, obj):
    """Write via a temp file in the same directory, then rename.

    os.replace is atomic on both POSIX and Windows, so a reader either sees the
    old complete file or the new complete one. Writing in place means a crash
    partway through leaves truncated JSON, which _load_all() then skips - the
    experiment disappears from the ledger without anything saying so.
    """
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def next_iteration():
    os.makedirs(LOG_DIR, exist_ok=True)
    used = []
    for f in os.listdir(LOG_DIR):
        stem, ext = os.path.splitext(f)
        if ext == '.json' and stem.isdigit():
            used.append(int(stem))
    return max(used) + 1 if used else 1


def source_hash(path):
    """Fingerprint of a solution's source, so an identical rerun is detectable."""
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


_hash_index: dict[str, dict] = {}
_hash_index_loaded = False


def _load_hash_index():
    global _hash_index_loaded
    if _hash_index_loaded:
        return
    _hash_index_loaded = True
    if not os.path.isdir(LOG_DIR):
        return
    for name in sorted(os.listdir(LOG_DIR)):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(LOG_DIR, name)) as fh:
                rec = json.load(fh)
        except (ValueError, OSError):
            continue
        h = rec.get('source_hash')
        if h:
            _hash_index[h] = rec


def find_by_hash(h):
    """A previous record with this exact source, or None."""
    _load_hash_index()
    return _hash_index.get(h)


def find_twin(predictions_hash, metrics, exclude_iteration=None):
    """A previous record that produced the same model, or None.

    Different code producing the same output means nothing was actually tested.
    Seen in practice: solutions that warm up with BCE, fine-tune with a new
    loss, then keep whichever checkpoint scored best - the warmup always won,
    so the final model was the parent and three separate "experiments" scored
    0.601400 to six decimal places. Those entered the ledger as evidence that
    the new losses did not help, which is not what happened.

    Bit-identical predictions are the strong signal, but two runs of the same
    code in separate processes differ in the last float bits (torch reductions
    are not deterministic across processes), so that alone misses real no-ops.
    Identical GAUC *and* nDCG@5 to six decimals is the practical test: three
    independent metrics agreeing exactly is not a coincidence.
    """
    for rec in _load_all():
        if exclude_iteration is not None and rec.get('iteration') == exclude_iteration:
            continue
        if rec.get('status') not in ('ok', 'no-op'):
            continue
        if predictions_hash and rec.get('predictions_hash') == predictions_hash:
            return rec, 'identical predictions'
        if all(rec.get(k) is not None and rec.get(k) == metrics.get(k)
               for k in ('GAUC', 'nDCG@5')):
            return rec, 'identical GAUC and nDCG@5'
    return None, None


_reported_corrupt: set[str] = set()


def _load_all():
    if not os.path.isdir(LOG_DIR):
        return []
    out = []
    for name in sorted(os.listdir(LOG_DIR)):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(LOG_DIR, name)) as fh:
                out.append(json.load(fh))
        except (ValueError, OSError) as exc:
            # Skipping keeps the run alive, but skipping *silently* means an
            # experiment vanishes and nothing ever says so. Report each bad
            # file once - _load_all is called several times per iteration.
            if name not in _reported_corrupt:
                _reported_corrupt.add(name)
                log_event('corrupt_record',
                          '%s: %s: %s' % (name, type(exc).__name__, exc),
                          file=name)
    return out


def verdict(primary, status):
    if status == 'no-op':
        return 'no-op'          # not a result; the change never reached the model
    if status == 'duplicate':
        return 'duplicate'
    if status != 'ok' or primary is None:
        return 'failed'
    delta = primary - BASELINE_VALID
    if delta >= EPSILON:
        return 'KEPT'
    if delta <= -EPSILON:
        return 'worse'
    return 'noise'


def _scored():
    """Experiments that actually ran and produced a number, oldest first.

    Errors and crashes are skipped - three crashes in a row should not end the
    search. 'no-op' records tested nothing (the change was discarded before it
    reached the model), so they are not evidence that the search has run out of
    ideas either.
    """
    return [r for r in _load_all()
            if r.get('status') == 'ok'
            and r.get('verdict') != 'no-op'
            and r.get('valid_primary') is not None]


def converged(n=N_CONVERGE):
    """True when the score has not improved by more than epsilon in n tries.

    The official rule: "converged when validation score has not improved by
    more than a small threshold epsilon over the last N consecutive
    iterations". *Improved* means against the best so far - so this compares
    the best of the last n experiments with the best of everything before them.

    It deliberately does NOT reuse verdict(). verdict() answers a different
    question - "is this good enough to count as beating the baseline?" - and
    measures against the fixed 0.6015. The two agree only while the agent is
    below target. Once it clears 0.6035 every result is 'KEPT' forever, even
    one that repeats its parent exactly, and convergence could never fire.
    Observed: iterations 10-12 improved by 0.0003 in total, a seventh of
    epsilon, and all three were labelled KEPT.
    """
    recs = _scored()
    if len(recs) <= n:
        return False                 # need a prior best to improve *on*
    window, before = recs[-n:], recs[:-n]
    best_before = max(r['valid_primary'] for r in before)
    best_window = max(r['valid_primary'] for r in window)
    return (best_window - best_before) < EPSILON


def convergence_status(n=N_CONVERGE):
    """How close the run is to being stopped, in the rule's own terms.

    Shown to the agent every iteration. Without it the agent cannot know it is
    on a clock: observed behaviour is five consecutive micro-tweaks of one idea
    spanning 0.0003, which is exactly the pattern this rule exists to end. An
    agent told "you have one try left before the run ends" can change direction
    instead; an agent told nothing keeps adjusting a constant.
    """
    recs = _scored()
    out = {'n': n, 'epsilon': EPSILON, 'scored': len(recs),
           'window_improvement': None, 'converged': False}
    if len(recs) <= n:
        return out
    best_before = max(r['valid_primary'] for r in recs[:-n])
    best_window = max(r['valid_primary'] for r in recs[-n:])
    out['window_improvement'] = round(best_window - best_before, 6)
    out['converged'] = out['window_improvement'] < EPSILON
    return out


def write(record):
    # Set here rather than at each call site so an early return cannot leave a
    # record without a verdict.
    record.setdefault('verdict',
                      verdict(record.get('valid_primary'), record.get('status')))
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, '%04d.json' % record['iteration'])
    _write_json_atomic(path, record)

    h = record.get('source_hash')
    if h:
        _hash_index[h] = record

    if not os.path.exists(LEDGER):
        with open(LEDGER, 'w') as fh:
            fh.write(HEADER)

    p = record.get('valid_primary')
    sd = record.get('primary_std')
    line = '| %d | %s | %s | %s | %s | %s | %s | %s |\n' % (
        record['iteration'],
        record.get('parent') or '-',
        (record.get('hypothesis') or '').replace('|', '/')[:90],
        ('%.4f' % p) if p is not None else '--',
        ('%.4f' % sd) if sd is not None else '--',
        ('%+.4f' % (p - BASELINE_VALID)) if p is not None else '--',
        record.get('verdict', '?'),
        record.get('by', 'agent'),
    )
    with open(LEDGER, 'a') as fh:
        _lock(fh)
        try:
            fh.write(line)
        finally:
            _unlock(fh)
    return path


def annotate(iteration, **fields):
    """Add fields to an already-written record.

    The harness writes the experiment record, but token usage is only known to
    the agent loop, which learns it after the run returns. This lets the loop
    fold that in rather than leaving cost in stdout, where it is lost when the
    terminal closes. Total token spend is a required submission deliverable.
    """
    path = os.path.join(LOG_DIR, '%04d.json' % iteration)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (ValueError, OSError):
        return None
    rec.update(fields)
    _write_json_atomic(path, rec)
    h = rec.get('source_hash')
    if h:
        _hash_index[h] = rec
    return rec


def totals():
    """Cumulative resource usage across every logged iteration, for the
    Feasibility deliverable (total tokens, total compute seconds)."""
    out = {'tokens_in': 0, 'tokens_out': 0, 'cost_usd': 0.0,
           'compute_seconds': 0.0, 'iterations': 0}
    if not os.path.isdir(LOG_DIR):
        return out
    for name in sorted(os.listdir(LOG_DIR)):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(LOG_DIR, name)) as fh:
                rec = json.load(fh)
        except (ValueError, OSError):
            continue
        out['iterations'] += 1
        out['tokens_in'] += rec.get('tokens_in') or 0
        out['tokens_out'] += rec.get('tokens_out') or 0
        out['cost_usd'] += rec.get('cost_usd') or 0.0
        out['compute_seconds'] += rec.get('seconds') or 0.0
    out['cost_usd'] = round(out['cost_usd'], 4)
    out['compute_seconds'] = round(out['compute_seconds'], 1)
    return out


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
