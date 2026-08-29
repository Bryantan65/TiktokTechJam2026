"""A training-only holdout, so an idea can be screened without spending a
public-validation experiment.

The agent develops against `valid`, and `valid` is also what picks the
early-stopping epoch inside every experiment. Selecting the best of thirty
experiments on the same split that chose each one's checkpoint is two rounds of
selection on one set of labels. It has held up so far - our best solution moved
+0.0040 on valid and +0.0039 on test - but the pressure grows with the length of
the search, and every hunch currently costs a real experiment to test.

This module cuts the TRAIN window in two by date: earlier days to fit on, later
days to score on. Same direction as the official split (fit on the past, predict
the future), same row format, so `encode()` and every existing solution pattern
work unchanged.

    from devdata import load          # instead of `from data import load`
    splits = load(data_dir)           # {'train': ..., 'valid': ...}

There is deliberately no 'test' key. A solution cannot reach for test data here
even by accident, because in this view it does not exist.

WHAT THIS MODULE DOES NOT DO
It reports nothing about any particular feature. No coverage tables per field,
no label lift, no ranked list of what looks promising. That would be a person's
research findings smuggled into a tool, and the point of the harness is that the
agent's own candidate code decides what gets tested. `describe()` reports the
shape of the data and the types of its fields - the things you need to know to
write correct code, not the things you want to know about the problem.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'kuairand-starter-kit')
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from data import load as _official_load       # noqa: E402  unmodified

# How many of the last training days to hold out. Read from the environment so
# the harness and the solution cannot disagree: the harness scores predictions
# against its own copy of the holdout, and if the two derived different cuts the
# rows would not line up. run.py sets this before launching a solution.
HOLDOUT_DAYS = int(os.environ.get('HARNESS_DEV_HOLDOUT_DAYS', 5))


def load(data_dir, holdout_days=None):
    """Official train split, re-cut by date into fit / holdout.

    Returns {'train': rows, 'valid': rows} in the official row format:
    (date, user_id, video_id, author_id, tab, duration_ms, label).

    Note the ids are STRINGS, as the official loader returns them. Mixing them
    with ints read from a CSV via pandas produces dictionary lookups that
    silently miss on every row - it does not raise, it just returns nothing and
    the feature reads as absent everywhere. `describe()` prints the type of each
    field for this reason.
    """
    rows = _official_load(data_dir)['train']
    n = HOLDOUT_DAYS if holdout_days is None else holdout_days
    days = sorted({r[0] for r in rows})
    if n < 1 or n >= len(days):
        raise ValueError('holdout_days must be 1..%d, got %r' % (len(days) - 1, n))
    cut = days[-n]
    return {'train': [r for r in rows if r[0] < cut],
            'valid': [r for r in rows if r[0] >= cut]}


_FIELDS = ('date', 'user_id', 'video_id', 'author_id', 'tab',
           'duration_ms', 'label')


def describe(splits):
    """Shape and data-contract diagnostics. Prints and returns a dict.

    Deliberately generic: row counts, date ranges, positive rate, the type and
    an example of every field, and how much of the holdout the fit window has
    seen at all. Enough to catch a broken join or a type mismatch cheaply;
    nothing that does your feature selection for you.
    """
    out = {}
    for name in ('train', 'valid'):
        rows = splits[name]
        dates = [r[0] for r in rows]
        labels = [r[6] for r in rows]
        out[name] = {
            'rows': len(rows),
            'dates': (min(dates), max(dates)) if rows else None,
            'users': len({r[1] for r in rows}),
            'videos': len({r[2] for r in rows}),
            'positive_rate': round(sum(labels) / len(labels), 4) if rows else None,
        }
        print('%-6s %8d rows  %s..%s  %6d users  %5d videos  %.1f%% positive'
              % (name, out[name]['rows'], out[name]['dates'][0],
                 out[name]['dates'][1], out[name]['users'], out[name]['videos'],
                 100 * out[name]['positive_rate']))

    if splits['train']:
        ex = splits['train'][0]
        print('\nfield types (a mismatch here fails silently, not loudly):')
        for f, v in zip(_FIELDS, ex):
            print('   %-12s %-6s example %r' % (f, type(v).__name__, v))
        out['field_types'] = {f: type(v).__name__ for f, v in zip(_FIELDS, ex)}

    tr, va = splits['train'], splits['valid']
    seen_u = {r[1] for r in tr}
    seen_v = {r[2] for r in tr}
    seen_p = {(r[1], r[2]) for r in tr}
    n = len(va) or 1
    out['coverage'] = {
        'user': round(sum(r[1] in seen_u for r in va) / n, 4),
        'video': round(sum(r[2] in seen_v for r in va) / n, 4),
        'pair': round(sum((r[1], r[2]) in seen_p for r in va) / n, 4),
    }
    print('\nholdout rows whose ... appeared in the fit window:')
    for k in ('user', 'video', 'pair'):
        print('   %-6s %6.2f%%' % (k, 100 * out['coverage'][k]))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data_dir',
                    default=os.path.join(ROOT, 'rec_datasets', 'KuaiRand-Pure', 'data'))
    ap.add_argument('--holdout_days', type=int, default=None)
    a = ap.parse_args()
    describe(load(a.data_dir, a.holdout_days))
