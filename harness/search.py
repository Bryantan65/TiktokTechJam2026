"""UCT node scoring: which node is worth expanding, not just which scored best.

The agent's search is greedy. The prompt tells it to expand the best node while
it shows signal, and across thirteen runs that is what it does - runs 10 and 11
spent 16 and 12 consecutive experiments on one branch. MLE-bench reports the
same framing: AIDE's greedy tree search reaches 39.6% on MLE-bench Lite while
non-greedy search over the same operators reaches 47.7%.

Greedy fails in a specific way. A node that scored 0.6045 once and was never
touched again is not known to be worse than the incumbent - it is UNMEASURED.
Greedy treats "scored lower" and "barely explored" as the same thing. UCT does
not: it adds a bonus for nodes with few children, so a promising-but-unexplored
node competes with a well-worked incumbent.

    UCT(n) = value(n) + C * sqrt(ln(N) / (1 + children(n)))

`value` is the part that needs care here. Raw primaries live in a range of about
0.006, so normalising by (max-min) would turn seed noise into large differences
and the ranking would chase measurement error. Instead the scale is the
organisers' own epsilon: a node at the current best scores 1.0, a node a full
epsilon below scores 0.0, and everything between is linear. A gap that the
competition calls meaningless produces a value difference this function also
calls small.

This module only RANKS. It does not choose - the agent still names its own
parent, and `uct_rank_of` records where its choice sat so adherence can be
measured after the fact rather than assumed.
"""
import math
import os

EPSILON = 0.002          # official convergence threshold, used here as the value scale

# Exploration weight. Chosen so an unexplored node half an epsilon below the
# best (value 0.5) ties with the best node once that best has ~3 children:
#   1.0 + C*sqrt(ln20/4) == 0.5 + C*sqrt(ln20/1)  ->  C ~= 0.58
# Below ~0.3 the bonus never overturns a ranking and this is greedy with extra
# steps; above ~1.0 it ignores score entirely. Env-overridable so a run can be
# configured without editing code.
C = float(os.environ.get('HARNESS_UCT_C', 0.5))


def _scored(recs):
    """Records carrying a real valid measurement, oldest first.

    Screens ran on the dev holdout and are not comparable; no-ops, duplicates
    and failures never produced a number. None of them is evidence about a node.
    """
    out = [r for r in recs
           if isinstance(r.get('valid_primary'), (int, float))
           and r.get('split', 'valid') == 'valid'
           and r.get('verdict') not in ('screen', 'no-op', 'failed', 'duplicate')]
    return sorted(out, key=lambda r: r.get('iteration') or 0)


def _parent(r):
    try:
        return int(r.get('parent'))
    except (TypeError, ValueError):
        return None


def child_counts(recs):
    """How many experiments have branched off each node, scored or not.

    Deliberately counts every attempt, including crashes. A node whose child
    crashed HAS been visited - the agent spent an experiment there - and
    pretending otherwise would send it straight back to the same place.
    """
    kids = {}
    for r in recs:
        p = _parent(r)
        if p is not None:
            kids[p] = kids.get(p, 0) + 1
    return kids


def rank(recs):
    """Nodes ordered by UCT, best first.

    Returns a list of dicts: iteration, primary, value, children, bonus, uct.
    Empty until at least one node has scored, since there is nothing to rank.
    """
    sc = _scored(recs)
    if not sc:
        return []
    best = max(r['valid_primary'] for r in sc)
    kids = child_counts(recs)
    n = max(len(sc), 2)          # ln(1) is 0, which would zero every bonus
    out = []
    for r in sc:
        gap = best - r['valid_primary']
        value = max(0.0, 1.0 - gap / EPSILON)
        c = kids.get(r['iteration'], 0)
        bonus = C * math.sqrt(math.log(n) / (1 + c))
        out.append({'iteration': r['iteration'],
                    'primary': r['valid_primary'],
                    'value': value,
                    'children': c,
                    'bonus': bonus,
                    'uct': value + bonus,
                    'hypothesis': (r.get('hypothesis') or '')[:70]})
    out.sort(key=lambda d: -d['uct'])
    return out


def uct_rank_of(recs, parent_iteration):
    """Where `parent_iteration` sat in the UCT ranking, 1-based, or None.

    Recorded per experiment so "did the agent follow the ranking" is a number in
    the log rather than an impression from reading hypotheses.
    """
    try:
        p = int(parent_iteration)
    except (TypeError, ValueError):
        return None
    for i, d in enumerate(rank(recs), 1):
        if d['iteration'] == p:
            return i
    return None


def render(recs, top=5):
    """The ranking as prompt text, or '' when there is nothing to say."""
    r = rank(recs)
    if len(r) < 3:
        return ''
    lines = [
        '',
        '**Where the search says to expand next.** Each node scores its own '
        'result plus a bonus for being under-explored - a node with no children '
        'is unmeasured, not known to be worse. `value` is 1.0 at the current '
        'best and 0.0 a full epsilon (0.002) below it.',
        '',
        '```',
        '  node   primary    value  children   bonus    UCT',
    ]
    for d in r[:top]:
        lines.append('  #%-4d %.6f   %.2f   %5d     %.2f   %.2f'
                     % (d['iteration'], d['primary'], d['value'],
                        d['children'], d['bonus'], d['uct']))
    lines += ['```', '']
    head = r[0]
    greedy = max(r, key=lambda d: d['primary'])
    if greedy['iteration'] == head['iteration']:
        # Greedy and UCT agree. Saying "this is not your best scorer" here would
        # be false, and a ranking that misdescribes itself gets ignored.
        lines.append(
            'Top of that list is **#%d**, which is also your best-scoring node '
            'and has %d %s. Greedy and UCT agree this turn, so expanding it is '
            'the right move for both reasons.'
            % (head['iteration'], head['children'],
               'child' if head['children'] == 1 else 'children'))
    else:
        lines.append(
            'Top of that list is **#%d** - *not* your best scorer. #%d scored '
            'higher (%.6f) but already has %d %s, so another child of it is '
            'worth less than a first child of #%d (%.6f, %d %s). Expanding the '
            'best node for the fifth time buys less than expanding one nobody '
            'has touched.'
            % (head['iteration'], greedy['iteration'], greedy['primary'],
               greedy['children'], 'child' if greedy['children'] == 1 else 'children',
               head['iteration'], head['primary'], head['children'],
               'child' if head['children'] == 1 else 'children'))
    lines.append(
        'If you branch from somewhere outside this top %d, say why in your '
        'hypothesis.' % min(top, len(r)))
    return '\n'.join(lines)
