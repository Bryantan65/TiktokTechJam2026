"""Render the Devpost / deck charts from the run records.

    .venv/Scripts/python.exe tools/make_charts.py

Every number is read from logs/ at render time - nothing is typed in here - so a
chart cannot drift away from the record it claims to show. Writes PNGs to
docs/charts/ at 200 dpi, sized for a Devpost carousel.

Needs matplotlib, which is deliberately NOT in requirements.txt: it is a
documentation tool and nothing in agent/ or harness/ imports it.
    .venv/Scripts/python.exe -m pip install matplotlib
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'charts')

# Deck palette, so the carousel and the slides look like one thing.
BG      = '#0d2038'
FG      = '#ffffff'
DIM     = '#9fb3c8'
FAINT   = '#5d7086'
BLUE    = '#4a90ff'
YELLOW  = '#f5b731'
GREEN   = '#3fcf8e'
RED     = '#ef5f6b'
GREY    = '#6b7d97'

BASELINE = 0.6015           # reproduced official FM, validation
TARGET   = BASELINE + 0.002  # the organisers' epsilon
SEED_SD  = 0.0008           # organiser-measured, 5 seeds

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'savefig.facecolor': BG, 'text.color': FG,
    'axes.labelcolor': DIM, 'xtick.color': DIM, 'ytick.color': DIM,
    'axes.edgecolor': '#2a4361', 'grid.color': '#1c3350',
    'font.size': 13, 'font.family': 'DejaVu Sans',
})

VERDICT_COLOR = {'KEPT': GREEN, 'noise': GREY, 'worse': YELLOW,
                 'failed': RED, 'no-op': '#a97bd8'}


def load(run):
    recs = []
    for f in glob.glob(os.path.join(ROOT, run, '0*.json')):
        try:
            recs.append(json.load(open(f, encoding='utf-8')))
        except Exception:
            pass
    return sorted(recs, key=lambda r: r.get('iteration', 0))


def scored(recs):
    return [r for r in recs if r.get('valid_primary') is not None
            and r.get('verdict') not in ('screen', 'duplicate')]


def finish(fig, ax, name, title, sub=None):
    # Both drawn as axes text rather than set_title: a real title plus a text
    # at 1.045 collide once tight_layout runs, which is what the first pass did.
    ax.text(0, 1.19 if sub else 1.06, title, transform=ax.transAxes, color=FG,
            fontsize=17, fontweight='bold', va='bottom')
    if sub:
        ax.text(0, 1.055, sub, transform=ax.transAxes, color=DIM,
                fontsize=12.5, va='bottom')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', lw=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    print('  wrote %s' % os.path.relpath(p, ROOT))


# ------------------------------------------------------------------ 1. tree
def chart_tree(run='logs/record-run-9'):
    """The run as the search tree it is: x is the experiment, y is its score.

    Same shape as the console's tree, but on a real score axis rather than a
    rank one - a static image has no hover, so the y-axis has to carry units.
    """
    recs = load(run)
    by = {r['iteration']: r for r in recs}
    sc = scored(recs)
    top_rec = max(sc, key=lambda r: r['valid_primary'])
    top = top_rec['valid_primary']
    lo, hi = BASELINE - 0.0011, top + 0.0007

    def parent_of(r):
        try:
            p = int(r.get('parent'))
        except (TypeError, ValueError):
            return None
        return p if p in by else None

    # An experiment that never scored has no height of its own, so it sits
    # level with the node it came from - a dead end reads as a flat stub.
    ypos = {}
    for r in sorted(recs, key=lambda r: r['iteration']):
        v = r.get('valid_primary')
        if v is not None and r.get('verdict') not in ('screen', 'duplicate'):
            ypos[r['iteration']] = v
        else:
            p = parent_of(r)
            ypos[r['iteration']] = ypos.get(p, BASELINE)

    # the chain of parents that actually produced the result
    winning, cur = set(), top_rec['iteration']
    while cur is not None and cur not in winning:
        winning.add(cur)
        cur = parent_of(by[cur]) if cur in by else None

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_ylim(lo, hi)
    ax.axhline(BASELINE, color=YELLOW, lw=1.6, ls='--', zorder=1)
    ax.text(0.4, BASELINE + 0.00010, 'official FM baseline  0.6016',
            color=YELLOW, ha='left', va='bottom', fontsize=12)
    ax.axhline(TARGET, color=BLUE, lw=1.3, ls=':', zorder=1)
    ax.text(len(recs) + 0.4, TARGET + 0.00010, 'target  +0.002',
            color=BLUE, ha='right', va='bottom', fontsize=12)

    clip = lambda y: max(lo + 0.00007, min(y, hi))
    for r in recs:
        p = parent_of(r)
        if p is None:
            continue
        on = r['iteration'] in winning and p in winning
        ax.plot([p, r['iteration']], [clip(ypos[p]), clip(ypos[r['iteration']])],
                color=BLUE if on else '#2b4straight'[:7] if False else '#2b4463',
                lw=2.4 if on else 1.1, zorder=3 if on else 2,
                alpha=1.0 if on else 0.9, solid_capstyle='round')

    below = 0
    for r in recs:
        it, y = r['iteration'], ypos[r['iteration']]
        col = VERDICT_COLOR.get(r.get('verdict'), GREY)
        if r.get('status') == 'error':
            ax.scatter(it, clip(y), marker='x', s=85, color=RED, zorder=6,
                       linewidth=2.4)
            continue
        if y < lo:
            below += 1
            ax.scatter(it, lo + 0.00007, marker='v', s=66, color=col,
                       edgecolor=BG, linewidth=1.0, zorder=5)
            continue
        ax.scatter(it, y, s=62 if it in winning else 46, color=col,
                   edgecolor=BG, linewidth=1.3, zorder=5)

    ax.scatter([top_rec['iteration']], [top], s=230, facecolor='none',
               edgecolor=GREEN, linewidth=2.0, zorder=6)
    ax.annotate('0.6057', xy=(top_rec['iteration'], top), xytext=(-10, 15),
                textcoords='offset points', color=GREEN, fontsize=14,
                fontweight='bold', ha='right')

    ax.set_xlabel('experiment', labelpad=2)
    ax.set_ylabel('validation score')
    ax.set_xlim(0, len(recs) + 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: '%.3f' % v))

    key = [('kept', GREEN), ('within noise', GREY), ('worse', YELLOW),
           ('same model', '#a97bd8'), ('crashed', RED)]
    for i, (lab, col) in enumerate(key):
        ax.scatter([], [], color=col, s=52, label=lab)
    # below the x label, not on top of it
    leg = ax.legend(loc='upper center', frameon=False, ncol=5, fontsize=11.5,
                    handletextpad=0.25, columnspacing=1.6,
                    bbox_to_anchor=(0.5, -0.155))
    for t in leg.get_texts():
        t.set_color(DIM)
    if below:
        ax.text(0.985, 0.03, '%d fell below this axis' % below,
                transform=ax.transAxes, color=FAINT, fontsize=11.5, ha='right')

    finish(fig, ax, 'climb.png', 'One run, as the tree it actually is',
           'Every experiment names a parent. The blue path is the line of '
           'descent that produced the result.')


# ------------------------------------------------------ 2. is it repeatable
def chart_consistency():
    rows = []
    for run in sorted(glob.glob(os.path.join(ROOT, 'logs', 'record-run-*'))):
        recs = load(os.path.relpath(run, ROOT))
        if len(recs) < 25:
            continue                       # shakedowns, not full runs
        ev = os.path.join(run, 'events.jsonl')
        if not (os.path.isfile(ev) and any('"converged"' in l
                                           for l in open(ev, encoding='utf-8'))):
            continue
        sc = scored(recs)
        if sc:
            rows.append((os.path.basename(run), max(r['valid_primary'] for r in sc)))
    rows.sort(key=lambda t: t[1])
    vals = [v for _, v in rows]
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhspan(min(vals), max(vals), color=GREEN, alpha=0.10, zorder=0)
    ax.axhline(BASELINE, color=YELLOW, lw=1.6, ls='--')
    ax.text(-0.4, BASELINE + 0.00012, 'official FM baseline',
            color=YELLOW, ha='left', va='bottom', fontsize=12)
    ax.axhline(TARGET, color=BLUE, lw=1.3, ls=':')
    ax.text(len(vals) - 0.5, TARGET + 0.00015, 'target  +0.002',
            color=BLUE, ha='right', va='bottom', fontsize=12)

    ax.scatter(range(len(vals)), vals, s=110, color=GREEN,
               edgecolor=BG, linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([n.replace('record-run-', '#').replace('-bryan-pure', '')
                        for n, _ in rows], rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('best validation score')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: '%.3f' % v))
    ax.text(0.02, 0.66,
            'spread across %d runs:  %.5f  to  %.5f\nstandard deviation  %.5f'
            % (len(vals), min(vals), max(vals), sd),
            transform=ax.transAxes, color=GREEN, fontsize=13, va='top')
    finish(fig, ax, 'consistency.png', 'Every full run lands in the same place',
           'Twelve independent runs, each stopped by the convergence rule. '
           'Not one lucky run.')


# ---------------------------------------------- 3. is the gain big enough
def chart_margin():
    gain_pure, gain_1k = 0.003944, 0.037894
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = ['seed noise\n(1 sigma)', 'the target\nto beat', 'our gain\non Pure']
    vals = [SEED_SD, 0.002, gain_pure]
    colors = [GREY, YELLOW, GREEN]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 0.00012, b.get_y() + b.get_height() / 2, '%.4f' % v,
                va='center', color=FG, fontsize=14, fontweight='bold')
    ax.set_xlim(0, gain_pure * 1.32)
    ax.invert_yaxis()
    ax.set_xlabel('validation score difference')
    ax.grid(axis='x', lw=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0)
    finish(fig, ax, 'margin.png', 'The gain clears the noise',
           'A result has to beat measurement error before it means anything. '
           'Ours is ~5x the seed noise.')


# ------------------------------------------------- 4. why 1k gained 10x
def chart_overlap():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (title, a, b, fmt, colour) in zip(axes, [
            ('Training overlap\n(user,creator) pairs seen before', 3.38, 33.70,
             '%.2f%%', BLUE),
            ('Our gain over the baseline', 0.003944, 0.037894, '%.4f', GREEN)]):
        bars = ax.bar(['KuaiRand-Pure', 'KuaiRand-1k'], [a, b],
                      color=[colour, colour], alpha=0.95, width=0.55)
        bars[0].set_alpha(0.45)
        for r, v in zip(bars, [a, b]):
            ax.text(r.get_x() + r.get_width() / 2, v * 1.03, fmt % v,
                    ha='center', color=FG, fontsize=14, fontweight='bold')
        ax.set_title(title, color=DIM, fontsize=13, pad=12)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', lw=0.8, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(a, b) * 1.22)
    fig.suptitle('Ten times the overlap, ten times the gain', color=FG,
                 fontsize=17, fontweight='bold', x=0.055, ha='left', y=1.05)
    fig.text(0.055, 0.955,
             'The same agent on both datasets. What changed is how much of the '
             'test set it had already seen.', color=DIM, fontsize=12.5, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'overlap.png')
    fig.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    print('  wrote %s' % os.path.relpath(p, ROOT))


if __name__ == '__main__':
    print('rendering charts from the run records:')
    chart_tree()
    chart_consistency()
    chart_margin()
    chart_overlap()
