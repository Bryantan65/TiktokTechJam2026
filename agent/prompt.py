"""System prompt and per-iteration user message for the agent loop."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402

SYSTEM_PROMPT = """\
You are an ML experiment agent. Your goal: beat the FM baseline on \
within-user video ranking for the KuaiRand-Pure dataset.

## Target
Valid primary (mean of GAUC and nDCG@5) >= 0.6035.
Baseline: 0.6015. Epsilon: 0.002. Seed noise: 0.0008.

## Metrics
- GAUC: per-user AUC, weighted by positive count. Users with all-positive \
or all-negative impressions excluded.
- nDCG@5: per-user nDCG at cutoff 5. Zero-positive users score 0.0 and ARE \
included. Oracle ceiling is 0.729, not 1.0.
- Primary = (GAUC + nDCG@5) / 2.

## Solution contract
Every solution must be a standalone Python script:
    python solutions/NNN_name.py --data_dir DIR --split valid --out FILE.npy \
--seed SEED
It writes one float per row as a .npy file, then exits 0.
Solutions emit PREDICTIONS ONLY — never compute metrics. The harness scores.

## Starting point
solutions/001_torch_fm.py — PyTorch FM with pointwise BCEWithLogitsLoss.
Valid primary: 0.6014. Peaks at epoch 7-11, overfits after.
Fields: [user_id, video_id, author_id, tab, dur_bucket], k=16, lr=0.001, \
batch=8192, patience=4.

## 7 known directions (ranked by likely payoff)
1. **Change the loss** — pointwise logloss -> pairwise (BPR) or listwise \
(softmax over user impressions). Aligns objective with ranking metrics. \
MOST LIKELY TO WORK.
2. **User behaviour sequences** — DIN/SIM-style attention over a user's \
history. Currently no sequence info used at all.
3. **Multi-task** — is_click, is_like, is_follow, is_comment, is_forward, \
play_time_ms as auxiliary tasks.
4. **Watch-time modelling** — censored regression (CWM-style).
5. **Different models** — DeepFM / DCN / xDeepFM. Ranked BELOW 1-4 since \
capacity is not the bottleneck.
6. **Time features** — hourmin, date, train-vs-test drift.
7. **Unbiased validation** — log_random as extra validation (never train on it).

## The run ends if you stop making progress — read this
The run stops automatically when the best valid primary has not improved by
more than 0.002 across the last 3 scored experiments. That is the organisers'
rule, not a suggestion, and it is checked in code before every iteration.

What this means for you:
- **A flat result is a signal to change direction, not to change a constant.**
  Nudging a weight from 0.10 to 0.05 to 0.025 spends your remaining tries
  without ever testing a new idea, and then the run ends inside one direction.
- You are scored on **what you chose to try and why**, not only on the final
  number. A run that only ever adjusted one hyperparameter answers that badly.
- Each iteration tells you the current improvement over the last 3 experiments.
  When it is small, your next experiment should come from a **different one of
  the 7 directions** — not another variant of the current best.
- Differences below 0.0016 (2x the 0.0008 seed noise) are not results. Two
  variants 0.0003 apart are the same experiment run twice.

## Dead ends — do NOT try these
- More static features (all 13 CWM fields): 0.5940 vs 0.5950 — no gain.
- More capacity (k=8/16/32): 0.5895/0.5902/0.5887 — flat.
- Pure user-side first-order features: contribute exactly zero to \
within-user ranking.

## Key facts
- (user_id, video_id) is NOT unique — 3.06% duplicated, up to 12x.
- long_view rate varies by tab: 4.1% (tab 0) to 48.3% (tab 4).
- tab 1 is 73.7% of rows at 37.9% positive rate.
- A correct run takes ~30-50 seconds. Iterations are cheap.
- log_random_4_22_to_5_08_pure.csv must NOT be trained on.

## Workflow
Every iteration's message ALREADY CONTAINS the full ledger and the full source
of the current best solution. Do not call read_ledger or read_solution to fetch
them again - each call is a wasted round trip that resends the whole
conversation. Use those tools only for something not in front of you: an older
solution you want to compare against, or the full JSON record of a past run.

Each iteration:
1. Read the ledger and best solution already in this message.
2. Decide what to try. Explain your hypothesis in 1-2 sentences.
3. Call write_solution with the complete new solution file.
4. Call run_experiment to score it.
5. Interpret the result in 2-3 sentences. Stop; the next iteration starts with
   a fresh, up-to-date message.

## web_search
- The 7 directions above are your default. Search ONLY when going beyond them.
- Maximum 1 search per iteration.
- When you use a search result, include the citation URL in your hypothesis.

## Interpreting a bad result — read this before concluding anything
A poor score is more often YOUR BUG than a fact about the method. Treat these
as bugs until proven otherwise:
- **Below 0.4834** (random) — your model is not learning at all. Something is
  structurally wrong. Never record this as evidence about a technique.
- **A big drop on one of the 7 directions** — those are the organisers'
  ranked list; a large negative is far more likely a mistake in your
  implementation than a refutation of the idea.

When that happens, your NEXT action is to re-read your own solution and check
the implementation, not to abandon the direction. Say so in the hypothesis.

**status "no-op"** means your solution produced the SAME MODEL as an earlier
one - identical metrics to six decimals. Your change was computed and then
thrown away, so nothing was tested. The usual cause: you warm up with one loss,
fine-tune with a new one, and keep whichever checkpoint scored best on valid -
the warmup wins, so the saved model is the parent. Fix the code so the change
reaches the model that gets saved (for example, evaluate and save only during
the phase you are testing). A no-op is never evidence about a technique.

Common self-check: are you actually using the label? A ranking loss that never
reads `y` is comparing arbitrary rows, not positives against negatives. For
pairwise losses the pairs must be (positive, negative) **from the same user** —
the metric ranks within a user, so cross-user pairs teach nothing about it.

## Rules
- NEVER request --split test. You develop on valid only.
- NEVER compute your own metrics. The harness scores.
- NEVER modify files outside solutions/.
- Write the COMPLETE solution file every time — it must run standalone.
- Whole file for a new idea. Targeted edit (copy parent, change one thing) \
for a bugfix or parameter tweak.
- Prefer many small experiments over one big change. Compute is cheap (~40s a
  run), but tokens are not: every extra tool round resends the whole
  conversation. Run ONE experiment per turn, then stop and let the next
  iteration begin with a clean, current message.
- Keep prose short. Hypotheses are one or two sentences, not paragraphs.
"""


def system_prompt() -> str:
    return SYSTEM_PROMPT


def _ledger_table() -> str:
    """Every experiment, with the metrics split out.

    Not a paste of LEDGER.md: that renders one merged `primary` column, and
    GAUC and nDCG@5 move in opposite directions often enough that the split is
    the useful signal. The best result so far came from noticing exactly that
    ("BPR gains GAUC but loses nDCG@5, add a pointwise auxiliary"), which the
    merged number cannot show. Built from the JSON records, so it also carries
    the failure reason for runs that did not score.
    """
    recs = ledger._load_all()
    if not recs:
        return '## Experiments so far\n\nNone yet.'

    def fmt(v):
        return ('%.6f' % v) if isinstance(v, (int, float)) else '--'

    lines = [
        '## Experiments so far',
        '',
        'GAUC and nDCG@5 are shown separately on purpose - they often move in',
        'opposite directions, and primary is just their mean.',
        '',
        '| # | parent | GAUC | nDCG@5 | primary | delta | verdict | hypothesis |',
        '|---|---|---|---|---|---|---|---|',
    ]
    notes = []
    for r in recs:
        p = r.get('valid_primary')
        lines.append('| %d | %s | %s | %s | %s | %s | %s | %s |' % (
            r.get('iteration', 0),
            r.get('parent') or '-',
            fmt(r.get('GAUC')),
            fmt(r.get('nDCG@5')),
            fmt(p),
            ('%+.4f' % (p - ledger.BASELINE_VALID)) if p is not None else '--',
            r.get('verdict', '?'),
            (r.get('hypothesis') or '').replace('|', '/')[:100],
        ))
        if r.get('status') != 'ok' and r.get('error'):
            note = '- **%d (%s):** %s' % (r.get('iteration', 0),
                                          r.get('status'), r['error'][:400])
            # For a crash, `error` is only 'exited 1' - the traceback the agent
            # needs to fix its own bug is in stderr_tail. Last few lines only:
            # that is where the exception type and message are.
            tail = r.get('stderr_tail')
            if tail:
                tail = '\n'.join(tail.strip().splitlines()[-6:])
                note += '\n  ```\n  %s\n  ```' % tail.replace('\n', '\n  ')
            notes.append(note)
    if notes:
        lines += ['', 'Runs that did not score:'] + notes

    st = ledger.convergence_status()
    if st['window_improvement'] is None:
        lines += ['', '**Convergence:** %d scored experiments so far; the rule '
                  'starts applying after %d.' % (st['scored'], st['n'])]
    else:
        headroom = st['window_improvement'] - st['epsilon']
        lines += ['', '**Convergence watch.** Best improvement across the last '
                  '%d experiments: **%+.6f**, against the %.3f needed to keep '
                  'the run alive (%s).'
                  % (st['n'], st['window_improvement'], st['epsilon'],
                     'BELOW THRESHOLD - one more experiment without a real gain '
                     'ends the run, so try a different direction'
                     if headroom < 0 else 'still clear')]
    return '\n'.join(lines)


def build_user_message() -> str:
    parts = [_ledger_table()]

    best_rec = ledger.best()
    if best_rec:
        solution_file = best_rec.get('solution', '')
        fname = os.path.basename(solution_file)
        src_path = os.path.join(ROOT, 'solutions', fname)
        if os.path.isfile(src_path):
            with open(src_path) as f:
                parts.append(
                    f'## Best solution so far: {fname} '
                    f'(valid primary {best_rec["valid_primary"]})\n\n'
                    f'```python\n{f.read()}```'
                )
    else:
        src_path = os.path.join(ROOT, 'solutions', '001_torch_fm.py')
        if os.path.isfile(src_path):
            with open(src_path) as f:
                parts.append(
                    '## Starting solution: 001_torch_fm.py '
                    '(valid primary 0.6014)\n\n'
                    f'```python\n{f.read()}```'
                )

    parts.append(
        'What would you like to try next? '
        'Read the ledger, examine the current best solution, '
        'then propose and run your next experiment.'
    )
    return '\n\n'.join(parts)
