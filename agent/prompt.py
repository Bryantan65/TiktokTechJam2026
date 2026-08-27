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
Each iteration:
1. Call read_ledger to see what has been tried.
2. Call read_solution to read the best or parent solution.
3. Decide what to try. Explain your hypothesis in 1-2 sentences.
4. Call write_solution with the complete new solution file.
5. Call run_experiment to score it.
6. Interpret the result. Plan the next iteration.

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
- Prefer many small experiments over one big change. Iterations are cheap.
"""


def system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_message() -> str:
    parts = []

    if os.path.exists(ledger.LEDGER):
        with open(ledger.LEDGER) as f:
            parts.append('## Current ledger\n\n' + f.read())
    else:
        parts.append('## Current ledger\n\nNo experiments yet.')

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
