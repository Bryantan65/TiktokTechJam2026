# Shakedown 02

Second debugging run, 2026-08-27. **Not an autonomy demonstration** — kept as
the evidence behind the fixes it produced, and as the before-picture for the
measurement change.

Superseded by the record run in `logs/iterations/`.

## Why it does not count as autonomous

```
1     by=human   agent_iter=None   14:52:23   the control
2-5   by=agent   agent_iter=1      14:54-14:57  process A, one conversation
6     by=agent   agent_iter=None   14:58:58   annotation missing - killed mid-turn
                                   ^ 4m23s gap: stopped, code edited
7-8   by=agent   agent_iter=1      15:03-15:04  process B - counter reset to 1
9-13  by=agent                     15:0x-16:0x  further sessions, more edits between
```

It opens with a human action, there is human intervention inside the loop, and
no process terminated on its own. Iteration 6's missing `agent_iteration` is
itself the evidence: annotation happens after a turn completes, so a null means
the process died first.

## What it found

Scores here are **single-seed** (seed 0). That is the headline finding — see
below.

| # | solution | primary | note |
|---|---|---|---|
| 1 | `001_torch_fm.py` | 0.601400 | human control |
| 2-4 | BPR variants | 0.6030-0.6033 | direction 1 |
| 5-6 | BPR + BCE auxiliary | 0.6035-0.6036 | first to clear epsilon |
| 7 | hard-negative mining | 0.598136 | the one informative failure |
| 8-12 | sampled softmax, aux weight nudged | 0.6037-0.6040 | **resolved nothing** |
| 13 | time features | crashed | `load()` returns tuples, not DataFrames |

### Bugs and design faults this run exposed

1. **`converged()` could never fire.** It borrowed `verdict()`, which measures
   against the fixed 0.6015, so past 0.6035 every result was `KEPT` forever —
   including one repeating its parent exactly. Iterations 10-12 improved by
   0.000089 and all three were `KEPT`.
2. **The agent was never told the convergence rule existed**, which is why it
   spent five iterations nudging one loss weight.
3. **Single-seed scoring was below the resolution needed.** Seed sweep of
   iteration 12: 0.603999 / 0.603210 / 0.602734, std **0.000639** — while
   iterations 8-12 differed by 0.0003 in total. Those five experiments measured
   nothing but seed noise, and the +0.0025 headline was seed 0 being lucky. The
   true gain is **+0.0019**.
4. **API errors that were not rate limits abandoned the iteration** with no
   retry, and a fatal error would have reproduced across every remaining one.
5. **Recovery events were printed, never logged.**
6. **`ledger.write()` was not atomic**, and corrupt records were skipped in
   silence.
7. **`load()`'s row shape was undocumented** to the agent, and `hourmin` /
   the feedback signals are not in it at all.
8. **`web_search` never fired once**, because the prompt told it not to.

### The lesson worth carrying

The grinding in iterations 8-12 looked like an agent failure. It was a
**measurement** failure: the harness reported 0.0003 differences as
differences, and the agent acted rationally on them. Better instructions would
not have fixed it.
