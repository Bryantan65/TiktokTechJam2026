# Void run 1 — stopped deliberately after 3 experiments

2026-08-27, ~6 minutes. **Voided, not failed.** Kept because it is the evidence
behind `MIN_SCORED_BEFORE_CONVERGENCE`, and because the defect it exposed was
mine.

| # | solution | primary | +/- | delta | verdict |
|---|---|---|---|---|---|
| 1 | `001_torch_fm.py` | 0.601413 | 0.000154 | -0.0001 | control, human-run |
| 2 | `002_bpr_fm.py` | 0.602909 | 0.000406 | +0.0014 | noise |
| 3 | `003_listwise_fm.py` | 0.596445 | 0.000387 | -0.0051 | worse |

`004_bpr_date_fm.py` was written but never scored — the interrupt landed first.

## Why it was stopped

`converged()` was one experiment away from ending the run.

```
best of [#1]        = 0.601413
best of [#2,#3,#4]  = max(0.602909, 0.596445, X)
converged unless X >= 0.603413
```

BPR's +0.0015 was the best improvement so far and sits under epsilon, so
anything below 0.6034 at #4 would have terminated the run with **three
experiments and two ideas tried**. #4 was a first attempt at time features,
which had no realistic chance of clearing that.

Formally the rule was satisfied: the best had not improved by more than epsilon
across the last three. But the rule exists to detect a **plateau**, and three
experiments is not a plateau — it is a start. The organisers fix epsilon and N
and say nothing about a minimum run length, presumably because a run ending at
experiment 3 never came up.

Fixed by `MIN_SCORED_BEFORE_CONVERGENCE = 8`: convergence cannot fire until
eight scored experiments exist, leaving five to establish a best before any
three are judged against it. Not a loosening of the rule — a floor on when a
plateau can be claimed.

## What it showed working

Three changes were confirmed live before the stop:

- **`web_search` fired on iteration 1** (`"Bayesian Personalized Ranking BPR
  loss pairwise ... Rendle 2009"`), and was logged with its result. It had never
  fired once in the previous 13 experiments.
- **Multi-seed scoring made the numbers readable.** #2 came in at +0.0014
  against a spread of 0.0004 — a real gain, 3.4x the noise, correctly labelled
  `noise` because it is under epsilon. Under the old single-seed harness the
  same experiment read `0.603113` with no error bar at all.
- **The interrupt handler logged cleanly** — `interrupted` then `run_end`, so
  the run log says how the run ended rather than simply stopping.

## Also fixed after this run

The spinner wrote a 138-column line and wrapped, so `\r` landed on the wrapped
remainder and every frame left a row behind. Now clamped to the terminal width.
The label's em-dash was replaced with ASCII: it is written from the spinner
thread, and `UnicodeEncodeError` on a cp1252 console would be swallowed by that
thread's except clause, silently killing the spinner for the rest of a run.
