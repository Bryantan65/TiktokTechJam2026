# Record run 2 — converged past target

2026-08-27. 9 iterations, 8 scored, $1.63, 1205 s compute. One uninterrupted
process, no human intervention, terminated on the organisers' convergence rule.
**This is the run.**

## Result

| # | parent | action | solution | valid | +/- | delta | verdict |
|---|---|---|---|---|---|---|---|
| 1 | - | control | `001_torch_fm.py` | 0.601413 | 0.000154 | -0.0001 | human |
| 2 | 1 | draft 1 | `002_bpr_fm.py` | crashed | | | failed |
| 3 | 2 | debug 2 | `003_bpr_fm_fix.py` | 0.603305 | 0.000485 | +0.0018 | noise |
| 4 | 3 | improve 3 | `004_bpr_fm_3neg.py` | 0.602712 | 0.000884 | +0.0012 | noise |
| 5 | 3 | improve 3 | `005_listwise_fm.py` | 0.595860 | 0.000432 | -0.0056 | worse |
| 6 | 3 | improve 3 | `006_bpr_bce_ensemble.py` | 0.603623 | 0.000238 | +0.0021 | KEPT |
| 7 | 6 | improve 6 | `007_two_bpr_bce_ensemble.py` | 0.603933 | 0.000256 | +0.0024 | KEPT |
| 8 | 7 | draft 6 | `008_time_features_two_bpr_bce.py` | **0.604598** | 0.000497 | **+0.0031** | KEPT |
| 9 | 8 | improve 8 | `009_time_richer_buckets.py` | 0.604297 | 0.000061 | +0.0028 | KEPT |

**Submission candidate: `008_time_features_two_bpr_bce.py`, 0.604598 ± 0.000497,
+0.0031 over baseline** — the validation-best checkpoint, which is what the rules
say is scored. Not #9: richer time buckets came in 0.0003 lower, inside the
spread, so they neither helped nor hurt.

## The search actually searched

```
1 → 2 → 3 ─┬─ 4
           ├─ 5
           └─ 6 → 7 → 8 → 9
```

Depth 6, with a real branch point at node 3 where three alternatives were tried
and the best one continued. Compare record-run-1, whose entire tree was
`2,2,2,2,2,2` — six first drafts off one node, nothing refined.

Every action label was used correctly and unprompted:

- **`debug 2`** on the crash — *"convert users and labels to NumPy arrays before
  sorting/fancy indexing. This tests the intended same-user BPR implementation
  rather than a loader bug."* That sentence is the whole reason for the change:
  the agent distinguishing "my implementation was broken" from "the method does
  not work". Record-run-1 had no way to say it.
- **Three `improve 3` branches** — more negatives (flat), listwise (much worse),
  BPR+BCE blend (the first result to clear epsilon).
- **`draft 6`** when the ensemble stopped paying, rather than adding a third
  member. The obvious failure mode here was riding the ensemble dial; it did not.

## Findings

**BPR blended with pointwise BCE is what works.** Plain BPR gives +0.0018;
blending in a calibrated pointwise term gives +0.0021, and averaging two
independent BPR components +0.0024. The ensemble's seed spread is also the
tightest of any agent solution (0.00024 against a typical 0.0005-0.0009), which
is the expected signature of averaging.

**Listwise softmax is genuinely wrong for this data, not a bug.** Three
independently written implementations across three runs: -0.0051, -0.0026,
-0.0056. A full softmax over a user's impressions assumes exactly one relevant
item, but this dataset is 33% positive, so a user's positives compete against
each other and the loss penalises ranking the second one well.

**Time features help on top of the ensemble, weakly.** +0.0007 at ~1.4 sigma —
suggestive, not established. Worth noting record-run-1 tested time features on
plain BPR and found nothing, so this may be an interaction rather than a
standalone effect.

## Known gap in this run's logs

`recovery_events` is `[]` on every record, including iteration 2 which crashed.
The field is populated only by API-level events; a *solution* crash never reaches
it. The information is not lost — the error, traceback, diagnosis and causal link
are all in the iteration records (#2 `status: error` + `stderr_tail`, #3
`parent: 2` + a `debug 2` hypothesis) — but `events.jsonl` shows no trace of the
run's best robustness demonstration. Fixed after this run; see HANDOFF.

## Convergence

```
window  #7,#8,#9   best 0.604598
before  #1..#6     best 0.603623
                        ─────────
improvement             0.000975   (epsilon 0.002)
```

Nine iterations ran but only eight scored — the crash at #2 never counted — so
the 8-experiment floor was met at exactly this check, the rule woke up for the
first time, and fired immediately.

**Worth recording: the agent was still improving when it was stopped.** The final
window went +0.0021 → +0.0024 → +0.0031, real measured progress, but no single
step was as large as epsilon. The rule cannot distinguish steady incremental gain
from being stuck. This is now the second consecutive run to converge at exactly
the floor.
