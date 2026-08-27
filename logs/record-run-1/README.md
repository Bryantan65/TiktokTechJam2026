# Record run 1 — converged, valid, but breadth-only

2026-08-27, 16:48:52 → 17:17:29. 28 minutes, $1.45, 8 experiments, zero
failures. **A legitimate autonomous run**: one uninterrupted process, no human
intervention, terminated on the organisers' own rule rather than a budget cap.

Kept as a real result *and* as the evidence for the search-policy fix.

## Result

| # | parent | solution | valid | +/- | delta | direction |
|---|---|---|---|---|---|---|
| 1 | - | `001_torch_fm.py` | 0.601413 | 0.000154 | -0.0001 | control, human-run |
| 2 | 1 | `002_bpr_fm.py` | 0.602871 | 0.000687 | +0.0014 | 1 loss |
| 3 | 2 | `003_listwise_fm.py` | 0.598912 | 0.000284 | -0.0026 | 1 loss |
| 4 | 2 | `004_time_bpr_fm.py` | 0.602617 | 0.000265 | +0.0011 | 6 time |
| 5 | 2 | `005_multitask_bpr_fm.py` | 0.602653 | 0.000393 | +0.0012 | 3 multi-task |
| 6 | 2 | `006_din_history_bpr_fm.py` | 0.602903 | 0.000591 | +0.0014 | 2 sequences |
| 7 | 2 | `007_watchtime_pairwise_fm.py` | 0.563960 | 0.000458 | **-0.0375** | 4 watch-time |
| 8 | 2 | `008_deepfm_bpr.py` | **0.603124** | 0.000494 | **+0.0016** | 5 DeepFM |

Stopped on `converged`: best improved by **+0.000253** across #6-#8, against
ε = 0.002. Six of the seven directions tested. 344k in / 29k out, 1371 s compute.

## The finding

Only the loss change does anything.

```
BPR alone         0.602871
+ time            0.602617
+ multi-task      0.602653
+ DIN sequences   0.602903
+ DeepFM          0.603124
                  ─────────
span              0.00025 — inside every one of their spreads
```

Time features, auxiliary feedback heads, sequence attention and extra capacity
each contributed **nothing measurable** on top of BPR, and each was implemented
correctly rather than crashed. That independently reproduces the organisers'
"the bottleneck is neither features nor capacity" and extends it to sequences
and multi-task.

The two negatives are informative rather than noisy. Listwise softmax assumes
one relevant item per list, but this data is 33% positive, so a user's positives
compete against each other. Raw watch time is the wrong target because
`long_view` is watch time **relative to duration** — ranking by raw play time
systematically favours long videos.

Also confirmed: `web_search` fired once per new direction, six times, all
logged; the data-contract prompt fix prevented the `load()`-returns-tuples crash
that killed the equivalent experiment in shakedown-02.

## Why it is not the submission

Look at the `parent` column. **Six consecutive experiments all branch from node
2.** Nothing was ever refined.

```
#1 <- None
#2 <- 1
#3 <- 2   #4 <- 2   #5 <- 2   #6 <- 2   #7 <- 2   #8 <- 2
```

That is a star of depth 2, not a search tree. Every direction got exactly one
first-draft attempt and was then abandoned — including DIN, multi-task and
censored watch-time, three of the hardest methods on the list, where a first
draft is evidence about the draft rather than the method. Iteration 7's -0.0375,
the single most surprising result of the run, was never diagnosed.

**The cause was the prompt, not the agent.** It said:

> When it is small, your next experiment should come from a **different one of
> the 7 directions** — not another variant of the current best.

Every result was "small" (+0.0011 to +0.0016, all under ε), so the agent
followed that instruction exactly and correctly: pivot every time, refine never.
That line was written to stop shakedown-02's grinding, and it over-corrected.

```
shakedown-02   all depth, no breadth   12 experiments, 1 direction, 5 weight tweaks
record-run-1   all breadth, no depth    8 experiments, 6 directions, 0 refinements
```

Neither is search. Fixed by giving the agent an explicit action each iteration
(`draft` / `improve` / `debug`), the rule that a direction is not exhausted
until it has had a working implementation *and* one refinement, and a **search
shape** line in every prompt that counts children per parent — so a star is
visible as a star while it is happening.

Convergence itself was correct. By the time it fired there was a genuine
plateau. But note it fired at exactly 8, the first moment the floor allowed, so
what depth would have found remains unknown.
