# Results summary

Deliverable 4. Final model output, the results table, and the resource usage
required to reach the converged result.

Every number here is read back from `logs/record-run-3/NNNN.json`,
`logs/record-run-3/events.jsonl` and `test_scores.json` — not from memory.

---

## Final model output

```
submission candidate   logs/record-run-3/solutions/027_deepfm_member.py
predictions            submission.csv   (170,588 rows, test split)
schema                 row_id,user_id,video_id,score   per the Starter Kit
validated              python kuairand-starter-kit/submit.py --check
```

The run stopped on the organisers' convergence rule — not on the 50-iteration
cap and not on the wall-clock ceiling. The scored checkpoint is the
validation-best at that point, which is iteration 27.

---

## KuaiRand-Pure — the required benchmark

Scoring formula: `delta(m) = score_agent(m) − score_baseline(m)` for each
metric, then equal-weighted. Baseline is the official FM from
`kuairand-starter-kit/baseline_scores.json`.

### Validation-best

| metric | ours | official FM baseline | delta |
| --- | --- | --- | --- |
| GAUC | 0.672469 | 0.6674 | **+0.005069** |
| nDCG@5 | 0.538518 | 0.5357 | **+0.002818** |
| primary — mean of the two | 0.605493 | 0.6016 | +0.003893 |
| **equal-weighted delta** | | | **+0.003944** |

### Hidden-test equivalent, scored once locally

The test labels ship with the dataset, so this is measured by discipline rather
than withheld by the harness. `harness/score_test.py`, official `evaluate.py`.

| metric | ours | official FM baseline | delta |
| --- | --- | --- | --- |
| GAUC | 0.665391 | 0.6610 | **+0.004391** |
| nDCG@5 | 0.531626 | 0.5282 | **+0.003426** |
| primary — mean of the two | 0.598508 | 0.5946 | +0.003908 |
| **equal-weighted delta** | | | **+0.003908** |

The validation gain transferred: +0.003944 on valid, +0.003908 on test.

Seed spread on this data is 0.0008 (organiser-measured, 5 seeds), and the
target margin was 0.002. The submitted result clears the target on both splits
and on both metrics independently.

### How close to the ceiling is this?

`+0.0039` is uninterpretable without knowing what was available to win.

```
oracle ceiling (valid), requires the labels                   0.8484
perfect knowledge of every video, nothing about the user      0.6197
ours                                                          0.6055
official FM baseline                                          0.6016
```

The gap to 0.8484 is not room. The oracle is computed from the realised labels,
so no model can reach it: the same user shown the same video on a different day
flips 23-35% of the time, and the best a model can do is predict the probability. Personalisation on Pure is close to unavailable:
the median user has 31 training rows covering 29 videos by 29 different
creators, and only **3.38%** of validation rows involve a (user, creator) pair
the model has ever seen — against **33.70%** on KuaiRand-1k. On Pure, 96.6% of
the time the next video is by a creator the model has never seen that user react
to, so any model falls back to popularity, which tops out at 0.6197 with perfect
knowledge and 0.6048 when estimated honestly.

Fifteen independent runs — different mechanism families, different underlying
models — landed within 0.0011 of each other (sd 0.00036). Full analysis with the
supporting measurements is in the README's limitations section.

---

## Bonus benchmarks

**KuaiRand-1k — attempted, `logs-1k/record-run-4`.** The organisers published a
baseline for KuaiRand-Pure only, so we measured one by running the kit's own
`baseline.py` unmodified on 1k data (see `docs/bonus-baselines.md`). Deltas are
per metric, equal-weighted, as the judging formula specifies.

### Validation-best

| metric | ours | kit FM baseline | delta |
| --- | --- | --- | --- |
| GAUC | 0.702723 | 0.6749 | **+0.027823** |
| nDCG@5 | 0.663265 | 0.6153 | **+0.047965** |
| primary | 0.682994 | 0.6451 | +0.037894 |
| **equal-weighted delta** | | | **+0.037894** |

### Test

| metric | ours | kit FM baseline | delta |
| --- | --- | --- | --- |
| GAUC | 0.701692 | 0.6730 | **+0.028692** |
| nDCG@5 | 0.654918 | 0.6049 | **+0.050018** |
| primary | 0.678305 | 0.6390 | +0.039305 |
| **equal-weighted delta** | | | **+0.039355** |

Predictions: `submission_1k.csv`, 4,132,081 rows, validated with the kit's own
`submit.py --check`. Winner: `logs-1k/record-run-4/solutions/040_light_context_member.py`.

`submission_1k.csv` is **not in the repository**: 4.1M rows is 118MB, over
GitHub's 100MB per-file limit. The Pure submission is small enough to track and
is. Rebuild it from the winning solution, which is tracked:

```
python harness/make_submission.py \
    logs-1k/record-run-4/solutions/040_light_context_member.py \
    --split test --out submission_1k.csv \
    --data_dir rec_datasets/KuaiRand-1K/data
```

**The 1k delta is ten times the Pure delta** (+0.0394 against +0.0039), which is
what the structural analysis predicts: 33.70% of 1k validation rows involve a
(user, creator) pair seen in training, against 3.38% on Pure. Personalisation is
available on 1k and very nearly unavailable on Pure.

### Reproducing it - read this before trying

The winner is **not** self-contained. It loads six member predictions from
`pred_cache/` that iteration 29 produced, and on a cache miss it substitutes
cheaper stand-in models rather than retraining them. It then still runs, still
prints a plausible score, and is no longer the model that was measured. Since
`pred_cache/` is gitignored, a fresh checkout hits exactly that path.

Run iteration 29 first, for each split you want:

```
PYTHONPATH=harness python logs-1k/record-run-4/solutions/029_aux_soft_multitask.py   --data_dir rec_datasets/KuaiRand-1K/data --split valid --seed 0 --out /tmp/x.npy

PYTHONPATH=harness python logs-1k/record-run-4/solutions/040_light_context_member.py   --data_dir rec_datasets/KuaiRand-1K/data --split valid --seed 0 --out preds.npy
```

Rebuilt this way the winner reproduces at **0.682994** against the ledger's
0.682683 - a gap of +0.000311, inside 1k's own seed spread of 0.00058 - with
zero fallback members used on either split. That check is what this number rests
on.

**Label-leakage check.** The winner's 26-solution lineage was traced by parent
pointer. No solution reads the label by index from the split it predicts; one
(`029`) opens a raw CSV, and only `log_standard_4_08_to_4_21`, the training
window, using the auxiliary feedback signals as training targets and sample
weights. The detector was validated against `logs-1k/record-run-2`, whose winner
is a deliberate label oracle scoring 0.997444 - it flags that run in 16 places,
including the winner.

**KuaiRand-27k — not attempted.** Assessed and set aside deliberately: it is
wider than 1k rather than deeper (~11,800 impressions per user in both), and the
one mechanism it could uniquely supply — deep per-user history — was tested at
27k's own depth using 1k and moved GAUC by +0.00073, t = 0.33.

---

## Resource usage — `record-run-3`

```
iterations used            30  of the 50 allowed
stopped                    converged on the organisers' rule (eps=0.002, N=3)
                           not the iteration cap, not the wall-clock ceiling

agent wall-clock           6 h 29 m        see note
solution compute           5 h 56 m
GPU-hours                  0               the submitted run was CPU-only

LLM calls                  136
tokens in                  2,293,488       of which 1,676,544 served from cache
tokens out                 169,094
tokens total               2,462,582
cost                       $3.65

human interventions        0
solution crashes           0
self-directed web searches 10
```

**On the 6 h 29 m.** The organisers' 6-hour ceiling was published on 2026-08-27,
after this run started, and the run predates the wall-clock guard that now
enforces it. It is disclosed rather than omitted — but it did not affect the
submitted artifact. The validation-best checkpoint, iteration 27, was reached at
**5 h 28 m elapsed**, and it was still the best at the 6-hour mark:

```
it27   5.46 h   0.605493   <- the submitted candidate
it28   5.71 h   0.605408
it29   5.98 h   0.605380
       ---- 6 h ceiling ----
it30   6.23 h   0.605438   worse than it27
```

Truncating the run at exactly 6 hours yields an identical submission. Every run
since respects the ceiling; runs 8, 9 and 10 finished in 1 h 38 m, 1 h 48 m and
1 h 58 m.

**On the zero interventions.** The process was launched with one command and not
touched until it stopped itself. That is the figure the autonomy criterion asks
for. Project-wide development figures across all 15 runs are reported separately
in `docs/resource-usage.md`.

---

## Reproducing

```
# 1. confirm the harness reproduces the published baseline
PYTHONPATH=harness;kuairand-starter-kit \
  python kuairand-starter-kit/baseline.py --model fm --seed 0

# 2. rebuild the submitted predictions from the winning solution
python logs/record-run-3/solutions/027_deepfm_member.py \
  --data_dir rec_datasets/KuaiRand-Pure/data --split test \
  --seed 0 --out preds.npy

# 3. validate the submission file
python kuairand-starter-kit/submit.py --check submission.csv
```

Full per-iteration logs — hypothesis, code diff, GAUC and nDCG@5 separately,
errors and recovery events — are in `logs/record-run-3/`.
