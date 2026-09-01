# Results summary

Deliverable 4. Final model output, the results table, and the resource usage
required to reach the converged result.

Every number here is read back from `logs/record-run-9/NNNN.json`,
`logs/record-run-9/events.jsonl` and `test_scores.json` — not from memory.

---

## Final model output

```
submission candidate   logs/record-run-9/solutions/031_global_ctr_tiebreak.py
predictions            submission.csv   (170,588 rows, test split)
schema                 row_id,user_id,video_id,score   per the Starter Kit
validated              python kuairand-starter-kit/submit.py --check
```

The run stopped on the organisers' convergence rule — not on the 50-iteration
cap and not on the wall-clock ceiling. The scored checkpoint is the
validation-best at that point, which is iteration 31.

---

## KuaiRand-Pure — the required benchmark

Scoring formula: `delta(m) = score_agent(m) − score_baseline(m)` for each
metric, then equal-weighted. Baseline is the official FM from
`kuairand-starter-kit/baseline_scores.json`.

### Validation-best

| metric | ours | official FM baseline | delta |
| --- | --- | --- | --- |
| GAUC | 0.673080 | 0.6674 | **+0.005680** |
| nDCG@5 | 0.538397 | 0.5357 | **+0.002697** |
| primary — mean of the two | 0.605738 | 0.6016 | +0.004238 |
| **equal-weighted delta** | | | **+0.004189** |

### Hidden-test equivalent, scored once locally

The test labels ship with the dataset, so this is measured by discipline rather
than withheld by the harness. `harness/score_test.py`, official `evaluate.py`.

| metric | ours | official FM baseline | delta |
| --- | --- | --- | --- |
| GAUC | 0.666150 | 0.6610 | **+0.005150** |
| nDCG@5 | 0.531544 | 0.5282 | **+0.003344** |
| primary — mean of the two | 0.598847 | 0.5946 | +0.004247 |
| **equal-weighted delta** | | | **+0.004247** |

The validation gain transferred: +0.004189 on valid, +0.004247 on test.

Seed spread on this data is 0.0008 (organiser-measured, 5 seeds), and the
target margin was 0.002. The submitted result clears the target on both splits
and on both metrics independently.

### How close to the ceiling is this?

`+0.0042` is uninterpretable without knowing what was available to win.

```
oracle ceiling (valid), requires the labels                   0.8484
perfect knowledge of every video, nothing about the user      0.6197
ours                                                          0.6057
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

`submission_1k.csv.gz` **is** in the repository. The plain CSV is 123 MB,
past GitHub's 100 MB per-file hard block, so the gzip is tracked instead at
44.6 MB and the CSV itself stays ignored. It restores byte-identical:

```
gzip -dc submission_1k.csv.gz > submission_1k.csv
```

Reducing score precision was the obvious alternative and does not work: four
decimals is still 39 MB and already moves nDCG@5 in the fifth decimal
(0.654932 against 0.654918). Only two decimals fits a 35 MB limit, and that
costs 0.00027 by introducing ties. Changing a submission to satisfy an upload
limit is the wrong trade.

To rebuild it from scratch instead, follow the reproduction steps below - the
winner needs its member caches present first, or it silently produces a
different model.

**The 1k delta is nine times the Pure delta** (+0.0394 against +0.0042), which is
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

## Resource usage — `record-run-9`

```
iterations used            32 scored of the 50 allowed  (33 records)
stopped                    converged on the rule, not the iteration cap and
                           not the wall-clock ceiling

agent wall-clock           2 h 34 m
solution compute           3 h 01 m     (exceeds wall clock: seeds run concurrently)
GPU-hours                  0            CPU-only

LLM calls                  93
tokens in                  2,789,194
tokens out                 182,752
tokens total               2,971,946
cost                       $4.52

human interventions        0
```

**Declared before the run, per FAQ 2.9.1.** The `run_start` event records
`min_scored_before_convergence: 30`, `max_experiments: 50` and
`max_wall_seconds: 21600` — the convergence floor fixed in advance and written
to the run log, and both hard caps configured correctly. The organisers permit a
team to declare its own epsilon, N and floor on exactly that condition.

**Why this run rather than `record-run-3`.** Run 3 scored 0.605493 on validation
against run 9's 0.605738 — a gap of 0.000245, well inside the 0.0008 seed
spread, so score is not the reason. Two other things are:

  - Its `run_start` records `max_experiments: 80` and no wall-clock ceiling, and
    it ran 6 h 29 m. That is outside the hard caps FAQ 2.9.1 restates.
  - Its `aligned_raw_features()` uses the row label as part of a lookup key for
    every split including test, with no `if sp == 'train'` guard. Run 9's
    lineage guards the equivalent state (`row_feats(row, update_label=(sp=='train'))`
    and `if sp == 'train'` around the label counters), so it never reads a test
    label. FAQ 2.9.3 prohibits using test labels in any way.

Run 3 remains in the repository as part of the run record.

---

## Reproducing

```
# 1. confirm the harness reproduces the published baseline
python harness/run_kit_baseline.py   --data_dir rec_datasets/KuaiRand-Pure/data --model fm --seed 0

# 2. rebuild the submitted predictions from the winning solution
PYTHONPATH=harness python logs/record-run-9/solutions/031_global_ctr_tiebreak.py   --data_dir rec_datasets/KuaiRand-Pure/data --split test   --seed 0 --out preds.npy

# 3. validate the submission file
python kuairand-starter-kit/submit.py --check submission.csv   --data_dir rec_datasets/KuaiRand-Pure/data
```

`submit.py` and `baseline.py` both live inside `kuairand-starter-kit/`, so
running them directly puts that directory on `sys.path` ahead of everything else
and the kit's Pure-hardcoded loader wins. That is fine for Pure and fails for any
other variant; `harness/run_kit_baseline.py` exists for exactly that reason.

Full per-iteration logs — hypothesis, code diff, GAUC and nDCG@5 separately,
errors and recovery events — are in `logs/record-run-9/`.
