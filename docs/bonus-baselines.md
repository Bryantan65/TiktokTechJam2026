# Baselines for the bonus benchmarks

`harness/ledger.py` refuses to grade a dataset whose baseline has not been
measured, and points here. This is that measurement.

## Why this file has to exist

The organisers published baseline scores for **KuaiRand-Pure only**
(`kuairand-starter-kit/baseline_scores.json` — `"dataset": "KuaiRand-Pure"`,
four models, no 1k or 27k entry anywhere). But the scoring formula needs one per
dataset:

> Within each dataset, the score is the equal-weighted average of each metric's
> absolute improvement over the official baseline on the hidden test set.
> ```
> delta(m) = score_agent(m) − score_baseline(m)
> ```

So a bonus submission has no reference number unless we produce one.

**What we did.** Ran the kit's own `baseline.py` — the organiser-provided
reference pipeline, numpy FM, `k=16 lr=0.001`, the same five categorical fields —
unmodified, on the 1k data. The problem statement defines the baseline as *a
pipeline*, not a number, so running their pipeline on another variant is not
"a baseline the team builds itself"; it is their baseline, measured where they
did not publish one.

**The one caveat, stated plainly.** `baseline.py` and the kit's `data.py`
hardcode Pure's filenames, so this needed `harness/data.py`, our variant-aware
loader. That loader changes which files are opened and nothing else — for Pure it
delegates to the kit's own `load()` verbatim, and all three Pure splits come back
row-identical. No hyperparameter, no field, no split date was altered.

## KuaiRand-1k

Kit FM, seed 0, validation split (`20220422–20220428`), 2,524,980 rows.

```
epoch 1   GAUC 0.6698   nDCG@5 0.6018   primary 0.6358
epoch 2   GAUC 0.6749   nDCG@5 0.6153   primary 0.6451   <- best
epoch 3   GAUC 0.6738   nDCG@5 0.5942   primary 0.6340
epoch 4   GAUC 0.6716   nDCG@5 0.5751   primary 0.6234
```

Early stop fired at epoch 6, best kept from epoch 2.

```
KuaiRand-1k, kit FM, seed 0
  valid   GAUC 0.6749   nDCG@5 0.6153   primary 0.6451
  test    GAUC 0.6730   nDCG@5 0.6049   primary 0.6390    <- the scored reference
```

**The test row is the one the scoring formula needs** - deltas are computed on
the hidden test set, not on validation. Alongside Pure's published numbers:

```
                    GAUC      nDCG@5    primary
Pure  test         0.6610     0.5282    0.5946   (organiser-published)
1k    test         0.6730     0.6049    0.6390   (measured here)

Pure  valid        0.6674     0.5357    0.6016   (organiser-published)
1k    valid        0.6749     0.6153    0.6451   (measured here)
```

A 1k agent result is scored as `mean(GAUC - 0.6730, nDCG@5 - 0.6049)`.

It peaks at epoch 2 and overfits from there, much earlier than Pure (epoch 7–11).
Expected: 1k has 4.4x Pure's training rows concentrated in 1/27th of the users, so
each user's parameters see far more gradient per epoch.

### Why the constant could not stay a constant

```
                  GAUC      nDCG@5    primary
Pure baseline    0.6674     0.5357    0.6016
1k baseline      0.6749     0.6153    0.6451     +0.0435
```

GAUC is nearly the same; **nDCG@5 is +0.08 higher**. 1k users have many more
impressions and a different positive rate, so the per-user ranking metric sits
somewhere else entirely.

A 1k run graded against Pure's 0.6015 computes `delta = +0.043` on *every*
experiment and stamps them all `KEPT`. Convergence still works — it compares
against the running best — but the verdict column becomes noise, and the agent is
told it beat the target on iteration 1. That failure is silent: nothing in the
logs looks wrong.

## KuaiRand-27k

Measured, with a caveat that has to travel with the number.

```
GAUC     0.688589
nDCG@5   0.641569
primary  0.665079      validation, seed 0, early stop at epoch 7 (peak epoch 3)
```

**Measured with the PyTorch port, not the kit's `baseline.py`.** That is a
weaker provenance claim than 1k's and the difference is not neutral. The port
reproduces the kit to 0.0001 on Pure (0.6015 vs 0.6016) but lands **0.0017
below** it on 1k (0.643428 vs 0.6451). A baseline biased low inflates every
delta measured against it, so this gap favours us. Any 27k result reported
against this number must say so.

The kit's own numpy script would settle it, at roughly 28 h of CPU: ~10 min per
epoch on 1k's 5M training rows, against 27k's ~139M. Worth doing if a 27k result
is ever submitted.

**Two things had to change before it would run at all.** 322M rows as Python
tuples need ~110 GB, against a container limit of 116 GB — the first attempt
reached 85 GB, kept climbing, and was killed with no traceback, because an OOM
kill leaves none. Note `free` reports the *host's* 503 GB rather than the
container's limit, so the failure looks impossible right up until it happens.

`data.load(only=['train','valid'])` now skips rows outside the requested splits
as the CSVs are read instead of building and discarding them. That is 208M rows,
~71 GB, and it fits. A baseline needs no test split: Deliverable 4 asks for the
validation-best score, and the hidden test is the organisers' to score.

```
loaded in 26 min   train 136,296,576   valid 71,149,570
                   322,278,385 rows seen, 207,446,146 kept
train 124 min      12.4 min per epoch on an RTX 4090
total 150 min
```

**On whether a 27k agent run is viable.** One full-data experiment costs about
two hours — 26 min to load, ~12 to encode, ~87 to train — against a 6 h ceiling,
so a run fits three experiments. Every good result we have arrived at experiment
23 (Pure) or 40 (1k). With the parsed-data cache, a training subsample and fewer
epochs it comes down to roughly 18, which is still short of that. The earlier
reasoning for setting 27k aside also still stands: it is wider than 1k rather
than deeper (11,713 vs 11,812 interactions per user), and the one mechanism it
could uniquely supply was tested at 27k's own depth using 1k and moved GAUC by
+0.00073, t = 0.33.

Reproduce with:

```
python harness/measure_baseline.py   --data_dir rec_datasets/KuaiRand-27K/data --device cuda --seed 0
```

## Reproducing

```
python harness/run_kit_baseline.py \
  --data_dir rec_datasets/KuaiRand-1K/data --model fm --seed 0
```

**Why not just set PYTHONPATH.** The obvious command -
`PYTHONPATH=harness;kuairand-starter-kit python kuairand-starter-kit/baseline.py`
- cannot work on a non-Pure variant, and this file used to recommend it.
Python puts the *script's own directory* at `sys.path[0]`, ahead of everything
on PYTHONPATH, so running a script that lives inside `kuairand-starter-kit/`
guarantees the kit's Pure-hardcoded `data.py` wins. `run_kit_baseline.py`
imports `harness/data.py` as `data` first, so the kit's `from data import
load` finds it already in `sys.modules` and never searches the path. The kit
itself is not modified.

Roughly 10 minutes per epoch on this machine (5,055,984 training rows, CPU).
