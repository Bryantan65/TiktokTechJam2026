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

**Validation baseline: GAUC 0.6749 / nDCG@5 0.6153 / primary 0.6451.**

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

Not measured. 322M interactions, 9.21 GB compressed, and the kit's loader builds
Python lists of tuples, so it would need a streaming rewrite before it could open
the file. See `HANDOFF.md` for why we stopped short: the mechanism 27k would
supply — deep per-user history — was tested at 27k's own depth using 1k
(11,713 vs 11,812 interactions per user) and moved GAUC by +0.00073, t = 0.33.

## Reproducing

```
PYTHONPATH=harness;kuairand-starter-kit \
  python kuairand-starter-kit/baseline.py \
    --data_dir rec_datasets/KuaiRand-1K/data --model fm --seed 0
```

Roughly 10 minutes per epoch on this machine (5,055,984 training rows, CPU).
