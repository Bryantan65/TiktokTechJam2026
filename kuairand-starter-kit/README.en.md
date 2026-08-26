# KuaiRand-Pure Starter Kit

*English translation of `README.md`. The Chinese original is authoritative; if
the two ever disagree, believe the original.*

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, no pandas, no sklearn.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration):

```bash
# run inside the Starter Kit directory; extracts to ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; pass it explicitly if your data
lives elsewhere.

`--model` accepts `fm` (the official baseline), `pop` (trivial baseline), or
`random` (lower bound, for sanity-checking your evaluation code). FM takes about
**40 seconds** end to end, on a single CPU core.

## Task definition (fixed — do not change)

| | |
|---|---|
| Task | **Within-user ranking** — each user's impressions in the evaluation set are ranked among themselves. No full-catalogue retrieval. |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = the mean of the two** |
| Splits | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with no positives | nDCG counts as 0.0 and is included in the mean. GAUC counts only users with `0 < positives < impressions`, weighted by positive count. |
| nDCG gain | `2^rel − 1` (equivalent to identity for binary labels) |

The implementation is `evaluate.py`; every convention is stated in its header
comment.

## Baseline ladder

Scores on the **test** set. **The FM row is what you have to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, self-check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The metric's real range: nDCG@5 tops out at 0.729, not 1.0

Of the 23,875 users in the test set:

| | share | effect on the metric |
|---|---|---|
| All-negative users (none of their impressions are `long_view`) | **27.1%** | nDCG is permanently **0** — no model can fix this. Excluded from GAUC. |
| All-positive users | **9.2%** | nDCG is permanently **1**. Excluded from GAUC. |
| Discriminative users | **63.7%** | the actual sample GAUC is computed over |

So even using the true labels as your prediction (an oracle, i.e. perfect
ranking) gets you only:

| | random | FM baseline | **oracle ceiling** | share FM has taken |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure your progress against the oracle, not against 1.0.** Seeing 0.5946 and
concluding "there's a long way to a perfect score" is a misreading — the
baseline has already taken about a third of the usable range, and the remaining
headroom is 0.27, not 0.41.

FM's standard deviation across 5 random seeds is **0.0008** in every metric. The
convergence rule follows from that: **ε = 0.002 (≈2.5σ), N = 3** — a run is
converged once three consecutive iterations fail to improve the validation
primary score by more than 0.002.

> Self-check: if your evaluation code doesn't produce primary ≈ 0.475 (±0.001)
> for `--model random`, your harness is broken. Fix that first.

## Submission format

CSV with a header, one row per row of the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| field | meaning |
|---|---|
| `row_id` | consecutive from 0, matching the row order of `data.load()[split]`. Deterministic: `log_standard_4_08_to_4_21_pure.csv` is read first, then `log_standard_4_22_to_5_08_pure.csv`, and original file order is preserved after filtering by date. |
| `user_id` / `video_id` | redundant, used only to verify alignment |
| `score` | your model's score for that row. Any real number — only relative order matters. NaN and Inf are rejected. |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the
> evaluation set — 3.06% of pairs in the test set are duplicated, up to 12 times.
> So it cannot serve as a primary key.

Generating and validating:

```bash
python3 submit.py --make  --split test  submission.csv   # generate a sample submission from the official FM baseline
python3 submit.py --check --split test  submission.csv   # validate format and alignment
python3 submit.py --score --split valid submission.csv   # validate and score (works locally on valid)
```

`--check` rejects: a wrong header, the wrong number of rows, gaps in `row_id`,
`user_id`/`video_id` that don't align with the evaluation set, and scores that
are non-numeric, NaN or Inf. **Run `--check` yourself before submitting.**

## Where to start changing things

The ordering below is **measured, not guessed**. Dead ends the organisers have
already tried are marked so you don't repeat them.

### Already measured: these two yield nothing. Don't waste iterations.

| what was tried | result |
|---|---|
| **More static features** — wiring in all 13 of CWM's feature fields (adding `music_id` / `video_type` / `upload_type` plus 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** for the 5 fields — indistinguishable within noise, slightly worse if anything |
| **More model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887 — essentially flat |

The reason: the `user_id × video_id` cross already captures most of the learnable
signal. Coarse buckets like `follow_user_num_range` are redundant once you have
`user_id`, and 1.14M rows won't support more capacity anyway. **The bottleneck is
neither features nor capacity.**

⚠️ Also note: **first-order terms on pure user-side features contribute exactly
zero.** Because ranking happens within a user, any term that is constant inside a
user's group cannot change the order within that group. (Measured: `item_pop ×
user bias` and plain `item_pop` score identically to the last digit.) User-side
features can only matter through **crosses with item-side features.**

### Unexplored: the headroom should be here

Ordered by our guess at likely payoff. **The organisers have not tested these —
they're left for you.**

1. **Change the loss function.** It's currently pointwise logloss, but the
   metrics (GAUC / nDCG) are **ranking** metrics. Switching to pairwise (BPR) or
   listwise (softmax over a user's impressions) aligns the objective with how
   you're scored. We think this is the most likely to work.
2. **User behaviour sequences.** The current features use **no sequence
   information at all**. Each user has hundreds to thousands of interactions in
   train; DIN / SIM-style interest modelling is a completely open direction.
3. **Multi-task learning.** The logs also contain `is_click`, `is_like`,
   `is_follow`, `is_comment`, `is_forward` and `play_time_ms`, which can serve as
   auxiliary tasks alongside the main `long_view` objective.
4. **Modelling watch time.** This is exactly [CWM](https://github.com/hyz20/CWM)'s
   contribution: it treats watch time as **censored regression** (true watch time
   is truncated when the video ends, so it uses a one-sided loss rather than
   squared error). A direction with genuine research depth.
5. **Different models.** DeepFM / DCN / xDeepFM. Given that capacity measurably
   isn't the bottleneck, **rank this below 1–4.**
6. **Time features and distribution drift.** `hourmin`, `date`, and the drift
   between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a
   randomised-exposure log (1.18M rows) and can serve as an extra unbiased
   validation set, to check whether your model only overfits biased traffic.

## Using your own model (including CWM)

`evaluate.py` is fully decoupled from any model. It needs three equal-length
arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids` — the user_id of each row in the evaluation set
- `labels` — that row's `long_view` (0/1)
- `scores` — your model's score for that row (any real number; only relative order matters)

So you can skip `baseline.py` entirely and use PyTorch, LightGBM, or
[CWM](https://github.com/hyz20/CWM)'s xDeepFM — just hand `scores` to
`evaluate()` at the end. **`evaluate.py` is the sole authority on scoring.**

> A caution on CWM: it depends on `torch==1.6.0` (a 2020 release, which probably
> won't install on a modern GPU), its loss optimises counterfactual watch time,
> and its evaluation label is a `long_view2` it reconstructs itself. It's the
> research code for a duration-debiasing paper — useful as an **advanced
> reference**, not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | metric implementation and every scoring convention. **Do not modify.** |
| `data.py` | data loading, official splits, feature encoding. Add features here. |
| `baseline.py` | the three baselines. FM is the one to beat. |
| `baseline_scores.json` | official scores, seed variance, convergence parameters. |
| `submit.py` | generate / validate a submission file. |
| `ablation_features.py` | the feature ablation, reproducing the "more features don't help" numbers. |
