# CLAUDE.md

TechJam 2026 Track 2: build an agent that autonomously improves an ML pipeline.

**`kuairand-starter-kit/` is the authority on the task.** Read
`kuairand-starter-kit/README.en.md` (English translation of the Chinese
original) and the header of `kuairand-starter-kit/evaluate.py` before changing
anything that touches scoring. What follows is the decision-relevant subset,
not a substitute.

## The task — fixed, do not change

| | |
| --- | --- |
| Task | within-user ranking over logged impressions. **No full-catalogue retrieval.** |
| Label | `long_view` — the raw 0/1 column |
| Metrics | GAUC and nDCG@5; **primary = mean of the two** |
| Splits | train `20220408-20220421` · valid `20220422-20220428` · test `20220429-20220508` |
| Zero-positive users | nDCG counts as 0.0 **and is included** in the mean. GAUC counts only users with `0 < positives < impressions`, weighted by positive count. |
| Convergence | ε = 0.002, N = 3 consecutive iterations |

`kuairand-starter-kit/evaluate.py` is the sole authority on scoring. Do not
modify it.

## The number to beat

**The agent only ever sees `valid`.** Test is refused by the harness and is
scored once, at the end. Compare against the **valid** column — the README's
ladder table shows test only, and mixing the two makes a +0.0018 gain look like
+0.009.

Primary = mean(GAUC, nDCG@5). From `kuairand-starter-kit/baseline_scores.json`:

| | valid primary | test primary |
| --- | --- | --- |
| random (self-check) | 0.4834 | 0.4753 |
| item popularity | 0.5807 | 0.5715 |
| **FM (official baseline)** | **0.6016** | **0.5946** |
| oracle ceiling | 0.8484 | 0.8645 |

**Agent target: beat `valid` 0.6015 by ≥ 0.002, i.e. reach 0.6035.**

FM's valid detail: GAUC 0.6674, nDCG@5 0.5357. Note the oracle ceiling is
*lower* on valid than test, so valid is not simply the easier split — FM is at
70.9% of ceiling on valid vs 68.8% on test.

Baseline config: FM, `k=16 lr=0.001 batch=8192 max_epochs=40 patience=4`,
fields `[user_id, video_id, author_id, tab, dur_bucket]`. Runs in **~40 s on one
CPU core** — iterations are cheap, so favour many small experiments.

Seed std is **0.0008** on every metric (5 seeds, organiser-measured).

**Verified on this machine (2026-08-27, seed 0):** random self-check test primary
**0.4757** vs 0.4753; FM test primary **0.5953** vs 0.5946, valid **0.6015** vs
0.6016. All within seed noise, so our numbers are comparable to the official
ones. Run takes 50 s; valid peaks at **epoch 7** and overfits after.

## Facts that will mislead you

- **nDCG@5's ceiling is 0.729, not 1.0.** 27.1% of test users have no positives
  at all (nDCG permanently 0); 9.2% are all-positive (permanently 1). Only 63.7%
  are discriminative. Measure progress against the **oracle primary 0.8645** —
  the baseline has already taken ~31% of the usable range, leaving 0.27 of
  headroom, not 0.41.
- **Pure user-side features contribute exactly zero.** Ranking is within-user, so
  any term constant inside a user's group cannot change that group's order.
  User features only matter through **crosses with item-side features**.
  (Measured: `item_pop × user_bias` scores identically to plain `item_pop`.)
- **Self-check:** `--model random` must give primary ≈ 0.475 (±0.001). If not,
  the harness is broken — fix that before anything else.
- **`(user_id, video_id)` is not unique** — 3.06% of test pairs are duplicated,
  up to 12×. Submissions key on `row_id`, not the pair.

## Already measured — do not spend iterations here

The organisers tested these and published the negative results:

| tried | result |
| --- | --- |
| More static features (all 13 CWM fields) | primary 0.5940 vs 0.5950 for 5 fields — no gain, slightly worse |
| More capacity (embedding k = 8 / 16 / 32) | 0.5895 / 0.5902 / 0.5887 — flat |

`user_id × video_id` already captures most of the learnable signal, and 1.14M
rows won't support more capacity. **The bottleneck is neither features nor
capacity** — which is exactly where an agent's instincts point first.

## Unexplored — the organisers' own ranking of where headroom is

1. **Change the loss.** Currently pointwise logloss, but GAUC/nDCG are *ranking*
   metrics. Pairwise (BPR) or listwise (softmax over a user's impressions)
   aligns objective with metric. Their pick for most likely to work.
2. **User behaviour sequences.** Current features use none. DIN/SIM-style
   interest modelling is completely open.
3. **Multi-task.** `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` as auxiliary tasks.
4. **Watch-time modelling** — censored regression, CWM's actual contribution.
5. **Other models** (DeepFM/DCN/xDeepFM) — ranked *below* 1-4, since capacity
   measurably isn't the bottleneck.
6. **Time features and drift** — `hourmin`, `date`, train-vs-test drift.
7. **Unbiased validation** — `log_random_4_22_to_5_08_pure.csv` (1.18M rows,
   randomised exposure) as an extra validation set to detect overfitting to
   biased traffic.

## Data quirks

- Data starts **04-09**, not 04-08 — the filename is nominal.
- `long_view` matches its documented rule (`play_time >= duration` for videos
  ≤18s, else `play_time >= 18s`) with **97.8% agreement**.
- `is_click` is **not** a click for most rows — it is UI-dependent (`valid_play`,
  a 7-second watch threshold, in the single-column UI; a real click in the
  two-column UI, 96.5% agreement). It is *not* the scored label, but it is a
  legitimate auxiliary signal under direction 3 above.

## The CWM codebase in `src/`

Vendored from hyz20/CWM (arXiv 2406.07932); `upstream` points there. **The
organisers explicitly do not recommend it as a starting point** — it optimises
counterfactual watch time and evaluates on a `long_view2` it reconstructs
itself. Useful as an advanced reference for direction 4.

It has been ported to a modern stack and reproduces the paper (AUC 0.7357 vs
0.735, nDCG@3 0.4848 vs 0.486) — which the Starter Kit says most teams will not
manage, so direction 4 is comparatively cheap for us. Setup in `requirements.txt`; `torchfm` is a
required dependency the upstream README omits.

Its `--split_mode cwm` matches the **official** split. `--split_mode
competition` was built against the pre-release problem statement's "first 50% by
time" wording and is **superseded** — do not use it.

## Superseded — ignore anything that says otherwise

Work predating the Starter Kit assumed label `is_click`, metrics NDCG@10 and
Recall@50, and a 50/50-by-time split. All wrong, and the old `RESULTS.md` has
been removed. The agent's own `LEDGER.md` and `logs/iterations/` are the results
record from iteration 1 onward.

## Repo gotchas

- Never commit `rec_datasets/` (620MB) or `techjaminfo*.docx` (pre-release
  material). Both gitignored. `models/*.pt` ignored; the small score JSONs are
  tracked deliberately as evidence.
- `src/utils/evaluate.py:8` imports a `model.trans_model` that does not exist —
  commented out; the names are unused.
- Prefer symbol names over line numbers in docs. Line numbers go stale every
  time the agent edits a file.
