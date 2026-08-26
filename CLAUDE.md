# CLAUDE.md

TechJam 2026 Track 2. Build an agent that autonomously improves this pipeline.
Vendored from the CWM paper repo (hyz20/CWM, arXiv 2406.07932); `upstream` still
points there. See `RESULTS.md` for the results ledger.

## Run it

```bash
cd src
../.venv/Scripts/python.exe prepare_data.py --group_num 60 --windows_size 3 --eps 0.5 --dat_name KuaiRand --is_load 0
../.venv/Scripts/python.exe main.py --fout ../models/<name> --dat_name KuaiRand \
    --model_name FM --label_name CWM --sigma 2 --c_inv 40 --randseed 61 --load_to_eval 0
../.venv/Scripts/python.exe comp/evaluate_comp.py --fout ../models/<name> --split val
```

Setup: see `requirements.txt` (torch installs separately — the CUDA build
is not on PyPI). `.venv` = Python 3.12 / torch 2.6.0+cu124 / pandas 3.0.5.
`--load_to_eval 1` re-scores a saved checkpoint without retraining (seconds).
Training ~6 min on the RTX 3060 Ti; scoring ~10 s.

## Splits — never touch test

`train 04-09..04-21` · `val = first 50% by time_ms of 04-22..05-08` · `test = last 50%`.

`--split_mode competition` (default) early-stops on a time-ordered tail of the
training period and never loads the test half. `--split_mode cwm` restores the
paper's split — it reads the competition test half, so use it only for
reproduction checks.

`comp/evaluate_comp.py` refuses `--split test` without `--allow_test`.

## Scoring

Label `is_click`, NDCG@10 and Recall@50, score = mean of per-metric absolute
delta over the baseline. Unknowns are config switches in `comp/evaluate_comp.py`
(`candidate_set`, `zero_positive`, `split_point`) — do not hardcode them.

Baselines and the reference-ranker table live in `RESULTS.md`.

## Facts that will mislead you

- **NDCG@10's floor is ~0.775, not 0.** Random scores 0.7749 (median 4
  candidates/user, 46% positives). A one-line item-CTR heuristic gets 0.8269.
  Read all NDCG@10 numbers against that floor.
- **Recall@50 is degenerate** under `candidate_set=impressions` — every ranker
  scores ~0.9999, because 99.8% of users have fewer than 50 impressions. Half
  the score is currently a constant.
- **`zero_positive` swings NDCG@10 by 0.15** (`skip` 0.8412 vs `zero` 0.6901).
  Never compare numbers computed under different settings.
- **`is_click` is mostly NOT a click.** Per kuairand.com it is UI-dependent: a
  real click in the two-column UI, but `valid_play` in the single-column UI —
  `play_time >= duration` (≤7s) or `play_time > 7s`. Verified: 96.5% agreement
  with that rule. `tab 1` (73% of rows, CTR 0.53) is single-column; `tab 0` (11%,
  CTR 0.09) is two-column. So the label is a **mixture of two regimes** under one
  name, and `tab` is the feature that separates them.
- Consequently `is_click` and `long_view` are thresholds on the *same* quantity
  (7s vs 18s), which is why P(click|long_view)=0.995 — near-tautological, not
  causal. CWM predicting watch time is therefore well aligned with the metric,
  not misaligned. The lever is that CWM ranks by *continuous* watch time while
  the metric wants P(watch > 7s) — a calibration/threshold problem, not a
  wrong-objective problem.
- The console prints the real AUC under a field labelled `GAUC`; the field
  labelled `AUC` is a hardcoded zero (`train_model2.py:_test_and_save`, passes `0,0` before
  `gauc_val`). The `_result.csv` writes `gauc_val` into a column named `AUC`.
- `long_view2` is **derived**, not a dataset column: `cal_ground_truth.py:29`
  thresholds `play_time_truncate` at its own 70th percentile. The raw data ships
  a different column called `long_view`.
- Data starts **04-09**, not 04-08 — the filename is nominal.
- **Seed noise is σ = 0.0007** on NDCG@10 (4 seeds, R5 in `RESULTS.md`). A change
  must gain **≥ 0.0016** over the baseline mean to clear 2σ; use ≥ 0.002 when
  comparing two single runs. Baseline mean is **0.839988**, not any one run.
  Recall@50's σ is 0.000016 — it is a constant.

## Repo gotchas

- `pre_kuairand.py:99` hardcodes retained columns; `is_click` and `time_ms` were
  added by us. Features come from a fixed list in `summary_dat.py`, so extra
  columns never reach training.
- `evaluate.py:8` imports a `model.trans_model` that does not exist — commented
  out; the names are unused. Several other `.pyc` files reference absent sources.
- `torchfm` (pytorch-fm) is a required dependency the README omits.
- Never commit `rec_datasets/` (620MB) or `techjaminfo*.docx` (pre-release
  material under Early Bird access). Both are gitignored.

## Open — pending the 28 Aug webinar

Candidate set (impressions vs full catalogue — decides whether Recall@50 means
anything), zero-positive convention, official baseline scores, ε/N, compute
budget, submission schema, exact click definition.
