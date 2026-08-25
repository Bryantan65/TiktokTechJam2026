# Results ledger

Curated reference numbers for TechJam 2026 Track 2. One row per result that
something else gets compared against.

**Scope.** This file holds reference points only — baselines and milestones.
Per-iteration agent runs are written by the harness to `logs/iterations/*.json`
and are not duplicated here; see [Provenance](#provenance) for why the two are
kept apart.

**Provenance.** Every entry records who produced it. Manual interventions are
how the competition scores autonomy, so `by` is a scored field, not
bookkeeping. Values:

| `by` | meaning |
| --- | --- |
| `human` | a person ran it or wrote the code that produced it |
| `agent` | produced by an autonomous iteration, no human input |
| `agent+N` | agent produced it, but needed N human interventions |

---

## Environment

Fixed for all results below unless a row says otherwise.

| | |
| --- | --- |
| GPU | NVIDIA RTX 3060 Ti, 8GB |
| Python | 3.12.10 |
| torch | 2.6.0+cu124 |
| numpy / pandas / scikit-learn | 2.5.2 / 3.0.5 / 1.9.0 |
| Dataset | KuaiRand-Pure, 1,436,609 raw rows |
| Preprocessed | 1,387,748 rows (3.4% dropped by CWM's `5 <= duration <= 400` filter) |

---

## R1 — CWM reproduction, paper's own metrics

Validates that the rebuilt environment is faithful. **Not** a competition
baseline: it measures `long_view2`, not `is_click`.

| | |
| --- | --- |
| Date | 2026-08-25 23:20 +0800 |
| By | `human` |
| Commit | `a8ca5b3` |
| Config | FM backbone, CWM loss, `sigma=2 c_inv=40 randseed=61` (`src/run.sh` defaults) |
| Split | CWM's own: train 04-08..04-21, val 04-22..04-28, test 04-29..05-08 |
| Label | `long_view2` |
| Artifact | `rec_datasets/WM_KuaiRand/FM_CWM_test_40_2_61_result.csv` |

| Metric | Published (Table 5) | Ours | Δ |
| --- | --- | --- | --- |
| AUC | 0.735 | 0.7357 | +0.0007 |
| nDCG@3 | 0.486 | 0.4848 | −0.0012 |

**Verdict: PASS** (tolerance ±0.005, set before the run).

Cost: 13 epochs to early-stop, ~28s/epoch, ~6 min total.

> ### ⚠️ R1 was measured under CWM's original split, which includes the test half
>
> This run used `--split_mode cwm`, the paper's own date cuts. Those predate the
> competition and do not match its splits:
>
> | | CWM's split | Overlap with competition splits |
> | --- | --- | --- |
> | early stopping | 04-22 .. 04-28 | **84.8%** of the competition *validation* half |
> | end-of-run metrics | 04-29 .. 05-08 | **100%** of the competition *test* half |
>
> So the AUC and nDCG@3 above were computed on data that includes the entire
> held-out test half. **Do not quote R1 as a validation-only result.**
>
> Nothing is contaminated: the checkpoint was already fixed by early stopping
> before that evaluation ran, so no information flowed back into the model, and
> R1 never enters the competition score — it exists only to prove the rebuilt
> environment reproduces the paper. R2 below, the actual baseline, is val-only
> and clean.
>
> Fixed in `--split_mode competition` (now the default), which early-stops on a
> time-ordered tail of the training period and never materialises the test half.
> See R4.

> The console prints this AUC under a field labelled `GAUC`; the field labelled
> `AUC` is a hardcoded zero. See `train_model2.py:255`.

---

## R2 — CWM baseline, competition metrics ← **the baseline to beat**

The organisers have not published baseline scores in the competition metrics,
and the paper's numbers measure a different label. So this is computed, not
looked up. Replace it if the Starter Kit publishes an official figure.

| | |
| --- | --- |
| Date | 2026-08-25 23:24 +0800 |
| By | `human` |
| Commit | `a8ca5b3` |
| Model | same checkpoint as R1 (`FM_CWM_test_40_2_61_model.pt`) |
| Split | **validation** = first 50% by `time_ms` of 04-22..05-08 |
| Label | `is_click` |
| Artifact | `rec_datasets/WM_KuaiRand/baseline_val_comp.json` |

| Metric | Value |
| --- | --- |
| NDCG@10 | 0.8412 |
| Recall@50 | 0.9999 |

> ### ⚠️ Superseded by R4 — optimistically biased
>
> The *scoring* here is clean: validation half only, test never touched. But the
> *model* being scored is R1's checkpoint, whose stopping epoch was chosen using
> 84.8% of this same validation half. Model selection saw the scoring data.
>
> The inflation is probably small — early stopping picks one scalar, not weights
> — but it is real and unquantified. **R4 retrains under the corrected split and
> supersedes this number as the baseline.**

Eval config: `candidate_set=impressions`, `zero_positive=skip`,
`split_point=0.5`. 23,194 users, 142,736 rows, median 4 candidates/user,
4,166 users (18%) with no clicks. Scoring cost ~10s.

---

## R3 — Reference rankers

Context for R2. Cheap rankers scored identically, to show what the metric's
floor and headroom actually are.

| | |
| --- | --- |
| Date | 2026-08-25 23:28 +0800 |
| By | `human` |
| Commit | `a8ca5b3` |
| Split / label | validation / `is_click`, same config as R2 |

| Ranker | NDCG@10 | Recall@50 | AUC |
| --- | --- | --- | --- |
| Random | 0.7749 | 0.9998 | 0.4995 |
| Original log order | 0.7835 | 0.9998 | — |
| Item CTR from train | 0.8269 | 0.9999 | 0.6822 |
| **CWM (R2)** | **0.8412** | **0.9999** | **0.7570** |

AUC is a diagnostic only — it is explicitly not scored (Appendix A.4). It is
kept because it is far more sensitive than NDCG@10 and so makes a better
inner-loop signal for the agent.

Three conclusions:

1. **Recall@50 is degenerate** under `candidate_set=impressions`. Every ranker
   scores 0.9998–0.9999, because 99.8% of users have fewer than 50
   impressions. Half the competition score is currently a constant.
2. **NDCG@10's floor is ~0.775, not 0.** CWM sits ~28% up the usable range.
3. **A one-line popularity heuristic reaches 78% of CWM's lift** over random.
   CWM's counterfactual watch-time machinery is worth +0.014 NDCG@10 over it —
   evidence that the baseline is optimising the wrong objective for this
   metric, and the most likely source of agent improvement.

---

## R4 — CWM baseline, corrected split ← **the baseline to beat**

Supersedes R2. Same model and config; the only change is that early stopping no
longer sees the competition validation half, and the test half is never loaded.

| | |
| --- | --- |
| Date | 2026-08-26 00:20 +0800 |
| By | `human` |
| Commit | `a8ca5b3` + split fix |
| Config | FM / CWM loss / `sigma=2 c_inv=40 randseed=61`, `--split_mode competition --es_frac 0.1` |
| Split | train 04-09..04-17 (992,048) · early-stop 04-17..04-21 (110,228) · report on competition val 04-22..04-30 (142,736) |
| Artifact | `rec_datasets/WM_KuaiRand/baseline_val_compsplit.json` |

| Metric | R4 (clean) | R2 (leaky) | Δ |
| --- | --- | --- | --- |
| **NDCG@10** | **0.8400** | 0.8412 | −0.0012 |
| **Recall@50** | **0.9999** | 0.9999 | ~0 |

The leakage was worth 0.0012 — small, as expected for a one-scalar selection
effect. **But seed variance is still unmeasured, so we cannot say whether 0.0012
is a real effect or noise.** Until a seed sweep exists, treat any delta below
roughly 0.005 as unproven.

Cost: 19 epochs to early-stop, ~27s/epoch (~8.5 min); scoring 6.05s.

---

## Open — affects every number above

Pending organiser confirmation (28 Aug webinar). All are config switches in
`src/comp/evaluate_comp.py`, so answers are a config edit, not a rewrite.

| Unknown | Current assumption | Impact if wrong |
| --- | --- | --- |
| Candidate set | `impressions` | **High.** `full_catalog` changes the achievable delta entirely and needs negative sampling |
| Zero-positive users | `skip` | Moderate — shifts baseline and agent equally, so delta mostly survives |
| Split point | 0.5 of rows by `time_ms` | Low — alternative reading is calendar midpoint |
| Exact click definition | raw `is_click` | Low — but doc defers "exact label definition" to the Starter Kit |
| Official baseline scores | R2 stands in | Replace R2 when published |
| Duration filter | CWM's `5..400` (drops 3.4%) | Unknown whether the official eval filters |

---

## Manual interventions

Counted from the first autonomous agent run onward. Setup work above is not an
intervention — the agent loop did not exist yet.

| Run | Interventions | Notes |
| --- | --- | --- |
| _(none yet)_ | — | agent loop not built |
