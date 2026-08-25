# TechJam 2026 — Track 2: Autonomous ML Research Agent

## The task
Build an LLM-driven agent that autonomously improves an ML pipeline:
reads results → proposes a change → applies it → trains → evaluates → repeats.
We build the agent. The agent improves the model. Not us.

## Benchmark
- Required: KuaiRand-Pure (data/KuaiRand-Pure/)
- Bonus: KuaiRand-1K, KuaiRand-27K
- Baseline: CWM (Counterfactual Watch Model, KDD'24, arXiv 2406.07932) in cwm/
- Metrics: NDCG@10, Recall@50. Label = is_click
- Score = mean absolute delta over baseline, on hidden test, scored once

## Splits (fixed by organizers)
- train: log_standard_4_08_to_4_21_pure.csv
- val:   log_standard_4_22_to_5_08_pure.csv, first 50% by time
- test:  log_standard_4_22_to_5_08_pure.csv, last 50% by time
Develop on train + val only. Never touch test.

## Judging
- 35% Technical Execution (delta over baseline + robustness/recovery)
- 20% Innovation (what the agent chose to try, and why)
- 20% Impact (autonomy — measured by count of manual interventions)
- 15% Feasibility (token spend + GPU-hours)
- 10% Presentation

## Architecture
1. cwm/ — the pipeline being optimised
2. harness/ — run_training(config) -> JSON
3. agent/ — the LLM loop
4. logs/ — per-iteration: hypothesis, diff, metrics, errors

## The harness contract (agreed, do not change without telling the team)
{"ndcg@10": 0.42, "recall@50": 0.63, "gpu_seconds": 180,
 "status": "ok", "error": null}

## Open questions for the 28 Aug webinar (2pm SGT)
- Which exact CWM config is the official baseline, and its published scores?
- What are ε and N for the convergence rule?
- Compute budget? (listed TBD)
- Submission output schema?
- What counts as a "manual intervention"?

## Current state
- [x] KuaiRand-Pure downloaded
- [ ] CWM vendored
- [ ] Baseline reproduced by hand
- [ ] Harness built
- [ ] Agent loop