## Inspiration

Agents are clearly where things are going, and most of what you see them do is fetch, summarise, or write boilerplate. Track 2 asked for something harder: an agent doing an ML engineer's actual job — hypothesise, write the code, run it, read the result, decide what's next.

The starter kit also rules out the obvious answers up front. The organisers published their own negative results: more features does nothing, more capacity does nothing. Those are the first two things anyone reaches for.

## What it does

The agent runs the improvement loop unsupervised: reads the task, writes a complete Python solution to a file, runs it, gets a validation score, decides what to change. Experiments form a tree, and it picks its next parent by Upper Confidence bounds applied to Trees (UCT) — the selection rule from Monte Carlo Tree Search — so it can return to an earlier branch instead of only refining its latest attempt.

The task is fixed by the organisers: rank each user's own logged impressions, label `long_view`, score = mean(GAUC, nDCG@5). The agent only sees train and validation — the harness holds the test labels and does the scoring, so a solution can't grade itself or pick which split it's graded on. Runs stop on the organisers' convergence rule (ε = 0.002, N = 3), inside their 50-experiment and 6-hour limits.

One command, then leave it.

## Results

KuaiRand-Pure, the required benchmark. Delta is per metric, equal-weighted, per the judging formula.

| | GAUC | nDCG@5 | equal-weighted delta |
| --- | --- | --- | --- |
| **validation** | 0.672469 | 0.538518 | **+0.003944** |
| **test** | 0.665391 | 0.531626 | **+0.003908** |
| official FM baseline (valid / test) | 0.6674 / 0.6610 | 0.5357 / 0.5282 | — |

The gain transferred. Seed noise is 0.0008 and the target margin was 0.002, so it clears the bar on both splits and on each metric independently.

**Cost to get there:** 30 of 50 iterations, converged rather than truncated. 2.46M tokens, $3.65, 0 GPU-hours, **zero human interventions**.

## How close to the ceiling is that?

We measured it, because "+0.0039" means nothing without knowing what was available to win.

| | validation primary |
| --- | --- |
| oracle — but it needs the realised labels, so no model reaches it | 0.8484 |
| perfect knowledge of every video, nothing about the user | 0.6197 |
| **ours** | **0.6055** |
| official FM baseline | 0.6016 |

**Personalisation on Pure is close to unavailable.** The median user has 31 training rows across 29 creators, and only **3.38%** of validation rows involve a (user, creator) pair ever seen in training — against 33.70% on KuaiRand-1k. The lookup table is empty at prediction time.

We tested the obvious escape route: learn taste over *content* — topic, music, length — since content repeats where creators don't. It fixes coverage, 3.7% to **75.2%**, but not the score. A tuned two-tower content model, swept over 18 configurations, lands at **0.5985 ± 0.0006** — and adding `video_id` and `author_id` back makes it slightly *worse*. If per-item memorisation were doing real work, restoring identity would help.

Fifteen independent runs, different mechanisms and different underlying models, all landed within 0.0011 of each other. Full analysis — including the unused `video_features_statistic` file, which turns out redundant and confirmed not leaking — is in the README and `docs/results.md`.

## How we built it

`harness/` owns the data, labels, scoring and ledger. `agent/` owns the prompt, loop and tools. A solution is a standalone file taking `--split` and `--out` that writes predictions; it never sees a label, and `run_experiment` refuses any split but validation in both layers.

**Each turn** the agent gets the ledger, picks an experiment to build on, reads its source, writes a new file, and runs it — scored on three seeds, appended as a row. Five tools, nothing else: `read_ledger`, `read_solution`, `write_solution`, `run_experiment`, `web_search`.

**The ledger is its memory** — one line per experiment, read in full every turn:

```
| #  | parent | hypothesis                                        | valid  | +/-    | delta   | verdict |
| 1  | -      | control: official FM ported to PyTorch, BCE        | 0.6014 | 0.0002 | -0.0001 | noise   |
| 2  | 1      | replace pointwise BCE with same-user BPR pairs     | 0.6029 | 0.0004 | +0.0014 | noise   |
| 27 | 24     | add a small DeepFM member to the node-24 ensemble  | 0.6055 | 0.0002 | +0.0040 | KEPT    |
```

The `parent` column makes it a tree rather than a list. The `+/-` is seed spread, and the header warns the agent that two rows closer together than their spread have not been told apart.

**It invented a tool we didn't give it.** There is no edit tool — `write_solution` takes a whole file, and 93% of the time it writes one (median 11.7 KB). But it worked out that when a change is small it can instead write a short file that loads its parent by path and overrides only what differs. Those are 7% of its solutions at half the size, including the winner of one of our best runs. Nothing in the prompt suggests it.

## Challenges we ran into

**The convergence rule** — the longest debugging effort. Taken literally, "no improvement over the last 3 iterations" fires almost immediately, because three non-improvements in a row is just what the start of a search looks like. Every run would have stopped at iteration 4 or 5, and nine of twelve *below* the target score. Every run actually found its best between iteration 23 and 29.

**Our own feedback loop was hours, not minutes.** The agent's loop is fast; ours is a full run — two to six hours to learn whether a harness change helped, 47 hours over 18 runs. And one run doesn't answer it: run-to-run standard deviation is 0.00036, so any improvement under ~0.0007 is invisible in a single run. We batched changes and distrusted single-run conclusions, including the flattering ones.

**Failure had to be information, not a dead end.** Hitting LightGBM's hard 10,000-row-per-query-group limit, the agent read the error, split the oversized users into chunks under the cap, and carried on — on the very next iteration, in three separate runs.

**1k had no baseline at all.** The organisers published one for Pure only, but bonus scoring is `agent − baseline` per dataset. We measured one by running the kit's *own* `baseline.py` unmodified on 1k, which needed a variant-aware loader that changes only which filenames are opened — Pure still returns row-identical splits.

## Accomplishments that we're proud of

Beat the baseline and **converged rather than running out of budget**, at $3.65 and zero interventions — one command, untouched until it stopped itself, including overnight.

Cut runtime from 6h29m to under two hours without weakening the measurement.

And it found things we didn't hand it: seed-ensembling to cut variance, and abandoning LambdaRank for a pointwise objective after finding it unstable.

## What we learned

The bottleneck was never features or capacity. It was the objective — the baseline optimises pointwise log-loss while the metrics are ranking metrics.

**The harness matters more than the prompt.** Seed noise, a convergence rule that fires too early, a baseline attached to the wrong dataset: each quietly produces confident, wrong conclusions, and none of them look like bugs in the log.

And an agent that recovers well beats one that never fails. The best thing in our logs isn't a clean run — it's the agent reading a crash, naming the cause, and fixing it in one step.

## What's next for ZuMianBao

**Use the hardware.** We profiled a live run at roughly 20% utilisation — 54 of 64 cores idle — while the agent already asks for several experiments at once on some turns and the harness runs them one after another.

**Cache the parsed data.** Measured at 6–7× faster with byte-identical output; about a third off a 1k experiment.

**Sequence models on 1k.** We tested DIN-style interest modelling on Pure and it landed at 0.6047 — the same band as everything else, which is what 31 impressions per user predicts. 1k's median user has 3,489, so that is where it should actually have history to model.

## Built with

**Development tools:** VS Code, Claude Code, Git/GitHub, RunPod (RTX 4090) for GPU runs, Windows and Linux.

**APIs:** OpenAI-compatible chat completions — GPT-5.5 for the record runs, with DeepSeek-V4-Pro and GPT-5.6 evaluated on an identical harness. A web-search tool the agent invokes itself when it wants to look up a method.

**Libraries and frameworks:** PyTorch, torchfm, LightGBM, XGBoost, NumPy, pandas, scikit-learn, SciPy, tqdm, `openai`, python-dotenv.

**Datasets:** KuaiRand-Pure (1.4M interactions, required) and KuaiRand-1K (11.7M, bonus). KuaiRand-27K was downloaded and assessed but not run. No external or hand-labelled data; the official `evaluate.py` is used unmodified as the sole scoring authority.
