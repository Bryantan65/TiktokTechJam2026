## Inspiration

We'd been building with AI mostly at the application level — wrapping LLMs around real problems, putting agents together that coordinate tools and automate workflows. That teaches you how to *orchestrate* AI, but at some point you notice there's a layer underneath you haven't touched: how models actually get trained, how you improve a pipeline end to end rather than just connect pieces together.

Track 2 sat exactly there: a real dataset, a real evaluation framework, a baseline with a defined margin to beat — run by the company with one of the most well-regarded recommendation systems in the world, on the problem it actually solves at scale. It asks an agent to do what an ML engineer does: hypothesise, write the code, run it, read the result, decide what's next. We understood that loop from one side and wanted to understand it from the other.

## What it does

The agent runs a closed improvement loop with no human input: read the ledger, write a complete Python solution, hand it to a harness that scores it on validation, use the result to choose what's next. It carries no memory between iterations — the ledger *is* the memory, so the run's whole state lives on disk rather than in the model's context.

Selection uses **UCT** (Upper Confidence bounds applied to Trees), from Monte Carlo Tree Search, so the agent can return to a branch abandoned twenty iterations ago instead of only refining its latest attempt.

- **Objective:** rank each user's logged impressions by watch completion (`long_view`)
- **Metric:** mean(GAUC, nDCG@5), scored by the organisers' `evaluate.py`, read-only
- **Splits:** train `04-08→04-21`, validate `04-22→04-28`, test withheld by the harness
- **Budget:** ε = 0.002 over 3 iterations, inside a 50-experiment / 6-hour cap

The harness enforces what the agent can't reason around: it cannot see test, cannot grade itself, cannot resubmit the same solution, cannot hang the run. Every experiment is scored on 3 seeds.

One command, then leave it.

## Results

Validation-best against the organisers' own model. Delta is per metric, equal-weighted, matching the judging formula.

**KuaiRand-Pure — required**

| | official FM | **ours** | delta |
| --- | --- | --- | --- |
| GAUC | 0.6674 | **0.673080** | +0.005680 |
| nDCG@5 | 0.5357 | **0.538397** | +0.002697 |
| primary | 0.6016 | **0.605738** | **+0.004189** |

**KuaiRand-1k — bonus**

| | kit FM baseline | **ours** | delta |
| --- | --- | --- | --- |
| GAUC | 0.6749 | **0.702723** | +0.027823 |
| nDCG@5 | 0.6153 | **0.663265** | +0.047965 |
| primary | 0.6451 | **0.682994** | **+0.037894** |

- The organisers published a baseline for Pure only; the 1k reference is *their own* `baseline.py` run unmodified on 1k.
- **Gain transferred:** on the local test split Pure holds at **+0.004247**, 1k improves to **+0.039355**. The hidden test is theirs to score.
- Seed noise is 0.0008 against a 0.002 target, so Pure clears the bar on each metric independently, not just the mean.
- **Cost:** 32 of 50 iterations, converged rather than truncated, in 2 h 34 m. 2.97M tokens, **$4.52**, 0 GPU-hours, **zero human interventions**.

**The 1k delta is nine times the Pure delta, and that's the whole story of this task.** Only **3.38%** of Pure's validation rows involve a (user, creator) pair seen in training. On 1k it's **33.70%**. Same agent, same harness — ten times the overlap, ten times the delta.

## How we built it

**The loop.** The agent gets the ledger, picks a node, reads its source, writes a new standalone file, runs it. Three seeds, scored on validation, appended as a row. Five tools and nothing else: `read_ledger`, `read_solution`, `write_solution`, `run_experiment`, `web_search`.

**It invented a tool we didn't give it.** There's no edit tool — `write_solution` takes a whole file, and 93% of the time it writes one (median 11.7 KB). But it worked out that when a change is small it can write a *short* file that loads its parent by path and overrides only what differs. Those are 7% of its solutions at under half the size, including the winner of one of our best runs. Nothing in the prompt suggests it.

**The ledger is the memory** — one line per experiment, read in full every turn:

```
| #  | parent | hypothesis                                       | valid  | +/-    | delta   | verdict |
| 1  | -      | control: official FM ported to PyTorch, BCE       | 0.6014 | 0.0002 | -0.0001 | noise   |
| 2  | 1      | replace pointwise BCE with same-user BPR pairs    | 0.6029 | 0.0004 | +0.0014 | noise   |
| 27 | 24     | add a small DeepFM member to the node-24 ensemble | 0.6055 | 0.0002 | +0.0040 | KEPT    |
```

The `parent` column makes it a tree. `+/-` is the seed spread, and the header warns the agent that two rows closer than their spread haven't been told apart — the commonest way to fool yourself on a tight metric. UCT scales value by the organisers' epsilon rather than the observed range, so seed noise can't reorder the ranking.

**Guardrails are mechanical, not prompted.** `harness/` owns data, labels, scoring and ledger; `agent/` owns the prompt, loop and tools. A solution writes predictions only — it never receives a label. Failures return as *text*, not exceptions, so a LightGBM crash becomes a fix on the next iteration rather than a dead run.

**Three decisions that mattered:** porting the baseline to PyTorch first — a wrong hand-derived gradient doesn't crash, it trains toward the wrong objective and reports a plausible number, and the port reproducing 0.6015 vs 0.6016 is why we trusted anything built after it. Three seeds per experiment, because one seed swung 0.0013 on identical code — 65% of the target margin. And a prediction cache, so screening an idea doesn't cost a validation experiment.

## Challenges we ran into

**The convergence rule — and asking instead of guessing.** Taken literally, "no improvement over 3 iterations" fires almost immediately, because three non-improvements in a row is just what the start of a search looks like. Every run stopped at iteration 4 or 5, nine of twelve *below* the target — while the actual best arrived between iteration 23 and 29.

We could have quietly changed it. Instead we emailed the organisers, and the answer went further than expected: the stated rule is **"the organisers' default stopping criterion, not a constraint on your run"** — a team may declare its own ε, N and floor, provided the values are fixed *before* the run and recorded in the run log. Ours are, in every `run_start` event. A deviation became a declared parameter.

They also corrected our arithmetic, which was worth more than the permission — we'd read three iterations as steady progress when those were *cumulative* deltas, and the window's real gain was 0.0010, genuinely below ε. The rule had fired correctly; we'd been misreading our own ledger.

We asked a second question too — could we pre-train on 1k and fine-tune on Pure? No: 1k spans the same window and shares users, so it imports Pure's test period. But they added that our reasoning about Pure's ~52 interactions per user was *sound*. That's our ceiling analysis, confirmed by the people who built the benchmark.

**Our own feedback loop was hours, not minutes.** The agent's loop is fast; ours is a full run — two to six hours to learn whether a harness change helped, 47 hours across 18 runs. And one run can't answer it: twelve converged runs landed within 0.0011 of each other, a run-to-run sd of **0.00036**, so any improvement under ~0.0007 is invisible in a single run. We distrusted single-run conclusions, including the flattering ones.

**The 1k trap.** 1k's nDCG@5 sits ~0.08 above Pure's, so running experiments without declaring the dataset scores everything against Pure's baseline and stamps every result a win. Nothing in the logs looks wrong. The harness now refuses to grade a dataset whose baseline hasn't been measured.

## What we learned

- **The bottleneck was never features or capacity** — the organisers had shown that, and it held. It was the objective: the baseline optimises pointwise log-loss while the metrics are ranking metrics. Almost everything that worked closed that gap.
- **The harness matters more than the prompt.** Seed noise, a rule firing too early, a baseline attached to the wrong dataset — each quietly produces confident wrong conclusions, and none look like bugs in the log.
- **Ask rather than assume.** One email turned our most attackable decision into a compliant declared parameter, and corrected a misreading we'd been building on.
- **An agent that recovers well beats one that never fails.** The best thing in our logs isn't a clean run — it's the agent reading a crash, naming the cause, and fixing it in one step.

## What's next for ZuMianBao

- **Run experiments in parallel.** We profiled a live run at ~20% utilisation — 54 of 64 cores idle — while the agent already asks for several experiments at once and the harness runs them one after another.
- **Finish the bonus benchmarks.** 27k is genuinely hard rather than merely large: 322M rows as Python objects need ~110 GB against a 116 GB container limit, so the loader must discard the test split as it reads. Its baseline is measuring as we write; one full-data experiment costs ~2 hours against a 6-hour ceiling.
- **Two models against one ledger.** The bake-off suggested different models get stuck in different places.

## Built with

**Development tools:** VS Code, Claude Code, Git/GitHub, RunPod (RTX 4090), Windows and Linux.

**APIs:** OpenAI-compatible chat completions — GPT-5.5 for the record runs, with DeepSeek-V4-Pro and GPT-5.6 evaluated on a byte-identical harness. A web-search tool the agent invokes itself.

**Libraries and frameworks:** PyTorch, torchfm, LightGBM, XGBoost, NumPy, pandas, scikit-learn, SciPy, tqdm, `openai`, python-dotenv.

**Datasets:** KuaiRand-Pure (1.4M interactions, required) and KuaiRand-1K (11.7M, bonus). KuaiRand-27K downloaded, baseline in progress. No external or hand-labelled data; the official `evaluate.py` is the sole scoring authority, used unmodified.
