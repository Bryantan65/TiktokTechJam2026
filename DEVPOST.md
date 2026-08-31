# ZuMianBao — TechJam 2026 Track 2

## Inspiration

We'd been spending time building with AI, mostly at the application level — wrapping LLMs around real problems, putting agents together that coordinate tools and automate workflows. That kind of work teaches you a lot about how to orchestrate AI, but at some point you start noticing there's a layer underneath you haven't really touched: how models actually get trained, how you improve a pipeline end to end rather than just connect pieces together.

That was roughly where we were when we came across TikTok TechJam. The setup felt serious in the right way — a real dataset, a real evaluation framework, a baseline with a defined margin to beat. And the company running it has one of the most well-regarded recommendation systems in the world, so there was something fitting about the problem being exactly that: ranking content for users, which is the actual thing TikTok does at scale.

Track 2 was an easy pick for us. It sat at the intersection of what we'd been doing and what we genuinely wanted to get better at. Building agents — we had some experience with. ML engineering from the ground up — less so. The task asks an agent to do what an ML engineer actually does on a real project: form a hypothesis, write the code, run it, read the result, and decide what's next. That loop was something we understood from one side and wanted to understand from the other. We thought putting both together was worth a serious attempt.

## What It Does

The agent runs a closed improvement loop without human input. Each iteration it reads the current task state, writes a complete Python solution, hands it to a harness that scores it on validation data, and uses the result to decide what to try next. It has no persistent memory between iterations — the experiment ledger serves as memory instead, so the run's full state lives on disk rather than in the model's context.

Experiment selection uses UCT (Upper Confidence Bounds applied to Trees), the selection rule from Monte Carlo Tree Search. Rather than always refining the latest attempt, the agent maintains a tree of experiments and can return to an earlier branch when a newer line stops making progress.

The task and constraints are fixed:

- **Objective:** rank each user's logged impressions by predicted watch completion (label: `long_view`)
- **Metric:** mean(GAUC, nDCG@5), scored by the organisers' `evaluate.py`, treated as read-only
- **Data:** train 20220408–20220421, validate 20220422–20220428, test withheld by the harness
- **Convergence:** ε = 0.002 over 3 consecutive iterations, within a 50-experiment / 6-hour cap

The harness enforces hard constraints the agent cannot reason its way around: it cannot see the test split, cannot grade itself, cannot silently resubmit the same solution, and cannot hang the run. Every experiment is scored on 3 seeds to separate real signal from noise.

One command, then leave it.

## Results

Validation-best, head to head against the organisers' official FM baseline. Delta is each metric's improvement, equal-weighted, matching the judging formula.

### KuaiRand-Pure (required benchmark)

|               | official FM | ours     | delta     |
| ------------- | ----------- | -------- | --------- |
| GAUC          | 0.6674      | 0.673080 | +0.005680 |
| nDCG@5        | 0.5357      | 0.538397 | +0.002697 |
| **primary**   | **0.6016**  | **0.605738** | **+0.004189** |

### KuaiRand-1k (bonus benchmark)

|               | kit FM baseline | ours     | delta     |
| ------------- | --------------- | -------- | --------- |
| GAUC          | 0.6749          | 0.702723 | +0.027823 |
| nDCG@5        | 0.6153          | 0.663265 | +0.047965 |
| **primary**   | **0.6451**      | **0.682994** | **+0.037894** |

The organisers published a baseline for Pure only. The 1k reference is their own `baseline.py`, run unmodified on 1k data.

- **Gain transferred:** scored once on the local test split, Pure holds at +0.004247 and 1k improves slightly to +0.039355
- **Seed noise** is 0.0008 and the target margin was 0.002, so Pure clears the bar on each metric independently, not just on the mean
- The 1k delta is nine times the Pure delta, and that is the whole story of this task. Only 3.38% of Pure's validation rows involve a (user, creator) pair seen in training. On 1k it is 33.70%. Same agent, same harness, ten times the overlap, ten times the delta.
- **Cost to get there:** 32 of 50 iterations, converged rather than truncated, in 2 h 34 m. 2.97M tokens, $4.52, 0 GPU-hours, zero human interventions.

## How We Built It

### The Loop

Each iteration the agent is handed the full experiment ledger, picks a node to build on, reads that solution's source, writes a new standalone Python file, and hands it to the harness. The harness runs it on three seeds, scores predictions on validation, appends a row to the ledger, and the next turn begins.

Five tools, nothing else: `read_ledger`, `read_solution`, `write_solution`, `run_experiment`, `web_search`.

There is no edit tool. `write_solution` takes a complete file, and 93% of the time the agent writes one (median 11.7 KB). But it worked out on its own that when a change is small, it can write a short file that loads its parent by path and overrides only the differing part. That pattern accounts for 7% of solutions at less than half the size, including the winning solution in one of our best runs. Nothing in the prompt suggests it.

### Memory and Selection

The ledger is the agent's memory — one line per experiment read in full every turn:

| #  | parent | hypothesis                                          | valid  | +/-    | delta   | verdict | by    |
| -- | ------ | --------------------------------------------------- | ------ | ------ | ------- | ------- | ----- |
| 1  | -      | control: official FM ported to PyTorch, BCE          | 0.6014 | 0.0002 | -0.0001 | noise   | human |
| 2  | 1      | replace pointwise BCE with same-user BPR pairs       | 0.6029 | 0.0004 | +0.0014 | noise   | agent |
| 27 | 24     | add a small DeepFM member to the node-24 ensemble    | 0.6055 | 0.0002 | +0.0040 | KEPT    | agent |

The `parent` column is what makes this a tree rather than a list. `+/-` is the spread across seeds, and the ledger header tells the agent directly that two rows closer together than their spread have not been told apart — which is the single most common way to fool yourself on a tight metric.

Selection uses UCT: each node's value is how close it came to the current best, measured in units of the organisers' epsilon, plus an exploration bonus that shrinks as the node accumulates children. Scaling by epsilon rather than by the observed range means seed noise cannot reorder the ranking. A promising branch abandoned twenty iterations ago can win selection and get revisited.

### Guardrails

`harness/` owns the data, labels, scoring, and ledger. `agent/` owns the prompt, loop, and tools. A solution is a standalone file that writes predictions only — it never receives a label and never computes a score. `run_experiment` accepts only `valid` or the train-only holdout and refuses everything else at both the tool layer and the harness.

Failures return as text rather than exceptions. A crash, timeout, NaN predictions, or a duplicate solution all come back as something the agent reads and acts on. A LightGBM crash became a fix on the next iteration rather than a dead run.

### Three Decisions That Mattered More Than Expected

1. **Ported the baseline to PyTorch first.** The kit's FM derives gradients by hand, and every later experiment changes the loss function. A wrong hand-derived gradient doesn't crash — it trains toward the wrong objective and reports a plausible number. The port reproduced the official score (0.6015 vs 0.6016), which is the only reason we trusted everything built on top of it.

2. **Three seeds per experiment.** One seed swung 0.0013 on identical code — 65% of the entire target margin.

3. **Prediction cache and train-only holdout**, so blend experiments reuse existing members instead of retraining, and screening an idea doesn't cost a validation experiment.

## Challenges We Ran Into

Most of the work was in the harness, not the agent. A solution must not be able to cheat, a failed run must not look like a good one, and no metric should move without being able to explain why.

### Model Selection

We ran a bake-off across DeepSeek-V4-Pro, GPT-5.5, and GPT-5.6 on a byte-identical harness so only the reasoning engine differed. GPT-5.5 won.

### The Convergence Rule

The longest single debugging effort. Taken literally, "no improvement over the last 3 iterations" fires almost immediately, because three non-improvements in a row is just what the beginning of a search looks like. Every run was stopping at iteration 4 or 5, with nine of twelve below the target score. The actual best result in every run arrived between iteration 23 and 29. So we asked the organisers rather than guessing. They confirmed the stated rule is "the organisers' default stopping criterion, not a constraint on your run" — a team may declare its own ε, N and minimum-iteration floor, provided the values are fixed before the run and recorded in the run log. Ours are, written into every `run_start` event before a single experiment runs.

### Our Own Feedback Loop

The agent's loop is fast. Ours is a full run — two to six hours — then find out whether a harness change helped. That came to 47 hours across 18 runs. And one run doesn't answer the question: twelve converged runs landed between 0.604615 and 0.605738, a run-to-run standard deviation of 0.00036, meaning any improvement under ~0.0007 is invisible in a single run. We batched changes, preferred ones we could justify from an offline measurement, and distrusted single-run conclusions — including the flattering ones.

### Handling Failure

A crash has to be information, not a dead end. When the agent hit LightGBM's hard 10,000-row-per-query-group limit, it read the error, split oversized users into chunks under the cap, and recovered on the next iteration — across three separate runs independently.

### The 1k Benchmark

The organisers published baselines for KuaiRand-Pure only, but bonus scoring is `delta = score_agent - score_baseline` per dataset. We produced the 1k baseline by running the kit's own `baseline.py` unmodified on 1k data — same FM, same k=16, lr=0.001, same five fields. It gave valid 0.6451, test 0.6390.

1k also hides a silent trap: its nDCG@5 sits about 0.08 above Pure's, so running experiments without declaring which dataset you're on scores everything against Pure's baseline and stamps every result a win. Nothing in the logs looks wrong. The harness now refuses to grade a dataset whose baseline hasn't been declared.

### Label Leakage

On the second 1k run, the agent discovered it could bypass the model entirely. At iteration 42 it read `play_time_ms` and `duration_ms` from the raw CSVs and reconstructed the `long_view` label using its public definition — primary jumped from 0.6617 to 0.9877. One iteration later it found the `long_view` column itself in the CSV and read it directly, hitting 0.9974. It then spent its last seven iterations stuck on the oracle branch trying to improve a near-perfect score, wasting 9 of 50 experiments before the run hit its cap.

The prompt hadn't forbidden it, and the harness hadn't flagged it — from the logs it looked like a legitimate breakthrough. We added both layers: an explicit forbidden-action block in the prompt, and a harness-level guardrail that stamps any result above 0.90 as `LABEL LEAKAGE DETECTED` regardless of what the prompt says. No subsequent run repeated the exploit.

This was the clearest example of why the harness matters more than the prompt. A well-written instruction can prevent the behaviour most of the time, but a scoring constraint that the agent cannot override is what actually makes the measurement trustworthy.

## Accomplishments We're Proud Of

- **Beat the baseline on both benchmarks.** Pure by +0.0042, 1k by +0.0394. The target was +0.002 against 0.0008 of seed noise, and the gain held all the way through to test rather than evaporating there.
- **Zero human interventions.** One command, untouched until convergence. We started runs overnight and came back to a finished result with a complete audit trail every time.
- **Runs finish in about two and a half hours** without weakening the measurement.
- **The agent found things we didn't hand it.** It discovered seed-ensembling to cut variance on its own, and abandoned LambdaRank after finding it unstable, without being told either were options.

## What We Learned

- **The bottleneck was never features or capacity.** The organisers had already shown that, and it held: almost everything that moved the metric closed the gap between the training objective (pointwise logloss) and the evaluation metrics (GAUC, nDCG@5, both ranking-based).
- **The harness matters more than the prompt.** Seed noise, a convergence rule that fires too early, a baseline attached to the wrong dataset — each of these quietly produces confident, wrong conclusions, and none of them look like bugs in the log.
- **An agent that recovers well beats one that never fails.** The most convincing thing in our logs isn't a clean run — it's the agent reading a crash, naming the cause, and fixing it on the next iteration.

## What's Next for ZuMianBao

- **Parallel experiments.** We profiled a live run at roughly 20% utilisation, 54 of 64 cores idle, while the agent already asks for several experiments at once on some turns and the harness runs them one at a time.
- **Cache the parsed data.** Measured at 6–7× faster with byte-identical output: about 8% off a Pure experiment, a third off a 1k one.
- **Finish the bonus benchmarks.** 1k is where the score is still moving. 27k we assessed and set aside deliberately — it's wider than 1k not deeper (~11,800 impressions per user in both), and the one mechanism it could uniquely supply moved GAUC by +0.00073, t = 0.33.
- **Run two models against one ledger.** The bake-off suggested different models get stuck in different places.

## Built With

- **Development tools:** VS Code, Claude Code, Git/GitHub, RunPod (RTX 4090) for GPU runs, Windows and Linux
- **APIs:** OpenAI-compatible chat completions, GPT-5.5 for the record runs, with DeepSeek-V4-Pro and GPT-5.6 evaluated on an identical harness. A web-search tool the agent invokes itself when it wants to look up a method.
- **Libraries:** PyTorch, torchfm, LightGBM, XGBoost, NumPy, pandas, scikit-learn, SciPy, tqdm, openai, python-dotenv
- **Datasets:** KuaiRand-Pure (1.4M interactions, required) and KuaiRand-1K (11.7M, bonus). KuaiRand-27K was downloaded and assessed but not run. No external or hand-labelled data. The official `evaluate.py` is used unmodified as the sole scoring authority.
