# An autonomous ML research agent for within-user video ranking

**TechJam 2026, Track 2.** An LLM-driven agent that improves a recommender
pipeline on KuaiRand-Pure without human help: it forms a hypothesis, writes a
complete solution, has it scored by a harness it cannot influence, reads the
result, and decides what to try next — until the organisers' convergence rule
stops it.

```
KuaiRand-Pure (required)   valid      test
official FM baseline       0.6016     0.5946
our agent's result         0.605738   0.598847
delta                      +0.0042    +0.0042
the organisers' bar        +0.0020    (epsilon, ~2.5 sigma)

KuaiRand-1k (bonus)        valid      test
kit FM baseline            0.6451     0.6390
our agent's result         0.682994   0.678305
delta                      +0.0379    +0.0394
```

The gain transferred on both — it did not evaporate off validation. The test
rows ship with the Starter Kit, so these are scored locally by discipline
rather than withheld; final judging scores the submitted predictions.

The agent found that gain on its own. It searched the literature, tried six of
the seven suggested directions, diagnosed its own crashes, and stopped when it
stopped making progress. **Zero human interventions during either run.**

## Demo

https://github.com/user-attachments/assets/caa2dd3d-ed48-4b82-ae48-2f87aaf36bc1

The 1k delta is nine times the Pure delta, from the same agent and the same
harness. That gap is the most informative result here, and the reason is
structural — see the limitations section.

---

## The task

Within-user ranking over logged impressions. For each user, order the videos
they were actually shown; the label is `long_view`. Scored on
**mean(GAUC, nDCG@5)** by `kuairand-starter-kit/evaluate.py`, which we treat as
read-only.

Not retrieval — there is no catalogue to search. The candidate set is fixed and
the only question is the order.

---

## How it works

```
agent/loop.py           ask the model -> run its tools -> log -> repeat
   |
   +-- agent/prompt.py     the task, the rules, and the ledger rebuilt each turn
   +-- agent/tools.py      five tools, and the jail that keeps them honest
   |
   v
harness/run.py          runs a solution on 3 seeds, scores it, never raises
   |
   +-- harness/ledger.py   logs/<run>/NNNN.json + LEDGER.md + events.jsonl
   v
solutions/NNN_*.py      standalone scripts the agent writes
```

Each iteration the agent gets a freshly rebuilt message containing the full
experiment ledger and the current best solution's source. It has no memory
between iterations — **the ledger is the memory**, which is why a killed run can
be restarted and resume knowing everything it learned.

### The guardrails are mechanical, not prompted

This matters more than any of the modelling. An agent that behaves because it
was asked nicely is not a robust agent.

| | how it is enforced |
| --- | --- |
| Cannot see the test split | the harness chooses the split; `--split test` is refused |
| Cannot grade itself | solutions emit predictions only; the harness owns the labels |
| Cannot escape `solutions/` | write and read paths resolve the realpath and refuse |
| Cannot silently repeat itself | source hash detects a rerun; identical metrics flag a no-op |
| Cannot hang or crash the run | 15-min timeout, `ast.parse` first, every failure caught |
| Cannot mistake noise for progress | every experiment is scored on 3 seeds and reports its spread |

The agent never had to be trusted with any of these.

---

## Setup

Python 3.12. Install torch first — the CUDA build cannot be resolved from PyPI,
and installing it via `requirements.txt` silently gives you the CPU wheel.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.6.0+cu124 True
```

Download **KuaiRand-Pure** from https://kuairand.com and place it at
`rec_datasets/KuaiRand-Pure/data/`. The dataset is ~195 MB and is not committed.

Copy `.env.example` to `.env` and add an OpenAI API key.

---

## Reproducing our results

### 1. Check the harness against the published numbers

Two checks, in order. First the organisers' own script, run directly — it is
untouched and does not follow our harness contract, so it is not run through
the harness:

```bash
python kuairand-starter-kit/baseline.py --model random   # expect primary ~0.4834 valid
python kuairand-starter-kit/baseline.py --model fm       # expect primary ~0.6016 valid
```

The `random` check matters more than it looks: if it does not land near 0.4834,
the scoring path itself is broken and every number downstream is meaningless.

Then our PyTorch port, through the harness:

```bash
python harness/run.py solutions/001_torch_fm.py --by human   # expect ~0.6014 valid
```

All three should land within 0.0008 — the organisers' 5-seed standard deviation
— of the published numbers. The port reproducing the baseline is what licenses
everything built on top of it.

### 2. Run the agent

```bash
python -m agent --run-name record-run
```

Creates `logs/record-run-N/`, seeds it with the human-run control at iteration 1,
and runs unattended until it converges or hits a cap (50 experiments, 6 h).
Takes 1-4 h depending on how expensive the solutions it writes turn out to be.

Follow it in a second terminal:

```bash
python harness/watch.py
```

### 3. Build and check a submission

```bash
python harness/make_submission.py <best_solution>.py --split test --out submission.csv
```

Writes through the official `submit.py` and validates with the official
`read_submission`, so the file is checked by the organisers' code, not ours.

### Verifying a run rather than repeating it

Every run is archived under `logs/` with its solutions, its full per-iteration
records, and its event log. Each has a matching immutable git tag
(`record-run-4-code`) pointing at the exact code that produced it.

The submitted Pure run is `logs/record-run-9`; the submitted 1k run is
`logs-1k/record-run-4`. `logs/record-run-1/README.md` and
`logs/record-run-2/README.md` carry the most detailed narrative write-ups, from
when the harness was still being shaken out.

---

## What the agent found

The submitted result is a seed-bagged pairwise-ranking ensemble with temporal
context, exposure counts and a global-CTR tie-break. The path `record-run-9`
took, with the delta over the official baseline at each new best:

```
#2   plain BPR, replacing pointwise BCE            +0.0017
#6   bag several seeded BPRs, blend by rank        +0.0030
#8   a DIN-inspired behaviour summary              +0.0031
#10  video content features on a sparse FM         +0.0036
#15  hour and weekday temporal context             +0.0037
#24  unlabeled sequential exposure counts          +0.0039
#27  fix a seed realisation that was unlucky       +0.0042
#31  global-CTR tie-break on close calls           +0.0042
```

Every step was the agent's own choice, and the hypothesis in the ledger explains
each one. Note `#27` and `#28`: the agent noticed a seed had landed badly,
wrote a fixed-seed wrapper, found it was a **no-op** because its predictions were
identical to the parent's, and said so in the next hypothesis before trying
again. Recognising its own null result is the behaviour we most wanted to see.

Record run 3 took a different route to nearly the same place — BPR bagging, then
temporal context after three debugging iterations, then a watch-time-weighted
member, ending at +0.0039. Four runs converging on the same score by four
different mechanisms is the substance of the saturation argument below.

Three findings worth stating, all of which the agent reached by measurement:

**Changing the loss is the only thing that reliably works.** Time features,
multi-task auxiliary heads, sequence attention and extra capacity each added
nothing measurable on top of a pairwise loss — five directions, all correctly
implemented, all inside their own seed spread.

**Direct nDCG optimisation does not beat a bagged FM here.** We added `xgboost`
and `lightgbm` to the environment and removed every hint pointing at them from
the prompt. The agent found LightGBM on its own, wrote a competent LambdaRank —
grouped by user, `label_gain: [0, 1]`, native categoricals — and measured it at
**-0.0032** standalone, **+0.0025** reused as a rank correction on top of the FM.
Both with real error bars. The single most plausible untried direction, tested
properly, loses.

**Listwise softmax is wrong for this data.** Six independently written
implementations, across five runs that could not see each other's ledgers,
scored **-0.0026, -0.0051, -0.0056, -0.0020, -0.0049 and -0.0049**. A softmax over a user's
impressions assumes exactly one relevant item, but this dataset is 33% positive,
so a user's positives compete against each other and the loss penalises ranking
the second one highly.

**Watch time is harmful as a target and useful as a weight.** Ranking by raw
play time scored -0.0375, because `long_view` is watch time *relative to
duration* — so raw play time simply favours long videos. The same signal used to
weight the training loss gained +0.0006.

---

## Which run we submitted, and why

Pure has sixteen record runs; the submitted one is `record-run-9`. Run 3 scored
0.605493 on validation against run 9's 0.605738 — a gap of 0.000245, well inside
the 0.0008 seed spread — so score is not the reason. Two other things are.

**The hard caps.** Run 9's `run_start` event records
`min_scored_before_convergence: 30`, `max_experiments: 50` and
`max_wall_seconds: 21600`, and it converged in 2 h 34 m. Run 3's records
`max_experiments: 80` with no wall-clock ceiling, and it ran 6 h 29 m — outside
the 50-iteration and 6-hour limits. The organisers permit a team to declare its
own epsilon, N and minimum-iteration floor provided the values are fixed before
the run and recorded in the run log; run 9 satisfies that condition, which is
why the floor is a declared parameter rather than a deviation.

**Test labels.** Run 3's winner builds a raw-log lookup keyed on
`(date, user, video, tab, dur, y)` where `y` is the label, and applies it to
every split including test with no guard — so which raw record, and therefore
which `hourmin`, attaches to a test row is conditioned on that row's label. The
channel is narrow, biting only on the 3.06% of test pairs that are duplicated,
but it is a test label being read. Run 9's lineage does the equivalent work and
guards it: `row_feats(row, update_label=(sp=='train'))` in node 24, and
`if sp == 'train'` around the label counters in node 28. It walks all three
splits to accumulate exposure counts, which are label-free, and never updates
label-derived state outside train.

Run 3 stays in the repository. It is part of the run record, and removing a
result because it was not submitted would be the wrong instinct.

---

## Limitations, and what we would do with more time

### The task is close to saturated, and we can show why

```
oracle ceiling (valid)                  0.8484   requires the labels
perfect knowledge of every video        0.6197   knowing nothing about the user
our result                              0.6057
official baseline                       0.6016
```

(That 0.6197 is measured by fitting each video's true rate on the validation set
itself, so it is optimistic — estimate the rates from half the data and it falls
to 0.6048. The honest non-personalised ceiling is somewhere in 0.605-0.62, which
is to say we are at or near it.)

Four independent searches make the same point empirically. Four runs, four
mechanism families, none able to read another's ledger:

```
global-CTR tie-break on a seed ensemble  0.605738   <- submitted
BPR ensemble + watch-time weighting      0.605493
mixed tab/hour ensembles                 0.605368
FM rank ensemble + DeepFM blend          0.605024
same-user sampled softmax                0.604615
                            mean 0.605248, sd 0.000397
```

0.610 sits twelve standard deviations above that mean. This is not a search that
needs more attempts; it is a task that has run out of signal.

The gap between 0.6057 and 0.8484 looks like room. It is not. **Only 1.62% of
validation rows involve a (user, video) pair the model has ever seen in
training**, and the median user has 31 training rows covering 29 videos by 29
different creators. There is almost no repetition from which to learn a person's
taste, so a model falls back on "which videos are good in general" — and that
tops out at 0.6197 even with perfect knowledge.

Two further limits are structural: 30.3% of users watched nothing at all, so
nDCG@5 scores them 0 no matter what (its true ceiling is 0.697, not 1.0); and
the same user shown the same video on a *different day* changes their mind
23-35% of the time, which no model can predict from these features.

### What we would improve

**Our own search wasted budget.** Record run 3 — not the submitted run, but
instructive — committed to a 10-model ensemble by experiment 24, after which every new idea was tested as a 5% weighted addition
to it — arithmetically incapable of changing the outcome, at 13-16 minutes per
attempt. Five genuinely different mechanisms were tested that way and none could
be resolved. We have since added a policy requiring a mechanism to be validated
standalone before it is blended, but it landed too late to benefit that run.

**We never used the randomised-exposure data.** `log_random` supports unbiased
off-policy evaluation, which would tell us whether our gains are real or an
artefact of biased logging. It cannot be trained on — it spans the evaluation
window — but it is the strongest available check on a result selected across
thirty experiments on validation alone.

**We did attempt the bonus benchmarks, and 1k confirmed the diagnosis above.**
1k has ~11,700 interactions per user against Pure's 52, and **33.70%** of its
validation rows involve a (user, creator) pair seen in training against Pure's
**3.38%**. Same agent, same harness, same code — and nine times the delta.
That is the cleanest evidence we have that Pure's ceiling is a property of the
data rather than of our search.

27k we measured a baseline for (0.665079 validation) but could not run the agent
on. 322M rows as Python tuples need ~110 GB against a 116 GB container limit; the
loader now discards the test split as it reads, which is enough for a baseline at
71 GB but not for a solution doing real work on top. Six agent attempts were all
OOM-killed with exit code -9. Making it viable needs the loader rewritten to
numpy arrays rather than Python tuples — the streaming rewrite our own notes
predicted from the start. We measured the alternative first: subsampling the
training split by 10x costs 0.0329 on 1k, against an agent whose best gain on any
dataset is 0.0379, so buying the speed that way costs almost everything it could
win.

**The agent writes everything from scratch.** Across four runs it hand-wrote
DeepFM, DIN-style attention and multi-task heads from memory, never using an
existing implementation, because nothing in its prompt said libraries were
available. That is a plausible source of false negatives on the harder
directions.

---

## Team

| | |
| --- | --- |
| **Bryan Tan** ([@Bryantan65](https://github.com/Bryantan65)) | Harness, ledger and guardrails; agent loop debugging; multi-seed scoring; convergence rule; robustness and run logging; submission tooling; benchmark analysis |
| **Zheng** ([@sgzm2011](mailto:sgzm2011@gmail.com)) | Agent loop implementation; per-run output management; record runs 3 and 9; test-scoring tool; ensemble solutions |
| **Kaibao** | KuaiRand-1k runs on GPU, including the submitted `logs-1k/record-run-4` |

Submitted: `logs/record-run-9` for Pure, `logs-1k/record-run-4` for the 1k bonus.

`src/` vendors the CWM reference implementation (hyz20/CWM, arXiv 2406.07932),
ported to a modern stack. It is reference material and is not part of the
submitted pipeline.

---

## Repository map

| | |
| --- | --- |
| `agent/` | the loop, its five tools, and the prompt |
| `harness/` | scoring, the ledger, seed sweeps, the run watcher, submission tooling |
| `solutions/` | the substrate the agent branches from |
| `logs/<run>-N/` | one folder per run: per-iteration records, event log, solutions |
| `kuairand-starter-kit/` | the organisers' kit. **Read-only.** |
| `HANDOFF.md` | the decisions and the reasoning behind them |
| `CLAUDE.md` | the task facts: label, metrics, splits, baselines, dead ends |
| `src/` | CWM reference implementation, not part of the submission |
