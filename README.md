# An autonomous ML research agent for within-user video ranking

**TechJam 2026, Track 2.** An LLM-driven agent that improves a recommender
pipeline on KuaiRand-Pure without human help: it forms a hypothesis, writes a
complete solution, has it scored by a harness it cannot influence, reads the
result, and decides what to try next — until the organisers' convergence rule
stops it.

```
official FM baseline (valid)   0.6016
our agent's converged result   0.6055        +0.0040
the organisers' bar            +0.0020       (epsilon, ~2.5 sigma)
```

The agent found that gain on its own. It searched the literature, tried six of
the seven suggested directions, diagnosed its own crashes, and stopped when it
stopped making progress. **Zero human interventions during the run.**

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

`logs/record-run-3/README.md` is the most detailed write-up; the others record
what each run was for and what it exposed.

---

## What the agent found

Its best result is an ensemble of pairwise-ranking models with temporal context
and a watch-time-weighted member. The path it took, with the delta over the
official baseline at each step:

```
#2   plain BPR, replacing pointwise BCE          +0.0014
#8   average three independently seeded BPRs     +0.0022
#10  extend that to seven members                +0.0030
#17  hour/day temporal context (after 3 debugs)  +0.0036
#24  a watch-time-weighted member added          +0.0040
```

Every step was the agent's own choice, and the hypothesis in the ledger explains
why it made each one. Note `#17` — the temporal feature took **three debugging
iterations** to align correctly against the raw logs, dropping to 0.5829 at one
point before the agent found that its lookup key included a field the raw logs
do not carry.

Three findings worth stating, all of which the agent reached by measurement:

**Changing the loss is the only thing that reliably works.** Time features,
multi-task auxiliary heads, sequence attention and extra capacity each added
nothing measurable on top of a pairwise loss — five directions, all correctly
implemented, all inside their own seed spread.

**Listwise softmax is wrong for this data.** Four independently written
implementations, across four runs that could not see each other's ledgers,
scored **-0.0026, -0.0051, -0.0056 and -0.0020**. A softmax over a user's
impressions assumes exactly one relevant item, but this dataset is 33% positive,
so a user's positives compete against each other and the loss penalises ranking
the second one highly.

**Watch time is harmful as a target and useful as a weight.** Ranking by raw
play time scored -0.0375, because `long_view` is watch time *relative to
duration* — so raw play time simply favours long videos. The same signal used to
weight the training loss gained +0.0006.

---

## Limitations, and what we would do with more time

### The task is close to saturated, and we can show why

```
oracle ceiling (valid)                  0.8484   requires the labels
perfect knowledge of every video        0.6197   knowing nothing about the user
our result                              0.6055
official baseline                       0.6016
```

(That 0.6197 is measured by fitting each video's true rate on the validation set
itself, so it is optimistic — estimate the rates from half the data and it falls
to 0.6048. The honest non-personalised ceiling is somewhere in 0.605-0.62, which
is to say we are at or near it.)

The gap between 0.6055 and 0.8484 looks like room. It is not. **Only 1.62% of
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

**Our own search wasted budget.** Run 3 committed to a 10-model ensemble by
experiment 24, after which every new idea was tested as a 5% weighted addition
to it — arithmetically incapable of changing the outcome, at 13-16 minutes per
attempt. Five genuinely different mechanisms were tested that way and none could
be resolved. We have since added a policy requiring a mechanism to be validated
standalone before it is blended, but it landed too late to benefit that run.

**We never used the randomised-exposure data.** `log_random` supports unbiased
off-policy evaluation, which would tell us whether our gains are real or an
artefact of biased logging. It cannot be trained on — it spans the evaluation
window — but it is the strongest available check on a result selected across
thirty experiments on validation alone.

**We never attempted the bonus benchmarks.** KuaiRand-1k has ~11,700
interactions per user against Pure's 52 — roughly 226x more history per person,
which is exactly the constraint identified above. It is the one place where
personalisation would genuinely become learnable.

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
| **Zheng** ([@sgzm2011](mailto:sgzm2011@gmail.com)) | Agent loop implementation; per-run output management; record run 3; test-scoring tool; ensemble solutions |

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
