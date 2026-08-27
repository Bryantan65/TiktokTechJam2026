# Handoff

State, decisions, and what's next. **`CLAUDE.md` holds the task facts** — label,
metrics, splits, baselines, dead ends. This file holds the *decisions* and the
reasoning behind them, which is the part that's expensive to reconstruct.

Last updated: 2026-08-27, commit `a146dfa` + this change.

---

## Where things stand

```
✅ environment verified against the official numbers
✅ official baseline reproduced (numpy) and ported (PyTorch)
✅ harness built and its guardrails tested
✅ agent loop built, debugged, and beating the baseline
✅ shakedown 01 + 02 (12 experiments, 8 bugs found and fixed)
✅ token cost cut 3x and measured
✅ robustness: API retry, run log, atomic writes
⬜ record run — fresh ledger, one uninterrupted process
```

Best so far: **valid primary 0.603999** (`solutions/012_softmax2_bce0025_fm.py`), against the 0.6035 target.
Spend to date: 422k in / 33k out, $2.79, 396 s compute over 12 experiments.

### What's built

| | |
| --- | --- |
| `kuairand-starter-kit/` | official kit. **Read-only, never modify.** `evaluate.py` is the sole authority on scoring. |
| `solutions/000_baseline.py` | untouched copy of the official FM |
| `solutions/001_torch_fm.py` | same model in PyTorch. **The substrate every experiment builds on.** |
| `harness/run.py` | runs one solution, scores it, logs it. Never raises. |
| `harness/ledger.py` | `logs/iterations/NNN.json` + `LEDGER.md`; owns the verdict rule |
| `agent/loop.py` | the while-loop: call model → run tools → annotate → repeat |
| `agent/tools.py` | the 5 tools the model can call, and their write jail |
| `agent/prompt.py` | system prompt + the per-iteration message, rebuilt from the ledger |
| `logs/shakedown-01/` | first debugging run, archived with what it found |
| `src/` | CWM paper code, ported and working. **Reference only** — see below. |

### Verified numbers

All on `valid`, which is the only split the agent sees.

```
random self-check   0.4757   (official 0.4753)   scoring path is correct
FM numpy            0.6015   (official 0.6016)
FM PyTorch          0.6014   (official 0.6016)   the port is correct
```

Both ports also match epoch-by-epoch and early-stop at the same epoch (11),
which is stronger evidence than the final number alone. Runtime: 50 s numpy,
32 s torch CPU, 26 s torch CUDA.

**Target: valid 0.6035** (baseline 0.6015 + the official ε of 0.002).

### What the agent has found so far

Every row is agent-authored except iteration 1, the human-run control.

```
 #  solution                             GAUC     nDCG@5   primary   verdict
 1  001_torch_fm.py                    0.667078 0.535722  0.601400  control
 2  002_bpr_fm.py                      0.669442 0.536784  0.603113  noise
 3  003_bpr_2neg_fm.py                 0.669911 0.536633  0.603272  noise
 4  004_bpr_4neg_fm.py                 0.669371 0.536566  0.602969  noise
 5  005_bpr2_bce_aux_fm.py             0.670322 0.536826  0.603574  KEPT
 6  006_bpr2_bce_aux02_fm.py           0.670300 0.536777  0.603539  KEPT
 7  007_hardneg_bpr2_bce_fm.py         0.663342 0.532930  0.598136  worse
 8  008_softmax2_bce_fm.py             0.670583 0.537236  0.603910  KEPT
 9  009_softmax3_bce_fm.py             0.670432 0.536940  0.603686  KEPT
10  010_softmax2_unique_negs_bce_fm.py 0.670412 0.536897  0.603654  KEPT
11  011_softmax2_bce005_fm.py          0.670612 0.537249  0.603930  KEPT
12  012_softmax2_bce0025_fm.py         0.670660 0.537338  0.603999  KEPT
```

All twelve sit inside direction 1 (change the loss) — the organisers' own
top-ranked direction. It never reached direction 2. Two things worth noting
before reading this as success:

- **The gain is +0.0025 against a 0.0008 seed σ.** That clears 2σ, but only
  just, and none of it has been confirmed on a second seed. Do that before
  treating 012 as the baseline for anything.
- **Iterations 8-12 are the same idea five times.** Sampled softmax with the
  auxiliary weight nudged: 0.10, 0.05, 0.025. The spread across all five is
  0.0003, well inside noise. That is the search grinding, not exploring — and it
  is exactly what `converged()` exists to stop.

The one genuinely informative failure is **7 (hard-negative mining, -0.0034)**.
It is also the only iteration where the agent proposed a mechanism rather than
a parameter, which is the behaviour Innovation (20%) is scored on.

---

## Decisions, and why

### The agent writes whole solution files, not configs or diffs

Rejected three alternatives:

- **Config-only** — the organisers already measured the two config dimensions
  that exist (capacity, features) and published both as dead ends. The spread
  across `k = 8/16/32` is 0.0015, *below* the 0.0008×2 noise floor. And their
  top three directions (pairwise loss, sequences, multi-task) are all new code,
  not values. A config-only agent is grid search with an LLM attached.
- **Diffs/patches** — fail to apply, apply partially, match the wrong location.
- **Hybrid config + plugin modules** — buys ~$4 of token savings across the whole
  competition, at the cost of a schema and a module loader. The plugin pattern
  also assumes you know in advance *which parts* vary; each of the organisers'
  directions varies a different part.

At 118 lines a solution costs ~1,500 output tokens ≈ $0.018. Input dominates and
is identical either way, so whole-file emission costs ~$3.60 more than
config-only across 200 iterations. Not a consideration.

**But: whole file for a new idea, targeted edit for a bugfix.** MLE-bench's
published lessons are explicit that precise debugging beats full rewrites —
regenerating to fix one error risks introducing a second. AIDE splits the same
way (draft/improve vs debug).

### Solutions emit predictions, never metrics

The harness owns the labels and calls `evaluate()` itself.

MLE-bench names *"never hard-code evaluation metrics"* as a documented failure:
an agent that grades itself eventually reports a number it didn't earn, and the
search then optimises toward a fiction. It also means a solution cannot choose
which split it is scored on.

Contract:

```
python solutions/NNN.py --data_dir DIR --split valid --out FILE.npy
→ one float per row, in split order, exit 0
```

### PyTorch, even though the score is identical

The official baseline derives its gradient by hand (`g = sigmoid(z) - y`). Every
promising direction starts by changing the loss, and a wrong hand-derivation
**doesn't crash** — it trains toward the wrong objective and reports a plausible
number. That would record a false negative against the organisers' top-ranked
direction, and nothing in the metrics would reveal it.

With autograd the loss is one line and the gradient is correct by construction.
Side benefit: 2× faster, so twice the iterations per hour.

This is also why the planned `reference/` shelf of hand-verified loss
implementations was **dropped** — the port removed most of the risk it addressed,
and pre-writing BPR edges into doing the agent's job, which is what Innovation
(20%) is scored on.

### No framework

The loop is a `while` with one `if`. LangGraph is for branching graphs;
PydanticAI is an abstraction over something already small. Three costs that
matter here: a heavy dependency tree in a repo whose house style is *"Python
3.9+ and numpy, nothing else"*; hidden prompt overhead that breaks prefix
caching (token spend is 15% of the grade and must be reported); and debugging
depth when something misbehaves at iteration 60.

Raw SDK + Pydantic for structured outputs. The SDK's tool runner handles the
call loop.

### The ledger is the memory; chat history is not

Measured 2026-08-27: one agent iteration cost 237k input / 15k output = $1.63,
and **input was 94% of it**. Not the prompt — a fresh turn is only ~4k tokens.
The cost was *replay*: every tool round resends the whole conversation, and one
iteration was running four experiments, so history grew to ~25k and each of ~16
API calls resent all of it. Quadratic.

Three changes, and the reasoning behind each:

- **Count cached tokens.** OpenAI serves repeated prefixes at a tenth of the
  input price. `TokenTracker` read only `prompt_tokens`, so it billed cache hits
  at full rate. Since total spend is a reported deliverable, that was
  *misreporting*, not just pessimism. Measured hit rate: **67%**.
- **Stop re-fetching what the agent was handed.** The per-iteration message
  already pastes the ledger and the best solution; the prompt then told the
  model to call `read_ledger` and `read_solution` for the same content. Each
  call is a whole extra round trip that resends everything — the round trips
  cost more than the content did.
- **Compact history to the last iteration.** Older turns are duplicates: the
  next message is rebuilt with a current ledger and current best solution.
  `_compact()` cuts only at `user` boundaries so a `tool` message can never be
  orphaned from the assistant turn that requested it.

```
                  before      after
per experiment    $0.409      $0.138       3.0x
API calls/iter    ~4-16       3
```

A 40-experiment autonomous run now costs **~$5.50**, so the cost cap can be set
where it is a backstop rather than the thing that ends the run.

**The near-miss worth recording.** Dropping `read_ledger` looked free. It was
not: `LEDGER.md` renders one merged `primary` column, while the JSON records
carry **GAUC and nDCG@5 separately** — and that split is where the best result
came from. Iteration 5's hypothesis reads *"BPR-2neg gives the best primary via
GAUC but loses a little nDCG@5; add a small BCE auxiliary."* Unreachable from a
merged number.

Fixed by generating the prompt's table from the JSON records instead of pasting
`LEDGER.md` (`prompt.py:_ledger_table`). Same size — 382 tokens vs 373 — and
strictly more signal than the agent had before the optimisation. **A token
optimisation that removes a metric the agent reasons over is a capability cut
wearing a cost-saving label.**

### valid vs test

The agent sees **valid only**; the harness refuses `--split test`.

Test isn't cryptographically hidden — it's in the downloaded file with labels.
It's hidden by discipline. Selecting on it makes the number meaningless: with
σ = 0.0008, the max of 100 draws sits ~2.5σ ≈ +0.002 above the mean by chance,
which is exactly the threshold for "a real improvement."

Check test **at most two or three times for the whole competition**, at
milestones, never feeding a decision. Its purpose is to confirm valid gains
transferred — MLE-bench lists validation overfitting as a known way these runs
fail, and test sits 10 days further from training than valid does.

---

## Guardrails, all tested

| | mechanism |
| --- | --- |
| Test set unreachable | harness picks the split; `--split test` refused |
| Duplicate work | source hash; returns the prior result, runs nothing |
| Syntax errors | `ast.parse` before execution, reports the line number |
| Crashes | caught; `stderr_tail` returned so the agent can act on it |
| Hangs | 15 min timeout (a correct run is ~30 s) |
| Bad output | wrong row count / NaN / Inf rejected before scoring |
| Noise mistaken for progress | verdict uses the official ε = 0.002, in code not prompt |
| Missing verdict | assigned inside `ledger.write()`, so an early return can't skip it |

| No-op mistaken for a result | identical GAUC *and* nDCG@5 to 6 dp ⇒ `no-op`, excluded from convergence |
| Agent escaping `solutions/` | `write_solution` and `read_solution` resolve the realpath and refuse |
| Overwriting a logged experiment | `write_solution` refuses an existing filename |
| Runaway spend | `MAX_EXPERIMENTS` / `MAX_COST_USD`, checked before every experiment |

The write jail closes the gap this file previously listed as open: without it
the agent could edit `harness/` and every guardrail above becomes a suggestion.

---

## Robustness — where the 35% actually lands

The rubric puts Robustness inside **Technical Execution (35%)**, next to the
primary metric, and defines it narrowly:

> Not judged by whether the agent ever hits a failure, but by **how it handles
> one** — recovering, retrying, or routing around a failed step so that long
> iterative runs neither crash, stall, nor diverge.

§2.4 adds a logging requirement that is easy to miss: each iteration must record
*"any error / recovery events."*

**Handled.** Solution crashes (`stderr_tail` returned, agent self-corrects),
hangs (900 s timeout), wrong row count / NaN / Inf, syntax errors caught before
execution, tool exceptions returned as JSON, malformed tool arguments, rate
limits (sleep 30 and retry), duplicate and no-op experiments, and stall
(`converged()` on the official ε/N rule).

**Three gaps, all closed 2026-08-27.**

**API failures are now classified and retried.** `_classify_error` checks fatal
patterns *before* retryable ones, because an auth failure can mention
"connection" in its message and must not be treated as transient. Unknown errors
default to retryable: a wasted call costs pennies, an abandoned run costs the
night. Backoff is exponential with jitter — identical sleeps mean every retry
collides again on a shared rate limit — capped at 60 s, five attempts.

```
transient 500 x2   3 calls, recovered, logged
503 forever        6 calls, gave up, iteration abandoned, RUN CONTINUES
invalid_api_key    1 call,  RUN STOPS
```

That last row was a gap the test exposed rather than one predicted: a fatal
error previously abandoned one iteration and the outer loop then reproduced it
on all 39 remaining ones. Retries also no longer consume `tool_rounds`.

**Recovery events are logged to two places.** `logs/events.jsonl` is the
chronological run log, append-only, one JSON per line — the only place a failure
*between* experiments can live, since it touches no experiment record. The same
events are also folded into the iteration record as `recovery_events`, so a
judge reading one iteration sees what happened during it without
cross-referencing timestamps. An empty list is meaningful: it says the iteration
was clean.

Kinds: `run_start`, `api_retry`, `api_recovered`, `api_gave_up`, `api_fatal`,
`budget_stop`, `converged`, `interrupted`, `crash`, `corrupt_record`, `run_end`.

The loop is wrapped so **how the run ended is always recorded** — Ctrl-C logs
`interrupted`, an unhandled exception logs `crash` with the traceback and then
re-raises, and both still emit `run_end`. A run log that simply stops is
indistinguishable from one whose terminal was closed; this one always says
which. That distinction is the whole autonomy claim.

**Ledger writes are atomic.** Temp file in the same directory, `fsync`,
`os.replace` — atomic on both Windows and POSIX. A crash mid-write now leaves
either the old complete file or the new one. `_load_all()` also stopped skipping
corrupt files silently: it logs `corrupt_record` once per file, so a vanished
experiment says so.

`logs/events.jsonl` is tracked in git for the same reason `logs/iterations/` is —
it is the evidence, not a byproduct.

---

## The autonomy run

The current ledger is **not** a demonstration of autonomy, and the records say
so themselves:

```
1   by=human   agent_iter=None   14:52:23   the control
2-5 by=agent   agent_iter=1      14:54-14:57  process A, one conversation
6   by=agent   agent_iter=None   14:58:58   annotation missing — killed mid-turn
                                 ↑ 4m23s gap: stopped, code edited
7-8 by=agent   agent_iter=1      15:03-15:04  process B — counter reset to 1
```

Three defects: it opens with a human action, there is human intervention inside
the loop, and neither process terminated on its own. Iteration 6's missing
`agent_iteration` is itself the evidence — annotation happens after a turn
completes, so a null means the process died first.

**What resets for the record run: the ledger.** `LEDGER.md` and
`logs/iterations/` archive to `logs/shakedown-02/`. **What does not reset: the
code.** The harness, agent, prompt and `001_torch_fm.py` are the artifact being
demonstrated; freezing them is the point.

```
1. commit and tag              proves the code was frozen
2. archive the ledger          logs/shakedown-02/
3. run the baseline by hand    iteration 1, by=human — the control
4. launch once                 python -m agent --max-iter 40
5. do not touch it
6. it stops itself             converged(), not a budget cap
7. score on test               once, at the end
```

Step 3 stays human-run: a baseline the agent reproduced itself is a weaker
control than one verified against the organisers' published number.

**Voids the run:** editing agent/harness/prompt code mid-run, killing and
restarting, hand-editing the ledger. **Does not:** an API outage or machine
crash, *provided it is documented* — infrastructure failure is not the agent
failing, and hiding the gap is worse than explaining it.

Aim for `converged()` to be what stops it. A run that ends on the official ε/N
rule is a stronger claim than one that ends on a budget cap, because it ends the
way the specification says it should. At $0.138/experiment, set
`AGENT_MAX_COST_USD=15` so cost cannot be the terminating condition.

---

## The CWM code in `src/`

Demoted. The organisers explicitly say it is *"not recommended as a starting
point"* — it optimises counterfactual watch time and evaluates on a `long_view2`
it reconstructs itself.

**But keep it.** It's ported to a modern stack and reproduces the paper (AUC
0.7357 vs 0.735), which the kit says most teams won't manage because
`torch==1.6.0` won't install. That makes direction 4 (censored-regression watch
time) comparatively cheap for us — a card to play later, not a starting point.

**Risk:** the repo contains a complete, coherent, *working* implementation of a
task that no longer exists — including a scorer that returns confident numbers
for the wrong metrics. That's more dangerous than noise, because it looks
authoritative. Mitigation is mechanical: the agent's write access is
`solutions/` only, and `read_state` never surfaces `src/`.

---

## Next

Ordered by what blocks the record run.

1. **Confirm the best solution on 2-3 seeds.** +0.0025 against σ=0.0008 clears
   2σ by a hair, on one seed. Everything downstream assumes this is real, and
   iterations 8-12 are now five consecutive softmax-weight tweaks spanning
   0.0003 — which is what a seed sweep exists to tell apart from progress.
2. **Decide whether search should ever fire.** `web_search` has never been
   called — not once in 12 experiments. Nothing triggers it: the prompt says
   *"search ONLY when going beyond"* the 7 directions, and the agent was handed
   those 7 ranked with #1 flagged as most likely. It has no reason to look.
   Either make it fire (e.g. "before starting a new direction, search for how it
   is usually implemented") or drop the tool. A dormant tool that appears in the
   schema every call is paying rent for nothing.
3. **Record run** — the protocol above. Budget two or three attempts; the first
   clean run usually fails on something never seen in development. Do not
   schedule it for the last night.

After that, and only after: the agent has spent all 12 experiments inside
direction 1. Directions 2 (sequences) and 3 (multi-task) are untouched, and the
softmax variants are already grinding at 0.0003 spread.

---

## Open

- ~~Is `log_random` trainable?~~ **Answered 2026-08-27: no.** Training on it
  means training on the evaluation period. Analysis and unbiased-evaluation
  experiments are fine; it must not fit the submitted model.
- ~~API key~~ — obtained; `.env`, gitignored. Model `gpt-5.5`.
- **Search provenance is unenforced.** The only rule is a prompt line asking the
  agent to put the citation URL in its hypothesis. Nothing logs the query or the
  result, and history compaction now drops the tool message after one iteration
  — so an uncited finding evaporates. Log searches into the record if search is
  kept at all.
- **Branch strategy** — the agent will commit on every iteration. Two humans
  plus an agent on `main` will collide.
- **`_budget_check()` counts every ledger row, not this session's.** Fine for a
  fresh-ledger record run; misleading if you ever want "20 more experiments" on
  top of an existing ledger.
