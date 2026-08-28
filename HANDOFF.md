# Handoff

State, decisions, and what's next. **`CLAUDE.md` holds the task facts** — label,
metrics, splits, baselines, dead ends. This file holds the *decisions* and the
reasoning behind them, which is the part that's expensive to reconstruct.

Last updated: 2026-08-28, code frozen at tag `record-run-4-code`, run 4 not yet launched.

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
✅ measurement: 3-seed scoring, spread shown to the agent
✅ convergence: fires correctly, floored at 8 experiments
✅ search policy: draft / improve / debug, refine before pivot
✅ submission path built and validated against the official checker
✅ **record run 3 (Zheng) — converged at +0.0040**
✅ compute caps aligned to the organisers' 50-iteration / 6 h rules
⬜ **score on test — once. Never done. See Next.**
⬜ write the submission
```

**Record run 3 is the result.** One uninterrupted process, no human
intervention, terminated on the organisers' convergence rule.

```
submission candidate   logs/record-run-3/solutions/027_deepfm_member.py
valid primary          0.605493   GAUC 0.672469  nDCG@5 0.538518
delta vs baseline      +0.0040                  target was +0.0020
stopped                converged, -0.000055 across the last 3
cost                   $3.65, 356 min compute, 30 iterations
```

Node 24 (`024_time_watch_userbalanced.py`) ties it at 0.605492 — the two are
indistinguishable. Either is defensible as the submission; #27 is the recorded
best by six decimal places.

Spend across every run to date: ~3.1M in / 235k out, ~$8.2.

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
| `harness/seedsweep.py` | measure one solution across seeds; writes nothing to the ledger |
| `harness/watch.py` | follow a long unattended run; one line per experiment and event |
| `harness/make_submission.py` | solution -> validated submission CSV. **The only tool allowed to touch test.** |
| `logs/<run>-N/` | every run auto-creates its own folder (Zheng, `ledger.init_run_dir`) |
| tags `*-code` | one per run, immutable, pointing at the code that produced it. **No `record-run-3-code`** — run 3 started 20:15 and the run-folder commit landed 23:54, so the exact code it ran is not cleanly identifiable. Left untagged rather than mislabelled. |
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

### The five runs

Each archive is kept because it is the evidence behind a fix, and deleting them
would leave the fixes looking like guesses.

| | what it was | what it produced |
| --- | --- | --- |
| `logs/shakedown-01/` | 9 experiments, first debugging run | 4 bugs in the agent loop |
| `logs/shakedown-02/` | 12 experiments, single-seed | 8 faults; see its README |
| `logs/void-run-1/` | 3 experiments, stopped deliberately | the convergence floor |
| `logs/record-run-1/` | 8 experiments, converged | the search policy (depth-2 star) |
| `logs/record-run-2/` | 9 experiments, converged at +0.0031 | the search policy working |
| **`logs/record-run-3/`** | **30 experiments, converged at +0.0040** | **the result** |

**shakedown-02** is the one worth reading. Twelve experiments, all inside
direction 1, ending in five consecutive tweaks of one loss weight across a
0.0003 spread — while that model's own seed noise is 0.0006. Its ladder was
noise, and its headline +0.0025 was seed 0 being lucky against a true +0.0019.
Its one genuinely informative row is iteration 7 (hard-negative mining,
-0.0034), the only time the agent proposed a *mechanism* rather than a
parameter — which is the behaviour Innovation (20%) is scored on.

**void-run-1** ran for six minutes and is the most useful of the three per
minute spent. It caught `converged()` about to end a run at three experiments,
and it confirmed three earlier fixes working live: `web_search` fired on
iteration 1 after never firing once in 13 previous experiments; multi-seed
scoring rendered BPR as +0.0014 against a 0.0004 spread where single-seed would
have shown a bare 0.603113; and the interrupt handler logged `interrupted` then
`run_end` instead of the log simply stopping.

**record-run-1** converged legitimately but its whole tree was `2,2,2,2,2,2` —
six first drafts off one node, nothing refined. That produced the draft / improve
/ debug policy.

**record-run-2** is the first run whose tree is a tree:

```
1 → 2 → 3 ─┬─ 4
           ├─ 5
           └─ 6 → 7 → 8 → 9
```

**record-run-3** (Zheng, 2026-08-28) is the deepest search yet: 30 experiments,
5 of the 7 directions, 10 self-directed web searches, and a genuine debug chain
at `14 → 15 → 16 → 17` where a broken raw-CSV alignment crashed to 0.5829 and
was diagnosed back to 0.6051 across three attempts.

It is also the first run where **direction 4 (watch time) paid off**: nodes
22-24 took it from +0.0034 to +0.0040 using play_time/duration as a *training
confidence weight*. Checked for leakage and clean — `play_meta['train']` only;
the valid and test arrays are computed and never used. Prediction uses IDs and
`hourmin`, which is known at impression time.

One thing to tidy before submission: the raw-CSV join key includes the label
(`lookup[(date, user, video, tab, dur, y)]`). It is defensible — it disambiguates
which duplicate impression a row is, using a property of a row you already hold —
but the feature-construction path touches `y`, which reads badly under scrutiny.
Joining on file order instead would remove the question.

### What is actually demonstrated

**BPR blended with pointwise BCE, plus temporal context.** Plain BPR gives
+0.0018; blending a calibrated pointwise term gives +0.0021; averaging two
independent BPR components +0.0024; adding hour/day features +0.0031. The
ensemble also has the tightest seed spread of any agent solution (0.00024 against
a typical 0.0005-0.0009), which is what averaging should do.

**Listwise softmax is genuinely wrong here, not a bug.** Three independently
written implementations across three runs: -0.0051, -0.0026, -0.0056. A full
softmax over a user's impressions assumes exactly one relevant item, but this
data is 33% positive, so a user's positives compete against each other.

**Time features are weak on their own and better on top of an ensemble.**
record-run-1 found nothing from them on plain BPR; record-run-2 got +0.0007 at
~1.4σ on top of the ensemble. Suggestive, not established.

**Raw watch time as a ranking target is actively harmful** (-0.0375,
record-run-1), because `long_view` is watch time *relative to duration* — ranking
by raw play time favours long videos. But watch time as a **training confidence
weight** works (+0.0006, record-run-3 nodes 22-24). Same signal, opposite result,
depending on whether it is the target or the weight.

**Sequences and multi-task have never clearly paid.** record-run-1 found nothing
from either; record-run-3's multi-task nodes (18, 20, 21) landed inside their
spreads.

### Where the gains came from, and where they stopped

```
#1  -> #2    +0.0015   BPR
#2  -> #8    +0.0008   3-member ensemble
#8  -> #10   +0.0008   more members
#10 -> #17   +0.0006   time features
#17 -> #24   +0.0004   watch-time confidence
#24 -> #30   +0.0000   six experiments, nothing
```

Halving at every step, then flat. **Adding ensemble members is measurably
exhausted** — six consecutive experiments at 0.6054-0.6055.

Two structural facts bound what is left. `nDCG@5` is at **77.3% of its ceiling**
(0.5385 of 0.6968) while GAUC is only 34.5% of the way from random to perfect —
so GAUC is the half with room, and it is also the half every BPR variant already
targets. And **60.2% of users have all their valid impressions in one tab**, so
tab is constant within those users and cannot affect their ranking at all; that
is why node 29's same-tab BPR gained nothing despite the 44-point rate spread
across tabs.

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

## Seed noise — the ladder is smaller than it looks

Measured 2026-08-27, three seeds each, scored outside the ledger (a seed sweep
is one experiment measured properly, not several experiments):

```
012_softmax2_bce0025_fm.py        001_torch_fm.py (baseline)
  seed 0   0.603999                 seed 0   0.601400
  seed 1   0.603210                 seed 1   0.601574
  seed 2   0.602734                 seed 2   0.601266
  mean     0.603314  std 0.000639   mean     0.601414  std 0.000154
```

**The real improvement is +0.0019, not the +0.0025 in the ledger.** Per seed:
+0.002599 / +0.001636 / +0.001468. Paired across three seeds it is genuinely
significant (t ≈ 5.4, p ≈ 0.03) — the gain is *real* — but its size was
inflated by seed 0 being the luckiest draw, and seed 0 is the only seed the
agent ever sees, because the harness hardcodes it. So the headline sits
**marginally below ε**, not above it.

**The more useful number is the std: 0.000639 for 012 against 0.000154 for the
baseline — 4x noisier.** Sampled softmax over sampled negatives adds two
stochastic layers the pointwise baseline does not have. This is a property of
the *method*, not of the run.

That settles iterations 8-12. Their spread was 0.0003 against a per-seed noise
of 0.0006, so **those five experiments were measuring nothing but seed noise** —
not "probably noise", measurably so. The ladder 0.603910 → 0.603999 is an
artefact of one random draw.

Which reframes the grinding: it was never mainly a prompt problem. **The agent
chased 0.0003 differences because the harness reported them as differences.**
Telling it not to grind treats the symptom; the cause is that a single-seed
score cannot resolve what it was being asked to compare.

**Open decision — should the harness average over seeds?**

| | cost | effect |
| --- | --- | --- |
| leave it | — | matches the organisers' design; agent keeps chasing noise |
| 3 seeds per experiment | 40 s → 2 min; 80 experiments ≈ 2.7 h | every verdict real; fixes grinding at the root, and makes `converged()` honest |
| single seed, verify the winner | ~2 min once | what was done here; the agent's *path* stays noise-driven |

Not a bug fix, a design choice. Single-seed development *is* the organisers'
intent: they set ε ≈ 2.5σ specifically so a single-seed gain of ε is unlikely to
be noise. But that calibration assumed the baseline's σ of 0.0008; our best
solution's is 0.0006 on its own and the *difference* of two noisy runs is
noisier still, so ε is doing less work here than intended.

---

## Convergence — the rule that decides when a run is over

ε = 0.002, N = 3, both organiser-fixed (Starter Kit README and §2.3 of the
problem statement). Not ours to tune.

**`converged()` could not fire, and this was only found by asking when a run
stops.** It borrowed `verdict()`, which answers a different question:

| | compares against | used for |
| --- | --- | --- |
| `verdict()` | fixed 0.6015 | the ledger's `KEPT` accept gate |
| `converged()` | **running best** | has the search stopped progressing |

The two agree only while the agent is below target. Once past 0.6035 every
result is `KEPT` forever — including one that repeats its parent exactly — so
convergence could never trigger. Iterations 10-12 improved the best by
**0.000089**, a twentieth of ε, and all three were labelled `KEPT`.

`converged()` now compares the best of the last N scored experiments against the
best of everything before them. `verdict()` is unchanged; the `KEPT` label was
never wrong, it was just answering a different question.

**What does not count toward convergence:** anything that failed to score.
Errors, crashes, timeouts and no-ops are skipped, so an agent debugging a new
direction is not penalised for it — verified in practice at iteration 13, which
crashed and cost nothing. This is the answer to "but some ideas need 5-10
iterations": the debugging ones are free; only *working but unimproving*
experiments burn the window.

**The strategic consequence, which is the part worth internalising.** Grinding
costs twice: it burns the three-iteration window *and* raises the bar any new
direction must then clear. Iterations 2-8 each improved, so the window kept
resetting and there was unlimited room. Iterations 9-12 gained 0.0003 in total
and left the agent needing +0.002 over 0.6040 within three tries. **Expensive
directions must be tried while still climbing, not after the cheap ones are
exhausted.**

Which is why the prompt now states the rule, plus the line that matters — *a
flat result is a signal to change direction, not to change a constant* — and
every iteration carries a live status line:

```
**Convergence watch.** Best improvement across the last 3 experiments:
**+0.000089**, against the 0.002 needed to keep the run alive (BELOW
THRESHOLD - one more experiment without a real gain ends the run, so try
a different direction).
```

**Verified, 2026-08-27.** One iteration against the converged ledger, with only
the loop's stop check bypassed. The agent pivoted from direction 1 to direction
6 unprompted — *"The loss direction has saturated, so switch to time drift
features"* — its solution crashed, and it diagnosed itself correctly:
*"`load()` returns split tuples/lists, not raw DataFrames ... This is an
implementation failure, not evidence against time features."* $0.19.

That is three behaviours confirmed at once: it pivots when told it is stuck, a
crash costs it nothing, and it does not record its own bug as evidence about a
technique.

### The floor — and the second bug, caught one experiment before it fired

Making `converged()` work exposed the opposite failure. It could now end a run
*too early*. Caught live in void-run-1, with #4 already training:

```
#1 0.601413 (control)   #2 0.602909 (BPR)   #3 0.596445 (listwise)

best of [#1]        = 0.601413
best of [#2,#3,#4]  = max(0.602909, 0.596445, X)
converged unless      X >= 0.603413
```

BPR's +0.0015 was the best improvement so far and sits under ε, so anything
below 0.6034 at #4 would have ended the run with **three experiments and two
ideas tried**. #4 was a first attempt at time features and had no realistic
chance of clearing it.

Formally the rule was satisfied. But it exists to detect a **plateau**, and
three experiments is a start, not a plateau. The organisers fix ε and N and say
nothing about a minimum run length — presumably because a run ending at
experiment 3 never came up.

`MIN_SCORED_BEFORE_CONVERGENCE = 8`: convergence cannot fire until eight scored
experiments exist, leaving five to establish a best before any three are judged
against it. **A floor on when a plateau may be claimed, not a loosening of the
threshold** — ε and N are untouched. Tested against void-run-1's actual
sequence and both sides of the boundary at 7 and 8.

The general lesson, which cost two runs to learn: **a stopping rule has two
failure modes and fixing one exposes the other.** The first version could never
stop; the second could stop before anything had been tried.

### The rule cannot tell steady progress from being stuck

Both record runs converged at **exactly 8 scored experiments** — the first moment
the floor allowed. In record-run-2 the agent was still improving when it was
stopped:

```
#7  +0.0021        window  #7,#8,#9  best 0.604598
#8  +0.0031        before  #1..#6    best 0.603623
#9  +0.0028                          improvement +0.000975  (need 0.002)
```

Three consecutive real gains, and the rule declared a plateau, because it only
counts single steps of at least ε. Refinement almost never produces a single
+0.002 jump; it produces a sequence of smaller ones. So the rule structurally
penalises the behaviour that Innovation rewards.

This is not ours to fix — ε and N are organiser-fixed. It is the substance of
`docs/email-convergence-question.md`, which asks whether refinement iterations
are meant to count toward N at all. Note the answer could go against us: if they
say the rule applies from the first opportunity, our 8-experiment floor has to
go.

**Backstops raised to 80 experiments / $15.** Neither should ever fire. What
they actually guard is a crash loop: `converged()` counts only *scored*
experiments, so an agent whose solutions all fail never converges.
`MAX_EXPERIMENTS` counts every ledger row, errors included, and is the only
thing bounding that case.

---

## Logging error and recovery events

`recovery_events` was populated only by `_event()`, which was wired exclusively
into the API-error paths. A *solution* crash never reached it — so every record
in record-run-2 read `recovery_events: []`, including the one that crashed, and
`events.jsonl` showed no trace of the run's single best robustness moment.

Nothing was lost: the error, traceback, diagnosis and causal link were all in the
iteration records (`#2 status: error` + `stderr_tail`, `#3 parent: 2` + a
`debug 2` hypothesis). It was a rendering gap, not a data gap — which is why the
backfill was possible.

`harness/run.py` now emits two events itself rather than relying on the loop:

- `solution_error` when an experiment fails to score, with the traceback
- `solution_recovered` when an experiment scores while its parent did not

record-run-2's log was backfilled from its records, with those entries flagged
`backfilled: true` so the reconstruction is never mistaken for a live capture.

**The general point:** a log built to prove one thing was silently failing to
record the very thing it existed for, and an empty field read as "nothing went
wrong" rather than "this was never wired up". Absence of an event is only
evidence if something was actually watching.

---

## The autonomy run

**The protocol, run three times now.** The ledger holds only the control between
runs; each frozen state has its own tag.

```
1. commit and tag              DONE - one tag per run, see below
2. archive the ledger          DONE - shakedown-02, void-run-1
3. run the baseline by hand    DONE - iteration 1, by=human, 0.601413 +/- 0.000154
4. launch once                 python -m agent          <- no flags
5. do not touch it
6. it stops itself             converged(), not a budget cap
7. score on test               once, at the end
```

**No `--max-iter`.** The default 100 sits above `MAX_EXPERIMENTS=80`; passing 40
would make the loop cap bind first and the experiment backstop dead code. They
are also not interchangeable — an iteration abandoned by an API failure burns a
loop pass but writes no ledger row, so `max_iter` must sit *above*
`MAX_EXPERIMENTS`, never below.

Expect ~1.5-2 h and ~$3-6. Experiments are ~85 s now (3 seeds), not ~28 s.

Step 3 stays human-run: a baseline the agent reproduced itself is a weaker
control than one verified against the organisers' published number. Its
`+/- 0.000154` is also the reference point for reading every later row — that is
what a *stable* model looks like here.

**One tag per run, never reused.** Each frozen state gets its own immutable tag
named for the run it produced, pairing with the archive directory:

```
void-run-1-code    531789f   ->  logs/void-run-1/
record-run-1-code  5d9eb77   ->  logs/record-run-1/
record-run-2-code  d0fdb97   ->  logs/record-run-2/
```

There was previously a single `autonomy-run-1` that got deleted and re-cut at a
new commit before each attempt. That breaks the moment anyone else has fetched
it — their tag disagrees with the remote and git refuses to pick a winner, so
`git pull` fails with a tag conflict. It also made the tag useless as a record,
since "the frozen code" meant three different commits depending on when you
looked. If you hit that error on an old clone:
`git fetch --tags --prune --prune-tags`.

Treat a tag as immutable. Name the next one after the run it will produce.

**Voids the run:** editing agent/harness/prompt code mid-run, killing and
restarting, hand-editing the ledger. **Does not:** an API outage or machine
crash, *provided it is documented* — infrastructure failure is not the agent
failing, and hiding the gap is worse than explaining it.

Aim for `converged()` to be what stops it. A run that ends on the official ε/N
rule is a stronger claim than one that ends on a budget cap.

### Why the first attempt was voided, and what that cost

Six minutes, and worth it. `converged()` was one experiment from ending the run
at three (see the convergence section). Stopping was the right call: watching it
happen would have taught nothing that the arithmetic did not already say, and
the run would have needed relaunching regardless.

Two process lessons, both mine:

- **Do not run `git add -A` while the agent is writing.** It swept the agent's
  in-flight `002_bpr_fm.py` and `events.jsonl` into a commit about something
  else. Harmless — git does not modify working files, and nothing the running
  process had imported changed — but it muddies the history at exactly the point
  where a clean history is the evidence.
- **Verify before believing a taint.** When the run turned out to have been
  launched before a commit, the question was answerable in one command:
  `git diff --stat autonomy-run-1 HEAD -- agent/ harness/`. The only difference
  was the agent's own output. A running Python process holds its modules in
  memory anyway, so a file edited on disk cannot reach it.

### Following a long run

`python harness/watch.py` emits one line per experiment and per notable event,
and exits when the run logs `run_end`. Rows whose delta is smaller than their
seed spread are marked `[< spread]` at the point of emission rather than left as
a tidy-looking number to be misread later — that confusion *is* shakedown-02.

It emits failures as loudly as successes on purpose. A watcher that prints only
good news is silent through a crash loop, and silence looks exactly like "still
running".

---

## Compute limits — now organiser rules, not our backstops

The problem statement of **2026-08-27** replaced `Compute budget: TBD` with a
hard specification:

> **50 iterations per benchmark run** (hard cap; the convergence rule
> ε = 0.002 / N = 3 normally triggers first), plus a **6 h wall-clock ceiling**
> per run as a backstop.

Two of our three caps are therefore compliance limits now, not choices:

| | value | whose rule |
| --- | --- | --- |
| `MAX_EXPERIMENTS` | **50** (was 80) | organisers |
| `MAX_WALL_SECONDS` | **6 h** (new) | organisers |
| `MAX_COST_USD` | **off** (0) | ours — disabled |

**record-run-3 would have been non-compliant.** It ran 20:15 → 02:44 — 6 h 29 m
— and nothing was watching the clock. It also used 30 of the 50 iterations, so
only the wall-clock rule was breached.

### Seeds run concurrently — the run is 2x faster

record-run-3 was **91% training compute** (356 min of 389 min wall clock), and
the three seeds of each experiment ran one after another while torch used only 6
of the machine's 12 cores. The seeds are independent by construction — same
code, different RNG — so that was throwing away half the machine.

`_run_seeds()` now launches all seeds at once and divides the cores between
them (`OMP_NUM_THREADS = cpu_count // n_seeds`). Dividing the threads is the
part that matters: three processes each grabbing half the cores would thrash.

```
sequential, 6 threads    99.9 s     0.601413 +/- 0.000154
concurrent, 4 threads    49.9 s     0.601413 +/- 0.000154
```

**Identical numbers, half the time.** On record-run-3's workload that is roughly
6 h 29 m -> 3 h 20 m, comfortably inside the 6 h ceiling.

Failure semantics are unchanged and were re-tested: one seed failing still fails
the whole experiment rather than averaging the survivors, the error names which
seed, the traceback survives, `solution_error` is logged, and a `finally` block
kills siblings so a crash cannot leave orphan processes.

**What this does not buy is a better score.** record-run-3 stopped on
`converged`, not on the clock — it used 30 of the 50 allowed iterations. Speed
buys compliance and headroom for more expensive experiments; it does not make
the search find more.

### The wall-clock stop reserves time rather than stopping at the line

The ceiling applies to the **run**, so an experiment that starts at 5 h 55 m and
takes 12 minutes still breaches it. `_budget_check()` therefore stops when
`elapsed + longest_experiment_so_far + 2 min` would cross the ceiling, using the
slowest experiment already logged rather than a fixed margin:

```python
longest = max(r['seconds'] for r in ledger._load_all()) + 120
if elapsed + longest >= MAX_WALL_SECONDS: stop
```

A fixed margin would be wrong here. record-run-3's experiments averaged ~12 min
because the agent kept adding ensemble members, and its slowest were far longer
— the cost per experiment grows during a run, so the reserve has to grow with
it.

### Token spend is reported, not capped

The same update also pinned how Feasibility is scored:

> scored only among submissions whose hidden-test primary score **exceeds the
> official baseline**, and graded in **three coarse tiers** (low / medium /
> high consumption) rather than a continuous ranking.

Two consequences. Cost only counts **if you beat the baseline** — so the primary
metric comes first and there is no point trading score for cheapness. And
because it is tiered, the difference between $1.63 and $3.65 is almost certainly
invisible; both are "low" against a 6 h / 50-iteration allowance.

The organisers say why: *"Without the quality gate the criterion would fight the
Primary metric — an agent that stopped after three iterations would look
cheapest and score worst."*

**So the dollar cap is now OFF by default** (`AGENT_MAX_COST_USD=0`). It could
only ever hurt: killing a compliant run part-way forfeits the score that *gates*
the Feasibility criterion, in exchange for a saving nobody measures. The run is
already bounded twice by the organisers' own limits — 50 experiments and 6 h —
which at record-run-3's rate is about $6.

Cost is still tracked and reported in full; only the cap is gone. Set
`AGENT_MAX_COST_USD` to a positive number to re-enable it.

**Do not optimise tokens further.** It buys nothing at the tier granularity and
risks the score the criterion depends on.

---

## Making a submission

`kuairand-starter-kit/submit.py` ships with the kit and had never been run until
after record-run-2. Solutions emit `.npy` arrays; a submission is a
`row_id,user_id,video_id,score` CSV, and the official `--check` rejects a wrong
header, a wrong row count, gaps in `row_id`, misaligned `(user_id, video_id)`
pairs, and non-finite scores. Nothing bridged the two, which is the classic way
to lose on the last night.

```
python harness/make_submission.py 008_time_features_two_bpr_bce.py \
    --split valid --out submission_valid.csv          # sanity, scores locally
python harness/make_submission.py 008_time_features_two_bpr_bce.py \
    --split test  --out submission.csv --seeds 1      # the real thing
```

It writes through the official `write_submission` and validates through the
official `read_submission`, so the file is checked by the organisers' code
rather than ours. It resolves a solution by name in `solutions/` or in any
`logs/*/solutions/`, so an archived candidate works without being copied back.

**It is the one thing in the repo that may touch `test`**, which is why it lives
outside `solutions/`, is absent from `TOOL_DISPATCH`, and leaves `run.py` still
refusing any split but valid. Scoring on test stays a decision a person makes,
at most two or three times for the whole competition.

### Archived solutions were unrunnable, and now are not

Solutions locate the starter kit relative to their own file
(`../kuairand-starter-kit`). That stops resolving the moment a run is archived
to `logs/<run>/solutions/`, so **every archived solution was silently
unrunnable**, including the submission candidate — the archives were not
reproducible, which is most of the point of keeping them. Both
`make_submission.py` and `run.py` now put the kit on `PYTHONPATH`, which grants
a solution no access it did not already have. Worth remembering if solutions are
ever moved again.

### The decision waiting for whoever submits: seed 0, or an average?

Measured on valid, using record-run-2's best solution:

```
seed 0 alone              0.604761
3 seeds, rank-averaged    0.605133   (+0.0004)
```

`--seeds N` rank-averages predictions across N seeds. Ranks rather than raw
scores, because different seeds put their logits on different scales and one
wide-ranged run would otherwise dominate the mean; only relative order is
scored, so ranks lose nothing.

**This is a decision about what the submission claims, not a formality.** The
agent's solution is itself a 3-component ensemble (two BPR + one BCE).
Rank-averaging three seeds of it submits a nine-component ensemble that *a human
constructed*, not one the agent ever proposed or validated.

- **Submit seed 0** — the artifact is entirely the agent's work. This is the
  recommendation. The +0.0004 is about 1σ of a single seed, and Autonomy is
  scored under Impact & Relevance (20%); trading "this is entirely the agent's"
  for a gain that small is a poor deal.
- **Submit the average** — defensible, and worth roughly 12% more delta
  (+0.0031 → +0.0035 against the Primary metric under Technical Execution, 35%).
  But only if it is **disclosed in the writeup** as a human-added seed ensemble.
  Presenting it as the agent's output would be false.

Note the two numbers are not directly comparable as measurements: 0.604761 is a
single draw with a std of ~0.0005, while the rank-average has essentially no
variance. The fair comparison is against the agent's 3-seed *mean of metrics*,
0.604598, which makes the ensemble's edge about +0.0005.

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

Ordered. Item 1 blocks the run; items 3-5 happen while it runs.

### 1. Benchmark the second machine before choosing where to run

Two candidates, and the paper specs do not settle it:

```
AMD Ryzen 5 5600X          6 cores / 12 threads, desktop, 3.7 GHz sustained
Intel Core Ultra 5 125H   14 cores / 18 threads, laptop, hybrid P + E cores
```

The 125H's fourteen "cores" are not fourteen of the same thing — 4 P-cores,
8 E-cores, 2 low-power E-cores — and torch will schedule onto the slow ones.
More importantly this is a **multi-hour all-core load**, and a laptop throttles
where a desktop holds its clocks. A short benchmark will not show that.

**Run this on the Intel machine, twice, back to back:**

```
python harness/seedsweep.py 001_torch_fm.py
```

**Reference, measured on the 5600X: 84.8 s.** Note `seedsweep.py` runs its three
seeds *sequentially*, unlike the harness — that is deliberate here, because a
sequential benchmark compares single-core-ish throughput rather than how well
each machine schedules three concurrent processes. Do not compare it to the
harness's ~50 s parallel figure; compare Intel-sequential against
5600X-sequential, i.e. against 84.8 s.

- **Slower than ~85 s** ⇒ the desktop is faster per unit work; prefer it.
- **Faster than ~85 s** ⇒ the Intel machine wins on raw speed, and its 14 cores
  should also parallelise the three seeds at least as well.
- **Second run meaningfully slower than the first** ⇒ thermal throttling. That
  is the decisive signal: it compounds over a 3 h load, and a laptop that starts
  fast can finish slower than the desktop. Prefer the desktop in that case even
  if the first run looked good.

`seedsweep.py` writes nothing to the ledger, so this is safe to run at any time,
including while something else is going on.

### 2. Launch record run 4

```
python -m agent --run-name record-run
```

No other flags. Auto-creates `logs/record-run-4/` and copies the control row in.

**Estimate on the 5600X: 2-4 hours.** record-run-3 was 356 min of compute plus
33 min of LLM; parallel seeds halve the compute, and the new standalone-before-
blending policy should make individual experiments much cheaper (1 model x 3
seeds instead of 10 x 3). It may also run more of them — cheaper experiments
delay convergence. The 6 h guard stops it either way.

**Pre-committed, before the run starts so it cannot become a rationalisation:**
run 4 replaces `record-run-3/solutions/027_deepfm_member.py` as the submission
candidate **only if it exceeds 0.605493 by at least 0.0005**. That solution's
seed spread is 0.0002, so the SE of a 3-seed mean is ~0.0001 and of the
difference ~0.00016 — the threshold is about three standard errors. Below it,
027 stands and we say so.

### 3. Rewrite the README — the biggest visible gap

It is still the **upstream CWM paper's readme**. A judge opening the repo sees
"Counterfactual Watch Model" and a paper abstract. The deliverables require:

```
project overview          setup and installation
steps to reproduce        limitations and what you would improve
team member contributions
```

The material exists — this file for the reasoning, `logs/record-run-*/` for the
evidence, and the ceiling analysis for a genuinely good limitations section.

### 4. Compile the resource-usage report

Deliverable 4 asks for total tokens (in + out), total agent wall-clock, and
iterations used out of 50. `ledger.totals()` has the first; `run_start` and
`run_end` in `events.jsonl` bracket the second.

**Report within-run interventions separately from development.** An unattended
run has **zero within-run interventions** — that is the number the autonomy
criterion asks for. Prompt and harness revisions between runs are development,
listed separately and honestly.

### 5. Generate and validate the submission, then score test ONCE

Note the wording. `make_submission.py` **generates and validates** a submission;
it does not score test, by choice. The test labels *are* in the downloaded file —
test is hidden by discipline, not by cryptography — so scoring locally is
possible and is a decision, not a capability.

**Agreed plan:** lock the candidate by the ≥0.0005 rule first, then score test
locally **once**, purely to learn whether the gain transfers. After the candidate
is fixed, so it cannot influence selection.

```
python harness/make_submission.py <best>.py --split test --out submission.csv --seeds 1
```

Read *"The decision waiting for whoever submits"* above and choose `--seeds 1`
or `--seeds 3` first.

Expect **~0.598 on test**: valid has run ~0.007 above test throughout. Compare
the *delta* against the official FM's test **0.5946**, never the raw valid
number. A delta far below +0.0040 would mean the gains were valid-specific,
which is worth knowing — the candidate was selected across 30 experiments on
valid alone.

### 6. Answer from the organisers

`docs/email-convergence-question.md` is sent. Question 2 — whether refinement
iterations count toward N — decides whether any further run should refine.

---

## If you want a higher score

**Not by hand, and not by naming a method in the prompt.** This is an autonomy
track: Innovation (20%) is scored on *what the agent identified as worth trying*,
so a method we supply is a result we produced, not one it found. The legitimate
lever is the agent's **capability** — what it can do and how it searches — never
its knowledge.

The agent finds methods on its own: ESMM, DIN, CWM and DeepFM all came from its
own web searches in record-run-3, and it found CWM with no access to `src/`,
which contains a working implementation.

Realistic reach is **0.607-0.608**; **0.610 needs a genuinely new source of
signal** and is unlikely given the deceleration above.

**Do not run repeatedly and submit the best.** That is selection on validation
across runs — it makes the final number less credible and under-reports actual
resource use. Re-running after *changing the agent* is legitimate and is what
run 4 is; re-running the same agent to fish for a good draw is not.

---

## Open

- ~~Is `log_random` trainable?~~ **Answered 2026-08-27: no.** Training on it
  means training on the evaluation period. Analysis and unbiased-evaluation
  experiments are fine; it must not fit the submitted model.
- ~~API key~~ — obtained; `.env`, gitignored. Model `gpt-5.5`.
- ~~Search provenance~~ — searches now log query and result to
  `logs/events.jsonl`, so a finding survives history compaction whether or not
  the agent cites it.
- ~~Directions 2-7 unproven~~ — record-run-1 tested six of the seven; only the
  loss change (direction 1) and, weakly, time features (6) do anything.
- **The +0.0007 attributed to time features is ~1.4σ.** The absolute level of
  record-run-2's best is solid; that specific increment is not established, and
  record-run-1 found time features gave nothing on plain BPR. If a further run
  reproduces it on top of an ensemble, that is an interaction worth stating; one
  observation is not.
- **The search policy has been exercised exactly once.** record-run-2's tree is
  the shape we wanted, but a single run is a single sample of the agent's
  behaviour, not evidence that the policy reliably produces it.
- **A solution already in the ledger cannot be re-measured in place.** The
  source-hash guard returns `duplicate`. Fine for a fresh ledger; use
  `harness/seedsweep.py` (which writes nothing) to re-measure anything old.
- **Branch strategy** — the agent will commit on every iteration. Two humans
  plus an agent on `main` will collide.
- **`_budget_check()` counts every ledger row, not this session's.** Fine for a
  fresh-ledger record run; misleading if you ever want "20 more experiments" on
  top of an existing ledger.
- **Test has never been scored.** Everything claimed here is validation-only.
  Until that one run happens, transfer is an assumption.
- **The submission artifact does not exist yet.** The tooling is built and
  validated on valid; nobody has produced `submission.csv`.
- **Spinner rendering is only fixed for width.** If the terminal is *resized*
  mid-run the padding is recomputed each frame, so it self-corrects, but a
  narrower window mid-write can still leave one stale row.
- **Prompt hygiene:** the agent's search queries have included the phrase
  "citation URL", leaked from the prompt line asking it to cite. Harmless, but it
  dilutes the query.
