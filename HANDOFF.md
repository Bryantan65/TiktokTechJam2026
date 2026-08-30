# Handoff

State, decisions, and what's next. **`CLAUDE.md` holds the task facts** — label,
metrics, splits, baselines, dead ends. This file holds the *decisions* and the
reasoning behind them, which is the part that's expensive to reconstruct.

Last updated: 2026-08-29, after record run 8 converged. Code frozen at tag
`record-run-9-code`. **Run 3 remains the submission candidate.**

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
✅ **record run 4 — converged at +0.0031, did not displace run 3**
✅ **record run 6 — converged at +0.0035, did not displace run 3**
✅ GBDT capability disclosed; agent found LambdaRank unprompted and it lost
✅ README rewritten from the upstream CWM readme to the project's own
✅ the agent can see what an experiment costs (`secs`)
✅ an unmeasured seed spread is no longer reported as a stable one
✅ **train-only holdout: the agent can screen an idea without spending valid**
✅ **test scored once — +0.0039. The gain transferred.**
✅ submission generated from the locked candidate and validated
```

**Record run 3 is the result.** One uninterrupted process, no human
intervention, terminated on the organisers' convergence rule. Run 4 was launched
to test whether the revised search policy would beat it; it did not, and the
pre-committed rule below settles that it does not replace it.

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

Spend across every run to date: ~7.8M in / 605k out, ~$16.8. Run 6 added $4.28
across 131 API calls with 72% of its input tokens served from cache.

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

### The runs

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
| `logs/record-run-4/` | 32 experiments, converged at +0.0031 | the search policy under test; run 3 held |
| `logs/record-run-5/` | 28 experiments, **interrupted** at +0.0039 | Zheng's; not a valid autonomy run |
| `logs/record-run-6/` | 34 experiments, converged at +0.0035 | the GBDT capability test; run 3 held |
| `logs/record-run-7/` | 3 experiments, **crashed** | a UnicodeEncodeError on an arrow; fixed |
| `logs/record-run-8/` | 31 experiments, converged at +0.0033 | the plateau detector; run 3 held |
| `logs/record-run-9/` | 33 experiments, converged, best 0.605738 | Zheng's; the headline is seed-picked, see below |
| `logs/record-run-10/` | 31 experiments, converged, best 0.604931 | the LightGBM/LambdaMART family; run 3 held |
| `logs/record-run-11/` | 30 experiments, converged, best 0.604653 | semi-hard sampling and LambdaRank weighting; run 3 held |
| `logs/record-run-12/` | 34 experiments, converged, best 0.605216 | Zheng's, on the cleaned prompt; run 3 held |
| `logs/record-run-13/` | 32 experiments, converged, best 0.605484 | first run on the broadened web_search; run 3 held |
| `logs/record-run-15-bryan-pure/` | 30 experiments, converged, best 0.605499 | 30/30 scored, no failures; run 3 held |

**Reading a ledger programmatically: filter on `verdict`.** Run 12's highest
`primary` across its JSON records is 0.605885 at experiment 5, which is *not* a
valid score - its verdict is `screen`, meaning it ran against the train-only dev
holdout. Its own follow-up on valid scored 0.6007. Taking the max over all
records without checking `verdict` overstates that run by +0.0007 and invents a
new best that never existed. Runs 9, 10 and 11 happen to be unaffected because
their top records are `KEPT`.

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

**record-run-4** (2026-08-28, 5600X) is the first run under the revised search
policy — standalone-before-blending, plus the environment disclosure. It is the
cleanest search of the six and it scored **lower** than run 3. Both facts matter.

```
best      #31  031_user_trend_position.py   0.604615 +/- 0.000141   +0.0031
          GAUC 0.6715   nDCG@5 0.5377
records   32 (30 scored, 2 no-op)   of the 50-iteration cap
wall      12:10:54 -> 14:30:07 = 2 h 19 m   of the 6 h ceiling
compute   110 min solution time - 128 API calls - 7 web searches - $4.24
stopped   converged, +0.000084 across the last 3
```

**It did not displace run 3.** The pre-committed rule required exceeding
0.605493 by >= 0.0005, i.e. 0.605993. It reached 0.604615 — short by 0.0014, and
below run 3 outright. Run 3 stands, and this is recorded because the rule was
written down before the run started precisely so it could not be renegotiated
afterwards.

**The tree is the deepest we have produced.** Twelve levels along its spine, with
breadth at the nodes that earned it:

```
1 -> 2 -> 3 -> 5 -> 7 -> 9 -> 11 -> 16 -> 19 -> 25 -> 31 -> 32
                    |            |     |
                    +- 5 kids    +- 4  +- 4
```

Compare record-run-1's `2,2,2,2,2,2` — six first drafts and nothing refined. The
policy change did what it was written to do.

**It found a different mechanism family and arrived at a lower ceiling.** Run 3's
spine was a ten-member ensemble; run 4 stayed close to single models and built on
**sampled softmax with 8-16 negatives drawn from the same user**, then layered
history statistics and a row-order user-trend term. Two independent searches,
different families, +0.0040 and +0.0031 — and the ~0.0009 between them is roughly
what run 3's ensembling was worth.

That is the uncomfortable reading of the policy change: it made the search
legible and made the score worse. Both runs are honest samples, but standalone-
first discourages exactly the multi-model averaging that produced run 3's best,
and the rule as written is too strict — see the wording fix under *Next*.

**Listwise softmax failed for the fifth time.** `#4`, a full softmax over each
user's impressions, scored 0.59659 — **-0.0049**. That is now five independent
implementations across four runs that could not see each other's ledgers:
-0.0026, -0.0051, -0.0056, -0.0020, -0.0049. The mechanism is understood (33%
positive rate means a user's positives compete against each other) and the
replication is about as clean as this setup can produce.

**Two no-ops caught and recovered from.** `#18` and `#21` produced identical
GAUC and nDCG@5 to an earlier node despite different code — the source-hash and
identical-metrics guards flagged both, the agent read the flag and moved on, and
`#22` scored 0.604073 immediately after `#21` failed. One `solution_recovered`
event in `logs/record-run-4/events.jsonl`. **Zero crashes, zero API failures,
zero human interventions.**

**record-run-6** (2026-08-28, 5600X) is the capability test: `xgboost` and
`lightgbm` were added to the environment and every hint pointing at them was
removed from the prompt first. It converged on the rule, inside both caps.

```
best      #29  029_blend_fm_deepfm.py   0.605024 +/- 0.000129   +0.0035
          GAUC 0.6720   nDCG@5 0.5381
records   34 (30 scored, 2 failed, 2 no-op)   of the 50-iteration cap
wall      18:30:52 -> 22:04:19 = 3 h 33 m     of the 6 h ceiling
compute   180 min - 131 API calls - 8 web searches - $4.28
stopped   converged, -0.000233 across the last 3 (it went backwards)
failures  0 crashes, 3 recoveries, 0 human interventions
```

It needed 0.605993 to displace run 3 and fell **0.00097 short**. Run 3 stands.

### What run 6 was for, and what it answered

**Capability disclosure alone is sufficient — naming the method is not needed.**
The agent found LightGBM on its own at `#15`, from nothing but `lightgbm 4.x` in
the installed list, and cited the docs page it searched. This is the cleanest
autonomy evidence we have: we made a thing possible, it decided the thing was
worth trying.

**Direct nDCG optimisation loses here.** That was the hypothesis behind adding
the libraries at all.

```
#15  015_lgbm_lambdarank_hist    0.598312 +/- 0.000192   -0.0032   standalone
#23  023_lgbm_fm_rank_correction 0.603956 +/- 0.000185   +0.0025   as a rank correction on FM
```

The implementation is competent — `objective: lambdarank`, grouped by user,
`label_gain: [0, 1]` correct for binary relevance, native categoricals, history
features from the best branch. Its one weakness is `num_boost_round=260` fixed
with no early stopping. Both rows carry real error bars, so neither number is
noise. LambdaRank does not beat a bagged FM on this data.

**A fourth independent search landed in the same band.** Four runs, four
mechanism families, none able to read another's ledger:

```
record-run-3   0.605493   BPR ensemble + watch-time weighting
record-run-5   0.605368   mixed tab/hour ensembles       (interrupted)
record-run-6   0.605024   FM rank ensemble + DeepFM blend
record-run-4   0.604615   same-user sampled softmax
                          mean 0.605002, sd 0.000498
```

0.610 is **ten standard deviations** above that mean. It is not a matter of more
draws; the distribution has no mass there. Combined with the ceiling analysis
(non-personalised half-fitted ceiling 0.6048, pair coverage 1.62%) this is the
saturation case, and run 6 is the strongest single piece of it because the
method it tested was the one most likely to break through.

**Listwise softmax failed a sixth time**, `#3` at -0.0049. Six implementations
across five runs that cannot see each other's ledgers: -0.0026, -0.0051, -0.0056,
-0.0020, -0.0049, -0.0049.

**The best reasoning artifact of any run is `#29`'s hypothesis:**

> *"DeepFM raised GAUC but hurt nDCG@5 versus the FM rank ensemble, so blend it
> readably at 65% with the prior FM rank+margin ensemble to keep the high-order
> signal while recovering top-5 ordering."*

It read the two metrics separately, worked out which model was better at which,
and combined them at a readable weight. That is only possible because the ledger
shows GAUC and nDCG@5 as separate columns rather than the mean.

### Three harness defects run 6 exposed

**1. The 3-seed measurement can silently measure nothing.** Solutions like `016`
hard-code `seed_bag = [0, 1, 2]` internally and ignore the harness's `--seed`,
so all three seeds produce byte-identical predictions and the row reports
`+/- 0.000000`. Six consecutive bests (0.603731 -> 0.604500) carry no error bar
at all, and the steps between them are smaller than the seed noise of every model
in the run that *did* vary. We paid 3x the compute for one number and presented
it as maximally stable. **Fix: compare `predictions_hash` across seeds; if they
match, skip the remaining seeds and label the row `deterministic`.**

**2. The agent is blind to cost, and it cost an experiment.** Its ledger has no
time column. Experiment cost grew from 82 s (single model) to 526 s (8-member
bag) as it enlarged the ensemble — roughly +42 s per member, x3 seeds = 24 model
trainings per experiment. At `#31` it proposed adding same-user BPR pressure
inside DeepFM, a good idea for this metric, and died on the 15-minute timeout
with no way to know it was already at 60% of the limit. The prompt already tells
it *"a cheap decisive experiment beats an expensive ambiguous one"* — advice it
could not act on.

**Fixed 2026-08-28.** Note the data was never missing: `seconds` and
`wall_seconds` have been in every `NNNN.json` since the harness was written. The
defect was purely in rendering — `_ledger_table()` did not show it, so the agent
was blind to a cost we were already recording. It now renders a `secs` column,
plus two things the number alone does not convey: that 900 s is fatal, and that
an N-member ensemble costs N times a single model *every time*, including when
the thing being tested is not the members. It also says to evaluate several
combination rules inside one solution rather than one per experiment — run 6
spent ~67 minutes, a third of its compute, retraining 24 models repeatedly to
re-weight predictions it already had.

**3. The convergence floor cuts both ways.** We raised it to 30 to stop
premature stopping. Run 6 found DeepFM at `#28`-`#29` with three experiments
left, then converged. The floor that saved run 1 truncated run 6.

Also worth noting: experiments 19-26 re-trained all 24 models from scratch to
test *combination rules* over predictions that already existed — `power_rankavg`,
`rank_margin_blend`, `margin50_rank_blend`. Caching predictions between
experiments would remove that waste, but it breaks the property that any archived
solution re-runs standalone and reproduces its number, which is doing real work
for the submission's credibility. Not recommended.

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
4. launch once                 .venv/Scripts/python -m agent   <- no flags
5. do not touch it
6. it stops itself             converged(), not a budget cap
7. score on test               once, at the end
```

**Launch with the venv's interpreter, not whatever `python` resolves to.** The
harness spawns solutions with `sys.executable`, so they inherit the Python that
started the run, and `_resolve_device()` picks CUDA only if that Python's torch
reports it. The venv holds `torch 2.6.0+cu124`; a system Python may hold a
CPU-only build. Getting this wrong does not fail - it silently runs on the CPU
and the run takes about 40% longer, on a wall-clock that is a scored criterion.

```
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"   -> True
```

`HARNESS_DEVICE=cpu` forces the CPU path when you want it; `auto` is the default
and falls back to CPU on its own if torch is missing or has no CUDA.

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

**Resolved 2026-08-28: `--seeds 1`.** The organisers' Q&A (below) asked entrants
not to lean on human intervention because "the goal is to evaluate the autonomous
agent's capabilities". A human-built seed ensemble worth half a sigma is exactly
the trade that guidance argues against. Submit seed 0.

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

## The test score — the gain transferred

Scored once, after the candidate was locked by the pre-committed rule, so it
could not influence selection. `harness/score_test.py`, official `evaluate.py`.

```
                    GAUC      nDCG@5    primary   delta
valid   0.672469    0.538518            0.605493  +0.0040
test    0.665391    0.531626            0.598508  +0.0039
```

**+0.0040 on valid became +0.0039 on test.** That was the real risk in this whole
project and it did not materialise. The candidate was chosen as the best of 30
experiments *on validation alone*, and validation was also used inside every
experiment to pick the early-stopping epoch — two rounds of selection on the same
split. A large valid-to-test drop would have meant the ladder was selection
noise rather than modelling. It transferred essentially intact.

Note the raw numbers still show the expected ~0.007 valid-to-test offset
(0.6055 vs 0.5985). That offset is a property of the splits, not of our method —
the official FM shows the same thing (0.6016 valid, 0.5946 test). **Always
compare deltas, never raw primaries across splits.**

A second file, `submission.csv`, generated earlier the same day, rescored to
0.598411 (+0.0038). The 0.0001 gap is process-level float nondeterminism, and
confirms both files came from the same candidate.

---

## The train-only holdout — screening without spending valid

`harness/devdata.py`. Cuts the TRAIN window by date: earlier days to fit on, the
last 5 to score on. Never touches valid, and has no `test` key at all, so a
solution cannot reach for test data even by accident.

It is a faithful proxy, which was not obvious in advance:

```
                 rows      positive rate   (user, video) pair coverage
official valid   124909    33.2%           1.62%
dev holdout      129898    33.5%           1.71%
```

Pair coverage is the number that defines this task — it is why personalisation
is unlearnable here — and the holdout reproduces it to within 0.1 points.

**Why it exists.** The agent develops against valid, and valid also picks the
early-stopping epoch inside every experiment. That is two rounds of selection on
one set of labels, and the pressure grows with the length of the search. It has
held so far (+0.0040 valid → +0.0039 test) but every hunch currently costs a
real experiment. MLE-bench requires exactly this of every agent it evaluates:
*"each agent must rely solely on a self-constructed runtime test set, a held-out
split from the original training data."*

**How it is wired.** `run_experiment(..., split='dev')`. Such a row is logged
and visible, but:

- `verdict` is `screen`, not a delta — its number is on a different scale
  (the control scores 0.6081 on dev against 0.6014 on valid) and comparing it
  to the 0.6015 baseline would be meaningless
- excluded from `_scored()`, so it cannot end a run by convergence
- excluded from `best()`, so it cannot become the incumbent
- it still costs wall clock, which is the honest price of a screen

`solutions/001_torch_fm.py` handles `--split dev`, so everything the agent
branches from inherits the pattern. `run.py` puts `harness/` on the subprocess
PYTHONPATH and pins `HARNESS_DEV_HOLDOUT_DAYS`, so the harness and the solution
cannot derive different cuts and misalign their rows.

**What it deliberately does not do.** `describe()` reports row counts, date
ranges, positive rates, per-field types and generic user/video/pair coverage.
It reports nothing about any particular feature — no per-field lift, no ranking
of what looks promising. That would be a person's research findings smuggled
into a tool. The agent's own candidate code decides what gets tested.

The field-type report earns its place: the official loader returns ids as
**strings**, and mixing them with ints read from a CSV via pandas produces
lookups that miss on every row *without raising*. The feature reads as absent
everywhere and the experiment looks like a clean negative result. That cost
three wasted attempts during the feature audit on 2026-08-28.

**Guards, regression-tested:** `--split test`, `--split train` and any unknown
split are refused by the harness; the agent's tool schema offers only
`valid | dev` and its handler refuses anything else independently.

---

## record-run-8 - ahead at iteration 13, lost in the tail

Converged on the rule at 0.604793 (+0.0033), 31 experiments in 97 minutes,
$4.01. Run 3 stands. The interesting part is not the number but *where* it was
lost.

```
matched best-so-far
run       it3        it8        it13       it20       it30
run-3   0.602909   0.603725   0.604521   0.605092   0.605493
run-8   0.603625   0.603625   0.604659   0.604746   0.604793
                              ^ AHEAD                ^ 0.0007 behind
```

It cleared the +0.0020 bar at **iteration 3** - the fastest of any run, against
run 3's it8 and run 6's it16 - and was still ahead at it13. Then it stopped.
From it13 to it31 it gained **+0.00013** across 18 experiments; run 3 gained
+0.00097 over the same span.

The tail explains all of it:

```
it20-it31   12 experiments, every one a LightGBM-residual variant
range       0.604632 .. 0.604793  =  0.000161
typical +/- 0.000205
            the ENTIRE spread is 0.79x ONE error bar
compute     37 of 67 minutes - 55% of the run
```

Twelve experiments that are statistically the same number. It found a good node
at it17 and spent half the run generating variations it could not tell apart.

**This is record-run-6's failure in new clothes.** Run 6 burned nine
experiments on ensemble re-weighting for +0.00003; we answered that with the
batching guidance in the `secs` block, and run 8 invented a different way to do
the identical thing. Fixing one shape of the pathology moved it rather than
removing it.

### Two things it got right, both worth keeping

**It reached for LightGBM unprompted, for the second run running.** Nothing in
the prompt names it; only `lightgbm 4.x` in the installed list. It crashed on a
misplaced bracket (`os.path.dirname(path, '..')`), read the traceback, and
shipped a fix in the next iteration.

**It independently invented the residual-ranker design we deliberately withheld.**
Its it17 hypothesis: *"the independent LambdaRank ranker degraded the strong FM
ensemble; use the FM rank ensemble as LightGBM `init_score` and feature so
LambdaRank learns residual top-k corrections rather than replacing the incumbent
ranking."* That is the stage-one/stage-two plan proposed on 2026-08-28 and kept
out of the prompt on purpose. The agent got there from its own failed
standalone attempt. Had we injected it, this would have been our result.

It did miss the part of that plan that mattered most: **out-of-fold**. It used
the ensemble's in-fold scores as `init_score`, which leaks stage-one fit into
stage two, and is the likeliest reason a correctly-built mechanism bought
nothing. The train-only holdout exists to build a time-forward stage-one score
cheaply, and across 31 iterations it took **zero dev screens**.

### The fix: report the plateau, do not diagnose its cause

`_plateau_note()` in `agent/prompt.py`. The agent can already see every `+/-`,
and the table already warns that two results closer than their spread have not
been told apart - but it has to infer a plateau by eyeballing a dozen rows, and
across two runs it did not. So the harness now counts it and says so.

It is a statement about the agent's own numbers: that the branch is
*unmeasurable*, not that it is wrong, with no suggestion of what to try instead
beyond "change mechanism, or screen it on `dev` where being wrong is cheap".
That last clause is also the only targeted nudge toward the unused holdout, at
the one moment it is obviously useful.

**Threshold tuned against all five archived runs, not chosen.**

```
thr   run-3            run-4        run-5    run-6    run-8
4     it12 (+0.00097)  it10         it16     never    it23
5     it13 (+0.00097)  it11         never    never    it24
6     it29 (1 left)    it32 (0)     never    never    it25 (6 left)
```

At 4 or 5 it fires on record-run-3 at iteration 12-13 - a **false alarm on the
best run we have**, which went on to gain +0.00097 after that point. At 6 it
catches run 8 at it25 with six experiments still to spend, and everywhere else
either never fires or fires with at most one experiment left, where it cannot
do harm.

Two subtleties in the implementation, both found by testing rather than
reasoning:

- A row with no spread (`None` from a screened seed or a deterministic
  solution, or `0.0` in records written before deterministic runs were
  detected) has an **unknown** spread, not a zero one. Treating it as zero
  makes it impossible to call indistinguishable, which is how run 6's
  re-weightings escaped detection entirely. It now substitutes the median
  measured spread.
- Run 6 still does not fire, and that is correct. Its nine re-weightings sat
  0.0005-0.0008 *below* its best - genuinely distinguishable, just useless.
  That is a different failure and the batching guidance already addresses it.

---

## The task is data-limited, not method-limited

The most useful measurement of the project, and it explains every other result
in this file. Run on `record-run-3`'s winner, predicting the real `valid`, with
every arm fitting on a slice of `train` only - no validation row is used for
training anywhere in this experiment, and `test` is never loaded.

```
A    6 days, OLD     fit 04-09..04-14, stop 04-15..04-21    0.589287
C    6 days, RECENT  fit 04-15..04-20, stop 04-21           0.591171
B   12 days, RECENT  fit 04-09..04-20, stop 04-21           0.604831

recency   C - A = +0.00188    same volume, six days newer      2.4 sigma
volume    B - C = +0.01366    same end date, twice the data   17 sigma
```

**Doubling the training data is worth +0.0137. The agent's entire modelling
gain across eleven runs is +0.0040.** Recency matters too but seven times less,
which also means the 8-day gap between the end of `train` and the start of
`test` is not what is holding the score down.

### Why this explains everything else

Three independent measurements now say the same thing from different angles:

```
median user has 31 training rows, k=16       1.9 observations per parameter
changing k (8/16/24/32)                      spread inside seed noise
weight decay 1e-6 -> 1e-3                    monotonically WORSE, -0.0038 at 1e-3
```

The `l2` result is the sharp one. If those thinly-estimated user vectors were
memorising noise, shrinking them would help. Shrinking them hurts, monotonically.
So **the model is under-fit, not over-fit** - what it has learned from two
observations per dimension is real signal, thinly measured. It is starved, not
confused.

That is why seven independent searches across four mechanism families all landed
within 0.0009 of each other, and why six hyperparameter configurations moved the
score less than the random seed does. They were all competing over the last
thousandth of a signal whose first fifteen thousandths are set by how much data
the split provides.

### Why we still trained on `train` alone

Refitting the final model on train plus valid is ordinary competition practice
and nothing in the rules forbids it. We did not do it, and the reasoning is
worth recording so the choice does not look accidental.

A TikTok ML engineer said in the official Q&A: *"try not to touch validation
until it's time to test."* That is guidance from the organisers and it settles
the question on its own. Two further reasons point the same way. Once `valid` is
in training the only instrument left is `test`, so deciding "refit or not" would
mean selecting on the split we are judged by. And every claim in this file rests
on `valid` having stayed a clean instrument across eleven agent runs - that is
why +0.0040 on valid carried over to +0.0039 on test, and it is the reason our
numbers are worth reading at all.

The size of what we passed on is small in any case. Data volume matters
enormously in the range we measured, but its marginal value has largely
collapsed by 13 days: 6->12 days is worth +0.0023/day, 12->13 days only
+0.0005/day. Valid adds 11% more rows at the flat end of that curve.

## What was tested and found empty, 2026-08-29

Each of these was a plausible idea with a mechanism behind it. All were measured
rather than argued about, and all came back inside seed noise or worse.

| tried | result |
| --- | --- |
| embedding dimension k = 8 / 16 / 24 / 32 | spread 0.00105, scattered; less than one seed sigma |
| learning rate 0.0005 / 0.001 / 0.002 | both changes slightly worse |
| weight decay 1e-5 / 1e-4 / 1e-3 | monotonically worse; -0.0038 at 1e-3 |
| `video_features_statistic_pure.csv` (52 cols) | best column scores 0.5804 against item popularity's 0.5807 - identical |
| `is_rand` exposure debiasing | 0 of 1,141,112 training rows have it set |
| user-tag taste history | 78% coverage, GAUC 0.5216 - a coin flip within users |
| transductive features from the impression list | eval-window exposure 0.4973, position 0.4833; blending them in hurts monotonically |
| the 18-second label threshold | subsumed - 99.99% of valid videos appear in train, so `video_id` already encodes every static property |
| seed rank-averaging on the winner | +0.00001. The earlier +0.0005 was measured on record-run-2's single BPR model; run 3's winner is already a 10-member ensemble and has done that variance reduction |
| ApproxNDCG / NeuralNDCG | published as on-par with LambdaRank, which we measured at -0.0032. Three untried methods ruled out by one measurement we already had |
| cold-start literature | addresses unseen ITEMS; ours are 99.99% seen. Wrong problem shape |

### Optimisation quality and transfer, 2026-08-29 (later)

The under-fit diagnosis has two halves. Capacity and priors were already ruled
out (k flat, weight decay monotonically worse). These test the other half -
whether the model is being trained hard enough - and whether a denser label can
substitute for the rows we cannot have. Both come back empty, and between them
they close the question.

| tried | result |
| --- | --- |
| batch size 8192 -> 2048 | 0.605136, -0.00018 |
| patience 4 -> 12, epoch cap 40 -> 100 | 0.605300, -0.00002 |
| both batch and patience together | 0.605123, -0.00020 |
| pretrain 3 epochs on `is_click`, fine-tune on `long_view` | -0.00062 |
| pretrain 8 epochs on `is_click`, fine-tune on `long_view` | -0.01151 |

**The model is under-fit but converged.** More updates, smaller updates and more
patience all do nothing, so it is not stopping early and it is not starved of
optimisation - it has already extracted what the data supports.

#### Why no auxiliary label can help, in one number

`is_click` fires on 46.3% of rows against `long_view`'s 33.7%, and yields 37%
more BPR pairs per epoch. It is the densest auxiliary signal available. But BPR
only learns from users who have both a positive and a negative:

```
long_view   pairable users  24290   pairs/epoch  382579
is_click    pairable users  24406   pairs/epoch  524927
```

**116 extra users. 0.5%.** The additional supervision lands almost entirely on
users the model already trains on. Every other engagement column is sparser than
`is_click` and falls on the same users, so this rules out the family, not just
the instance.

The degradation is also monotonic in pretraining length, which is the mechanism
showing itself: the click geometry is not a coarse version of the target that
fine-tuning refines, it is a different optimum that fine-tuning has to walk back.
Pretraining alone scores 0.584 - real signal, wrong target. This is a different
result from the multi-task experiment, which optimised both jointly; sequential
transfer was the untested variant and it is now tested.

#### The users BPR never trains are not a weak spot

`make_user_pairs` silently drops any user without both a positive and a negative,
so unlike the pointwise baseline, those users' embeddings receive no gradient.
That sounded like a real gap - it is not:

```
discriminative valid users whose embedding BPR never trains   508 of 12929 (3.9%)
their share of GAUC weight                                    3.63%

                        rows     GAUC     nDCG@5
BPR-trained users     119266   0.6715     0.5438
BPR-untrained users     5643   0.6919     0.4639
same slice, item popularity only                 0.6432     0.4502
```

They score **higher** GAUC than the users BPR does train, and well above item
popularity on their own slice. The item-side terms carry them, which is what the
within-user-ranking argument predicts. Nothing to recover here.

### The ensemble is saturated on both axes, 2026-08-29

The winner blends 10 members, 7 of which are the same FM/BPR model at different
seeds. The obvious hypothesis was that those 7 are near-duplicates and their
slots are headroom - spend them on distinct objectives instead and the
diversity term pays. **That hypothesis is wrong**, and the measurement is worth
keeping precisely because it is counter-intuitive.

```
full 10-member ensemble            0.605344

7 seed members only                0.605152
3 structural members only          0.602609

1 seed + all 3 structural          0.603737   -0.00161
3 seed + all 3 structural          0.604684   -0.00066
5 seed + all 3 structural          0.605356   +0.00001
7 seed + all 3 structural          0.605344    baseline

pairwise rank correlation
  seed vs seed                     r = 0.9165
  seed vs structural               r = 0.8519
  structural vs structural         r = 0.7973
```

Seed members are **not** near-duplicates. At r = 0.9165 they are meaningfully
decorrelated, and they carry the ensemble: the seven of them alone score 0.605152
against the full blend's 0.605344, so all three structurally different members
together contribute **+0.0002**. Removing seed members costs real score, and you
need five of them before the curve flattens.

So the diversity term is real but it comes from **seed variance, not
architecture**, and it plateaus at five members. An eighth seed does nothing; a
fourth objective is worth a fraction of a thousandth. Both axes are done.

(This is a different axis from the earlier "+0.00001 from seed rank-averaging",
which averaged whole ensembles across outer seeds. Both saturated, separately.)

### What this means for the score

The ensemble was the last untested mechanism. With data volume established as
the binding constraint and capacity, regularisation, optimisation, auxiliary
labels and now ensembling all measured flat, there is no known source of
another +0.0005 that would survive the move to test.

Valid can still be pushed higher by **selection** - run-to-run sd is 0.00034, so
the maximum over enough runs drifts up on its own. We have the receipt on what
that buys: a seed-picked run beat ours by +0.0002 on valid and +0.00034 on test.
Selection fits the valid split's noise, not its signal.

The one measurement that was genuinely higher is the cross-run blend at 0.6063,
six independent runs' winners rank-averaged. That is not selection - independent
runs really are decorrelated (r = 0.89-0.96 across runs against 0.98-0.99 within
one) - but it consumes N times the compute budget, and whether it counts as one
submission is exactly the open question in `docs/email-multirun-question.md`.

> **Superseded 2026-08-30.** Those r values are *global* correlation, which is
> nearly unrelated to the within-user ordering we are scored on. Remeasured with
> within-user rank correlation the comparison reverses: within one run 0.8811,
> across runs 0.9119. The blend gain is real; "independent runs are decorrelated"
> is not the reason for it. See *The multi-agent question, settled by
> measurement* at the end of this file.

### The tab decomposition, and same-tab negative sampling, 2026-08-29

`long_view` rates run 4.2% on tab 0 to 48.9% on tab 4, and the organisers flagged
conditioning on `tab` as an open and reasonable modelling decision. Splitting the
winner's own valid predictions by whether a scored pair crosses a tab boundary:

```
                    share of scored pairs    model AUC
same-tab  pairs           73.4%               0.6198
cross-tab pairs           26.6%               0.8256

51.3% of discriminative valid users span more than one tab
```

**The model is already near-solved on cross-tab ordering.** A single tab bias term
wins those pairs, because the base rates are so far apart. The overall GAUC of
0.672 is a blend of easy cross-tab pairs and the same-tab pairs it is much worse
at - and same-tab is three quarters of everything the metric scores. This is the
most useful structural fact we have about where the remaining error lives.

That looked like shortcut learning: uniform within-user negative sampling hands
the model ~27% easy pairs, so it can lower BPR loss through the tab bias without
learning within-surface affinity. The fix would be to draw negatives from the
same tab at rate p, making the tab term worthless.

```
p_same   primary              same-tab AUC   cross-tab AUC
0.00     0.602635             0.6150         0.8245
0.50     0.602517  -0.00012   0.6155         0.8251
0.75     0.603214  +0.00058   0.6159         0.8227
1.00     0.592396  -0.01024   0.6136         0.7699
```

`p=1.00` confirms the mechanism exists in the other direction: with no pair ever
distinguishing tabs the bias goes untrained and cross-tab AUC collapses.

But the shortcut theory is wrong. Same-tab AUC barely moves (0.6150 -> 0.6159)
even when the model is forced onto those pairs, so same-tab discrimination is
data-limited, not shortcut-limited - the same conclusion everything else reaches.

#### The +0.00058 was one lucky seed

Worth recording in full, because a single seed said the idea worked:

```
seed 0      p=0.00 0.602635   p=0.75 0.603217   diff +0.00058
seed 1009   p=0.00 0.603244   p=0.75 0.602253   diff -0.00099
seed 2027   p=0.00 0.603746   p=0.75 0.603187   diff -0.00056
seed 3037   p=0.00 0.602873   p=0.75 0.601186   diff -0.00169
seed 4051   p=0.00 0.603050   p=0.75 0.603042   diff -0.00001

paired diff  mean -0.00053   sd 0.00087   t = -1.36 on 4 df
```

Seed 0 was the only positive of five. Reporting the first result would have
published a gain that does not exist. Every number in this file that is quoted
as an improvement should be read against this: single-seed differences below
about 0.0008 are indistinguishable from nothing, and the only defence is
replication.

### Two model families converge to the same number

The strongest single argument that the ceiling is real rather than a failure of
imagination. Across runs 10 and 11 the agent ran **26 gradient-boosted tree
experiments** - LightGBM, LambdaMART, target encoding - unprompted, from nothing
but `lightgbm 4.x` in the installed package list.

```
LightGBM / LambdaMART, best     GAUC 0.6714   nDCG@5 0.5379   ~ 0.6046
FM/BPR ensemble (submission)    GAUC 0.6723   nDCG@5 0.5384   ~ 0.6053
```

Neural embeddings and boosted trees share no assumptions and fail in different
ways. They land **within 0.0007 of each other**. One family plateauing means the
search got stuck; two unrelated families arriving at the same place means the
signal in this data runs out around there.

Pure LightGBM with target encoding was clearly worse (GAUC 0.6573). The good
tree numbers are all blends where a tree model is one member among FM rankers.

### The agent's other objective experiments

Recorded for completeness - each was found by the agent without being named in
the prompt, and each lost to plain within-user BPR.

| tried | result |
| --- | --- |
| xendcg | on par, not better; appears in run 10's blends |
| semi-hard / score-based hard negative mining | 0.6012-0.6033, below the incumbent |
| multi-negative BPR (several negatives per positive) | roughly neutral |
| explicit user-video / user-author / user-tab cross fields | 0.6005, weak |
| listwise softmax over a user's impressions | competitive, not better |

### Zheng's 0.6057 is seed-picked

Worth recording precisely, because it was briefly treated as a target to beat.

`logs/record-run-9/solutions/028_seed0_history_tiebreak.py` accepts a `seed`
argument and then passes a literal `seed=0` to the model. The reported 0.6057 is
therefore one seed, not a mean.

```
honest 3-seed value    0.605353 +/- 0.000393
iterations 27-33 add   +0.00004

test    ours 0.598508    his 0.598847    gap 0.00034
```

The valid gap of +0.0002 became +0.00034 on test. This is the same trap the
same-tab sampling arm walked into on this side - a single seed said an idea
worked, and replication said otherwise. It is not misconduct, it is what
single-seed reporting does, and it is the reason every claim in this file that
matters is quoted with a seed count.

### Deep user history does not make taste learnable, 2026-08-29

The data-limited finding pointed at one escape: more history per user. The user
side is the underdetermined half - video vectors get 190 observations each
(1.44M rows / 7,551 videos), user vectors get 52 (1.44M / 27,285 users), which is
~3 per embedding dimension.

KuaiRand-1k has the SAME per-user depth as 27k (11,713 vs 11,812 interactions per
user) for 963 of Pure's users, so the mechanism is testable without the 9.21GB
27k download. We built per-user tag-affinity profiles from 1k's 04-08..04-21 file
only, added them as a bucketed field crossed with each Pure video's tag, and
split the valid score by whether the user had that deep history.

```
                   COVERED users GAUC        other users GAUC
control            0.67223 +/- 0.00427       0.66952 +/- 0.00058
tagtaste           0.67295 +/- 0.00163       0.66953 +/- 0.00085
paired diff        +0.00073, t = 0.33        +0.00001
```

**11,700 interactions of history moves a user's GAUC by +0.0007, t = 0.33.**
Scaled to 27k - every user covered - that is roughly +0.0004 on primary, half a
seed-noise unit for 9.21GB and five hours. **27k is not worth downloading.**

The detail that explains it: the deep-history users already scored *higher* than
everyone else before the feature was added (0.67223 vs 0.66952). There was no
deficit to fill. That is the third time this pattern has appeared - BPR-untrained
users also scored above average.

Caveat, stated because it bounds the claim: this is one feature design, and
Pure's 7,583 videos span only 44 distinct tags, which is a coarse taste signal.
Finer attributes (author, music, duration) were not tested. The claim is that the
most natural attribute-taste design gives nothing at 27k depth, not that 27k is
provably useless.

### GroupCE: cross-user ranking loss actively hurts, 2026-08-29

"Hierarchical Group-wise Ranking Framework for Recommendation Models"
(arXiv 2506.12756, AdKDD'25) quantises users into a hierarchy of clusters and
computes a listwise cross-entropy loss *within each cluster* rather than within
each user, so a sparse user borrows ranking signal from similar users. Reported
on KuaiRand: GAUC 0.6911 -> 0.6953 overall, and 0.6718 -> 0.6786 on cold-start
users - which is exactly the population our whole ceiling is made of.

Implemented as BPR plus a two-level group ListCE term (16 coarse clusters, 128
after residual quantisation, k-means on the trained user embeddings):

```
control        0.603282 +/- 0.000435
groupce l=0.3  0.597839 +/- 0.000310   -0.00544   t = -52.09 on 4 df
groupce l=1.0  0.594972 +/- 0.000384   -0.00831   t = -41.45 on 4 df
```

Every seed, both weights, monotonic in the weight. **Statistically the cleanest
result in this file, and clearly negative.**

The mechanism is independent of the implementation. GAUC is computed per user and
averaged; nDCG@5 is per user. Nothing in the metric rewards making one user's
scores comparable to another's, so every gradient spent on cross-user ordering is
spent against the objective. Their gain is measured against pointwise LogLoss,
which has no within-user structure at all - we replaced that in run 1 with
something tighter than what they add on top.

One implementation caveat: the group term takes its own optimiser step per group
batch rather than being summed into the BPR loss, so its effective weight exceeds
the nominal lambda. That is not a faithful reproduction. It does not change the
reading - an effect of -0.0054 is far too large to be an artifact of weighting,
and the direction is monotonic.

**This closes the last outstanding lead.** Four separate published methods have
now been ruled out by the same argument: they fix something our setup does not
have wrong.

### Which runs are clean autonomy evidence, and which is not

On 2026-08-29 at 11:13 and 12:18 two commits added our own findings to the
agent's instructions: a "Lessons from prior runs" section naming five results
with their deltas (*"Do NOT try listwise softmax again"*), and four extra
directions beyond the organisers' seven, one naming the exact CSV columns to
read and one describing group-wise ranking. It was reverted at 13:35 onto
`run13-clean-prompt`, but that branch was never merged, so `main` carried the
injected prompt until commit `c085354`.

**Every run started before 11:13 is clean.**

```
record-run-1    Aug 27 16:48    clean
record-run-2    Aug 27 17:38    clean
record-run-3    Aug 27 20:15    clean   <- the submission
record-run-4    Aug 28 12:10    clean
record-run-5    Aug 28 12:31    clean
record-run-6    Aug 28 18:30    clean
record-run-7    Aug 29 00:35    clean
record-run-8    Aug 29 01:04    clean
record-run-9    Aug 28 23:25    clean
record-run-10   Aug 29 02:48    clean
record-run-11   Aug 29 07:11    clean
record-run-12   Aug 29 13:14    CONTAMINATED - started 21 min before the revert
```

**record-run-12 is not autonomy evidence.** Its own ledger names the injected
directions: iteration 31 reads *"draft 8: train a readable 30% IPS-weighted BPR
content-FM member"* - direction 8 was one of the four added that morning - and
seven of its solutions are `content_*`, which is direction 9. It remains a valid
*modelling* run and its scores are real; it is simply not evidence that the
agent found those directions by itself.

Everything the submission's case rests on was measured before the injection
landed: eleven runs converging on BPR, seven listwise-softmax implementations
independently rejected across six runs, two unrelated model families landing
within 0.0007, and LightGBM reached for unprompted in runs 6, 10 and 11.

**The lesson is about merge hygiene, not vigilance.** The injection was spotted
within about an hour and reverted the same day. It survived because the revert
lived on a branch nobody merged, and because nobody re-read `agent/prompt.py`
while committing to `main` around it. A prompt is code that decides what a run
proves, and it deserves the same review as the harness.

### Two structural facts worth keeping

```
valid impressions whose video appears in train    99.99%
valid impressions whose (user, video) pair does    1.62%
same (user, video) seen twice in valid - labels agree   94.7%
```

The label is highly predictable from who-and-what. The problem is that you
almost never get to observe that pair before being asked about it. Content
features are redundant because the item is always known; personalisation is
unlearnable because the pairing almost never is.

### The ceiling, honestly estimated

Earlier versions of this file quoted a "perfect video knowledge" ceiling of
0.6197 on valid. That number was memorisation: it fit each video's rate on the
same rows it scored, roughly 21 per video. Cross-validated properly - fit on
train plus 90% of valid, score the held-out 10% - it collapses:

```
                        self-fitted   cross-validated
video rate                0.6146          0.5834
(video, tab) rate         0.6351          0.5919
(user, video) rate        0.8477          0.4988
```

The `(user, video)` collapse from 0.85 to 0.50 is the proof - that number was
the label leaking through one-row cells.

**Our submission beats perfect knowledge of any single feature by +0.014.** The
realistic ceiling is ~0.607 on valid, ~0.600 on test: seven runs span
0.6046-0.6055 with sd 0.00034, and blending six of them - the highest anything
reached - gives 0.6063.

---

## Organiser Q&A, 2026-08-28 — what it settles

A live Q&A session with the organisers. Four points bear on decisions we had
open; the rest was background on the recommendation funnel, which only confirms
what `CLAUDE.md` already says — our task is the **ranking** stage, not retrieval.

**A crashed run may be restarted, and that is not human intervention.** Their
wording: restarting a process that died on a network or other error is fine,
including from a separate session. A restart counts as intervention *only if you
change the agent's behaviour or parameters while doing it*.

This is more permissive than the rule we had been holding ourselves to, which
treated "killing and restarting" as voiding the run. Our design already meets
the condition and does so structurally rather than by discipline: the agent has
no memory between iterations — **the ledger is the memory** — so a restart
re-reads `logs/<run>/*.json` and resumes knowing everything. There is no
parameter to accidentally change because nothing is carried in process state.

It changes what we claim, slightly. Not *we never had to restart*, but *restarts
were permitted and we never needed one*: runs 3 and 4 both report zero crashes
and zero API failures.

**Do not lean on human intervention.** Asked whether an autonomous agent with
mediocre scores beats an accurate one needing human help, they said performance
gained by intervention tends not to survive the move to test, and that entrants
should focus on train/valid and on the agent's own capability. This is the same
line we drew ourselves — capability disclosure yes, knowledge injection no — and
it is what settles `--seeds 1` above.

**There is no validation leaderboard.** Entrants calibrate against their own
metrics and the published baseline, nothing else. So the ~0.007 valid-to-test gap
stays an estimate until the single local test check, which makes that check the
only ground truth we will have before submitting.

**The video is optional for Track 2** — recommended at ~3 minutes, but the other
four tracks require it and this one does not. If we skip it the written report
carries everything, which raises the bar on the README.

Also confirmed: free API keys from other providers are allowed if tokens run out.
Not currently a constraint at ~$8 total.

---

## Next

Items 1-3 are done. What remains is the submission itself.

### Done

**1. Benchmarked both machines.** `python harness/seedsweep.py 001_torch_fm.py`,
three sequential seeds: **5600X 84.8 s, Core Ultra 5 125H 280 s** — 3.3x slower,
with no throttling between back-to-back runs, so it is raw throughput and not
heat. The 125H's fourteen "cores" are 4 P + 8 E + 2 LP-E, and torch schedules
onto the slow ones. Run 4 went to the desktop, and any future run should too.

**2. Ran record run 4.** 2 h 19 m, 32 experiments, converged at +0.0031. It did
not displace run 3 under the pre-committed >= 0.0005 rule. Full write-up above.

**3. Rewrote the README.** Overview, guardrails table, setup, reproduction steps,
findings, limitations with the ceiling analysis, and per-member contributions.
The upstream CWM paper abstract is gone.

Also checked and closed: **`starterkitv2/` is byte-identical to
`kuairand-starter-kit/`** across all seven files — hash-compared, the only
difference is our own `README.en.md` translation living in the original folder.
Nothing to re-score, nothing to re-run. Delete the duplicate.

### 4. Compile the resource-usage report — open

Deliverable 4 asks for total tokens (in + out), total agent wall-clock, and
iterations used out of 50. `ledger.totals()` has the first; `run_start` and
`run_end` in `events.jsonl` bracket the second.

**Report within-run interventions separately from development.** An unattended
run has **zero within-run interventions** — that is the number the autonomy
criterion asks for. Prompt and harness revisions between runs are development,
listed separately and honestly.

### 5. Generate and validate the submission, then score test ONCE — open

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

**`--seeds 1` is settled** — see the resolution under *"The decision waiting for
whoever submits"*, and the organiser Q&A that decided it.

Expect **~0.598 on test**: valid has run ~0.007 above test throughout. Compare
the *delta* against the official FM's test **0.5946**, never the raw valid
number. A delta far below +0.0040 would mean the gains were valid-specific,
which is worth knowing — the candidate was selected across 30 experiments on
valid alone.

### 6. Answer from the organisers — partially in

`docs/email-convergence-question.md` is sent and unanswered. The 2026-08-28 Q&A
session answered other things (see that section) but not this. Question 2 —
whether refinement iterations count toward N — decides whether any further run
should refine.

### 7. ~~Loosen the ensembling rule before any run 5~~ — done (kept as-is)

The ensembling guidance already says *"test a new mechanism so its signal is
readable"* and *"never blend below 20%"*, which is the right balance. The
standalone-before-blending wording from run 4 was not added to the prompt for
run 5 — the existing guidance is sufficient.

### 8. Tidy run 3's raw-CSV join before submitting — open only if run 3 ships

`023`/`024` key the raw-log lookup on `(date, user, video, tab, dur, y)` — the
label is in the join key. It is defensible (it disambiguates which duplicate
impression a row is, from a property of a row you already hold) and it is not
leakage, but a feature path that touches `y` reads badly under scrutiny. Joining
on file order removes the question entirely.

### 9. Run 5 prompt and capability changes — done 2026-08-28

Runs 2-4 all plateaued at 0.604-0.606. Analysis of all four runs showed:

- The agent reaches ~0.604 in 7-10 experiments (BPR ensemble), then grinds
- It hand-writes every model from memory — LambdaRank, LightGBM rankers, and
  other library-backed approaches were unreachable
- It never tried direct nDCG optimisation, learned ensemble weights, per-tab
  calibration, or post-hoc rescaling
- It retries listwise softmax in every new run because runs don't share memory

The ceiling is a **capability problem**, not a memory problem. Cross-run memory
would only help us during development — the organisers run the agent once from
cold, so the deliverable must work in a single run.

Two changes to `agent/prompt.py`, both expanding general capability:

1. **New "Search strategy" section** — four bullets of general ML guidance:
   try ambitious directions early; direct metric optimisation beats proxy losses;
   post-hoc calibration is cheap; learned ensemble weights beat equal-weight.
   All general knowledge, no specific solutions named.

   **Revised the same day, after review.** The first draft of these bullets did
   name specifics, and the "no specific solutions named" claim above was not true
   of it. Three corrections:

   - *"e.g. LambdaRank optimises nDCG"* — removed. The principle (direct metric
     optimisation beats a proxy loss) is general; the named method is a hint, and
     it pointed straight at the library added in change 2 below. Naming both is
     effectively telling the agent what to try.
   - *"e.g. tabs with very different positive rates"* — removed. That is
     `CLAUDE.md`'s 44-point tab spread, which **we** measured and which no run has
     found independently in four attempts. Handing it over converts our analysis
     into the agent's. The general form — rescale per group if scales differ —
     stays, because that is textbook.
   - The convergence arithmetic was **wrong**. The bullet claimed a +0.001
     refinement step can never clear the 0.002 threshold. `converged()` compares
     `max(last 3)` against `max(everything before)`, so three consecutive +0.001
     steps clear it by +0.003; only an *isolated* small gain converges. And with
     `MIN_SCORED_BEFORE_CONVERGENCE = 30` nothing converges before experiment 30
     regardless. Worse, it pulled against the draft/improve/debug policy sitting
     directly above it in the same prompt — the policy written specifically to
     stop record-run-1's `2,2,2,2,2,2` breadth-only search. The ordering advice
     (ambitious first) is kept; the arithmetic is gone.

   Worth recording that the per-group rescaling bullet is **not** inert on this
   data, which is why the general form was kept rather than dropped: 39.8% of
   valid users span more than one tab, those users hold 57.3% of valid rows, and
   their dominant tab carries a median 66.7% of their impressions. For the other
   60.2% of users a per-tab rescale is a monotone transform inside a single group
   and cannot change their ranking at all.

2. **Library list expanded** — added `xgboost 3.x` and `lightgbm 4.x` to the
   "What is installed" section. Both added to `requirements.txt`. This unlocks
   LambdaRank/LambdaMART (direct nDCG optimisation) without telling the agent
   to use them — it discovers when and whether to.

What was deliberately NOT changed:
- No cross-run knowledge injected (dead ends, what worked before)
- No specific methods named in the prompt
- The 7 directions list unchanged
- All existing guardrails, search policy, and ensembling rules unchanged

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
signal** and is unlikely given the deceleration above. Run 4 supports this: a
fully independent search down a different mechanism family (sampled softmax
rather than a BPR ensemble) landed at +0.0031, below run 3's +0.0040. Two
searches converging near the same place from different directions is what a
ceiling looks like.

**Run 5 tests the ceiling rather than widening it.** Giving the agent
gradient-boosted ranking models (xgboost, lightgbm) opens a structurally
different family: every prior run optimised a proxy loss (BPR, BCE, sampled
softmax) and hoped the ranking metric would follow, and none had access to a
library-backed ranker that optimises nDCG directly.

Do not assume that raises the score. The measured ceiling analysis says
otherwise — we are already at 0.6055 against a half-fitted non-personalised
ceiling of 0.6048, with 1.62% pair coverage and a median user carrying 31 rows
across 29 videos. A gradient-boosted ranker sees exactly the same features and
the same coverage; it cannot manufacture signal that is not in the data.

**Which is why a null result is the valuable outcome, not the disappointing
one.** Right now the saturation argument rests on runs 3 and 4 converging to
+0.0040 and +0.0031 down two different mechanism families. If run 5 is handed
gradient boosting *and* direct nDCG optimisation and still lands near 0.605,
that is a third structurally independent search arriving at the same place, and
the saturation case becomes materially stronger. It is evidence, not a proof of
a global ceiling. That is worth more to the writeup than +0.0005 on the primary
metric would be.

Either way run 5 answers something. Frame it that way before launching, so a
flat result is read as evidence rather than as failure.

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
- ~~The search policy has been exercised exactly once.~~ **Four times now** —
  runs 2, 3, 4 and 6 all produced real trees. The draft/improve/debug policy
  reproduces. Run 6's shape: four children off `#29`, three each off `#1`, `#8`
  and `#12`.
- **New, from run 6:** two harness defects worth fixing before any run 7 — the
  agent cannot see experiment cost, and identical per-seed predictions are
  reported as `+/- 0.000000` rather than flagged as unmeasured. Both are written
  up under *Three harness defects run 6 exposed*.
- **A solution already in the ledger cannot be re-measured in place.** The
  source-hash guard returns `duplicate`. Fine for a fresh ledger; use
  `harness/seedsweep.py` (which writes nothing) to re-measure anything old.
- **Branch strategy** — the agent will commit on every iteration. Two humans
  plus an agent on `main` will collide.
- **`_budget_check()` counts every ledger row, not this session's.** Fine for a
  fresh-ledger record run; misleading if you ever want "20 more experiments" on
  top of an existing ledger.
- ~~Test has never been scored.~~ **Done 2026-08-28, recorded in
  `test_scores.json`.** See *The test score* below.
- **The submission artifact does not exist yet.** The tooling is built and
  validated on valid; nobody has produced `submission.csv`.
- **Spinner rendering is only fixed for width.** If the terminal is *resized*
  mid-run the padding is recomputed each frame, so it self-corrects, but a
  narrower window mid-write can still leave one stale row.
- **Prompt hygiene:** the agent's search queries have included the phrase
  "citation URL", leaked from the prompt line asking it to cite. Harmless, but it
  dilutes the query.

---

## The multi-agent question, settled by measurement, 2026-08-30

A teammate proposed a role-split agent (researcher / implementer / reviewer);
separately we proposed a parallel portfolio of independent agents. **Both were
refuted against our own logs, and the second one was refuted by a measurement
that also corrects a claim made earlier in this file.** What survived is small,
cheap and not architectural.

### The correlation claim in this file was measuring the wrong thing

Above, under the ensembling section, this file says independent runs are
decorrelated at "r = 0.89-0.96 across runs against 0.98-0.99 within one", and
uses that to explain the 0.6063 cross-run blend. **That figure is global
correlation, and global correlation is nearly unrelated to what we are scored
on.** GAUC and nDCG@5 are within-user, so the quantity that matters is
within-user rank correlation. Measured on the six blended winners:

```
run4 vs run5   global r = 0.9794   within-user rho = 0.9274
run4 vs run6   global r = 0.5023   within-user rho = 0.9140
```

Global swings from 0.49 to 0.98 while the within-user figure barely leaves 0.91.
Recomputed properly (`scratchpad/decorr.py`, no retraining - it reuses the
winner predictions cached by `blend6.py`):

```
WITHIN one run   run 3's ensemble members       mean rho = 0.8811
ACROSS runs      identical-config winners       mean rho = 0.9119
                                                gap      = -0.0307
```

**The sign is backwards from the portfolio thesis.** One run's own ensemble
members are *more* decorrelated than winners drawn from separate runs. Broken
out, within-run structural-vs-structural pairs reach rho 0.7975 - the most
diverse pairs anywhere in the archive - while cross-run winners sit at 0.9119,
about where *seed variants of one model* sit (0.9201).

So the 0.6063 blend is real but the explanation was wrong: it came from
averaging six strong models at rho ~0.91, not from independent context producing
diverse ones. Independent runs converge on the same mechanisms. **Diversity comes
from deliberately retaining different models, not from hoping independent search
produces them.**

One caveat, stated so nobody over-reads it: the within-run set is ensemble
*members* (some deliberately weak and built to differ) and the across-run set is
*winners* (all strong, all near the ceiling), so selection explains part of the
gap. It does not reverse the direction.

**Consequence:** a multi-trajectory portfolio would spend a week reproducing rho
0.91 when a single run already reaches 0.80 internally. Dropped. What replaces it
is correlation-aware retention - keep a weaker-but-different model as a diverse
reserve instead of discarding everything below the incumbent - measured with
**within-user rank correlation, never global Pearson**.

### The reviewer agent, audited against the record

The proposal listed six defects an always-on reviewer would have caught. Checked
against all 303 iterations:

```
string IDs joined against int keys    no instance found
labels entering join keys             13 hypotheses already reason about leakage;
                                      run-9 it=20 self-diagnosed it, fixed in it=21
screen score read as validation       OUR error in a scratchpad script, twice.
                                      logs/record-run-12/0005.json is correctly
                                      stamped verdict: screen
GroupCE without artifacts             also ours, outside the agent loop
CUDA installed, harness on CPU        REAL - harness config, fixed by an assert
ensemble refits unchanged members     REAL - already addressed via prompt caching
```

Four of six never happened inside the loop a reviewer would police. The two real
ones are exactly the class the proposal itself says should be *"code, not another
LLM opinion"*. The harness already implements no-op detection (7 caught),
solution fingerprinting, split-aware verdicts and seed spread; `devdata.py` has
no `test` key at all, so test access is structurally impossible rather than
policed. **No reviewer agent.**

### Explore vs exploit: real, but smaller than the labels suggest

Hypothesis openers across runs 4-11 (n=222): `improve` 65%, `draft` 22%,
`debug` 10%. Longest run of consecutive non-draft iterations: **16**, in run 10.

That overstates it. Reading run 10's actual chain, iterations 16-24 are labelled
`improve` but test LambdaRank, target stats, three-way blends and xendcg - four
or five genuinely different mechanisms. **The real pathology is the tail:
iterations 25-31, seven experiments all off node 24, all landing inside one error
bar of each other.** Counting by hypothesis prefix measures naming, not
behaviour, which also means the draft-vs-improve productivity table (drafts 50%
success vs improves 29.5%) cannot carry causal weight until mechanisms are
classified. Record `family_id` / `mechanism_id` / `parent` / `change_type` before
using that comparison for anything.

### The two-strike gate would have forbidden half our winners

Proposed policy: a parent branch takes a strike per non-improving child, closes
on the second. Replayed over every archived ledger (`scratchpad/strikes2.py`):

```
run-3   0.605493  027_deepfm_member.py          parent had 2 strikes   FORBIDDEN
run-4   0.604615  031_user_trend_position.py               3           FORBIDDEN
run-5   0.605368  026_mixed_tabhour_simple...              6           FORBIDDEN
run-8   0.604793  024_lgb_user_pref_residual.py            3           FORBIDDEN
run-9   0.605738  031_global_ctr_tiebreak.py               2           FORBIDDEN

winners forbidden: 5 of 12 runs (6 of 12 with a noise margin)
```

Run 3 is **our submission of record**. Run 9's 0.605738 is the seed-picked figure
- its honest 3-seed value is 0.605353, see *Zheng's 0.6057 is seed-picked* above
- but the replay question is unaffected either way, since the gate would have
closed that branch before the experiment ran at all. Run 5's winner arrived after
*six* consecutive non-improving children of the same parent. Winners routinely
arrive after a string of failures on one node; that is persistence paying off,
not grinding.

This independently rediscovers the calibration already recorded in
`agent/prompt.py`: the plateau detector's threshold comment says *"At 4 or 5 it
fires on record-run-3 at iteration 12-13, which still had +0.00097 to give - a
false alarm on the best run we have."*

**The distinction we were missing: non-improving is not the same as
uninformative.** Run 5's six failures were varied attempts carrying real
information. Run 10's it=25-31 were seven variants inside one error bar carrying
none. The thing worth suppressing is *unmeasurable* variation, not *unsuccessful*
variation - and the existing plateau detector already measures the right
quantity. Gate dropped; detector kept.

Its real defect is different and remains open: it counts backwards and needs six
consecutive indistinguishable results, so **it cannot fire until six experiments
have already been wasted**. Its own comment concedes this - *"fires with <=1
experiment left, where it cannot do harm"*. Runs 10 and 11 are the only two that
ran with it, and they hold the two longest chains (16 and 12). The open question
is whether it should *refuse* the experiment rather than advise against it. One
flag, testable in a shakedown, not yet done.

### A methodological trap worth remembering

The first replay tried to simulate the whole counterfactual tree - block a node,
delete its descendants, ask whether the winner survived. It produced identical
numbers under four different success definitions, which was the tell. **You
cannot replay a policy that changes what happens next against a fixed log of what
did happen**: if the gate blocks an experiment the agent does something else, so
the recorded descendants are not the counterfactual. Only the assumption-free
question is answerable from a log - *had this node's parent already taken two
strikes when the winner ran?* - and that is what the numbers above report.

### web_search was never restricted; our prompt was

`do_web_search` calls OpenAI's `web_search_preview` through `gpt-4o-mini`. That
is general web search - Kaggle, Hugging Face, GitHub, forums, all reachable.

```
0 of 86 queries mentioned kaggle / huggingface / github / competition / leaderboard
6 of 86 search results retained any URL at all
```

All 86 were academic-method queries, because the prompt said *"published methods
are explicitly in scope"* and *"standard formulation"*. The agent obeyed
perfectly. The capability was there the whole time and we told it to use a
library card. And the prompt's instruction to *"put the citation URL in your
hypothesis"* was near-unsatisfiable: `output_text` returns prose with links
stripped, so the provenance for research-derived ideas is thinner than it looks.

**Fixed 2026-08-30** in `agent/prompt.py`: the search section now names four
source families (papers, competition writeups, model/dataset cards,
implementations and library docs), tells the agent to name the kind of source in
the query, and tells it to ask for the URL *inside* the query since a link it
does not request is dropped. Budget rules unchanged - still 1 search per
iteration, still new-directions-only. Untested; run 13 is the first run with it.

Caveat: `gpt-4o-mini` does the searching and summarising, which is weak for
judging which parts of a Kaggle writeup matter. The pages are reachable now; how
much survives summarisation is unknown.

### author_id is nearly a copy of video_id on Pure

```
7,538 videos    6,482 authors    ->  1.2 videos per author
valid rows whose video is unseen in train : 17 (0.0%)
author rescues over video_id              : 3 rows
```

Almost every creator has exactly one video in Pure. As a categorical field
`author_id` carries almost nothing `video_id` does not, and as cold-start backoff
it rescues three rows out of 124,909.

**This reframes what our best features are doing.** At 1.2 videos per author,
"user x author history" is nearly the same object as "user x video history", so
the mechanism carrying our top scores (0.605438, 0.605367, 0.605216) is
**repeat-consumption memory**, not creator affinity. It also explains the
organisers' own result that `user_id x video_id` already captures most of the
learnable signal, and that the extra CWM fields add nothing - several of them
re-encode the same partition.

Unverified: whether this holds on 1k/27k, where the same authors should span many
more videos and `author_id` would become a real backoff. One CSV read answers it.

### Attribution correction for the writeup

**No organiser document forbids using `play_time_ms` as an input feature.** The
README mentions it once, under direction 3: `is_click`, `is_like`, `is_follow`,
`is_comment`, `is_forward` and `play_time_ms` *"can serve as auxiliary tasks
alongside the main long_view objective"*. Tasks, not features - an implication,
not a prohibition.

The exclusion is still correct and still ours: `long_view` is a deterministic
function of `play_time_ms` and `duration_ms` (97.8% agreement), so feeding it in
reads the label through a thin wrapper, and nothing stops you mechanically -
valid and test rows do contain the column, and `evaluate.py` only takes three
arrays. **Describe it as our judgment call, not as an organiser rule.**

By contrast `log_random_4_22_to_5_08_pure.csv` *was* stated by the organisers
(Q&A, 2026-08-27) - but not in the README, which only calls it an unbiased
validation set.

### Correction: runs 4-11 were not identical

Earlier analysis treated runs 4-11 as configuration-identical. The harness config
genuinely is - `(50 experiments, 21600s, min_scored 30, gpt-5.5)`, one distinct
value across all eight. But **`agent/prompt.py` was committed ten times inside
that window**, so the runs are not interchangeable and should not be pooled
without saying so.

### Where this leaves the architecture

```
multi-trajectory portfolio    DROPPED   measured - cross-run rho 0.91 vs within-run 0.88
always-on reviewer            DROPPED   4 of 6 claimed catches never happened in the loop
two-strike branch gate        DROPPED   would forbid 5-6 of 12 winners
research scout                DEFERRED  only if mechanism lock-on survives the prompt fix
broader web_search            DONE      2026-08-30, untested
device assert + reporting     TODO      ~20 lines
within-user correlation       TODO      ~20 lines, feeds diverse-reserve retention
family/mechanism identifiers  TODO      needed before the draft-vs-improve table means anything
binding plateau detector      OPEN      one flag; replay it against all 12 runs first
```

**This was a search-policy question, not a multi-agent one.** Every architectural
proposal died against the logs; the survivors are instrumentation.

### Runs 13 and 15, and what four clean runs agree on, 2026-08-30

Two runs overnight, both converged on the organisers' rule with experiments to
spare. Run 15 is the first run on the fixed duplicate guard and the cleanest run
in the archive: 30 records, 30 scored, no failure, no no-op, no duplicate.

```
                    primary     GAUC       nDCG@5     winner
run 3    (submitted) 0.605493   0.672469   0.538518   027_deepfm_member.py
run 13               0.605484   0.672606   0.538361   030_time_nodate_60.py
run 15               0.605499   0.672312   0.538686   029_time_seq_exposure_context.py
run 9    (3-seed)    0.605353   -          -          seed-picked headline was 0.605738
```

**Three independent searches, three unrelated winning mechanisms, a spread of
0.000015.** DeepFM ensemble member; time features blended at 60%; time plus
sequence plus exposure context. Against a seed sd of 0.0008 and a run-to-run sd
of 0.00034, that is four runs measuring the same ceiling with different
instruments. Nothing here promotes: the bar is +0.0016 over run 3 and the whole
spread is a hundredth of that.

Run 15's nominal +0.000006 over run 3 is six millionths. It is not a result.

### The broadened web_search changed behaviour, measurably

Run 13 was the first run on the 2026-08-30 prompt change (four named source
families, ask for the URL inside the query). Run 15 followed.

```
                            all 12 prior runs    run 15
searches                    86                   8
reached a practitioner source  0                 4
results retaining any URL      6                 21
```

Zero for eighty-six became four for eight. One query was explicitly
`Kaggle CTR ranking competition target encoding user item author historical
features leakage safe cite URL`, and the agent acted on it: iterations 16 and 17
were LambdaRank and LightGBM target-encoding blends. Both scored badly
(0.600108, 0.601697), so the Kaggle-derived idea failed on its merits - but the
change in *where the agent looks* is unambiguous, and the citation trail is now
21 URLs instead of 6 across a run an order of magnitude longer.

### GPU: about 1.4x, not the four hours it looks like

```
run                 recs    wall    sec/exp
record-run-3          30   6.48h        711   CPU
record-run-5          28   4.29h        500   CPU
record-run-9          33   3.63h        329   CPU
record-run-8          31   1.63h        130   CPU   <- faster than either GPU run
--------------------------------- GPU wired 08-29 23:53 ---------------------------
record-run-13         32   1.65h        138
record-run-15         30   1.66h        160
```

Comparing run 15 to run 3 suggests a 4.8-hour saving. It is not: run 8 finished
in 1.63h on the CPU, faster than both GPU runs. Per-experiment cost swings 5x on
the *same hardware* depending on whether the agent is fitting one FM or a
ten-member bag, and run 3 spent its time on heavy blends.

Against the CPU mean of 3.08h the GPU saves about 1.4h; against the CPU best it
saves nothing. **The controlled measurement - 1.40x on identical work - remains
the number to quote.** Live sampling agrees: utilisation spikes to 99% then falls
to 4-7% between bursts, because a k=16 FM over 7.5k videos cannot saturate a 3060
Ti and the per-batch host-to-device copy dominates. A bigger card buys nothing;
moving the training tensors onto the device once at startup might.

So "every run now finishes in 2 hours" is not safe from n=2. A run that stays on
light models does; a run that goes ensemble-heavy would still be 4h+ even at
1.40x.

### The duplicate guard had two defects, and the test found the second

Fixed in `44a63e6`, before run 15.

`find_by_hash` keyed on source alone, so the dev holdout's whole point - screen
cheaply, then confirm on valid - was blocked: run 13 screened at iteration 14,
tried the same code on valid at 15, and was rejected as a duplicate. One slot out
of 50 spent learning nothing, and the idea left neither confirmed nor refuted.

The first fix (key on `(source_hash, split)`) still failed, and only the test
showed why: **duplicate records were themselves being indexed**, so the rejection
became the prior that rejects the retry. A refusal is not a run and must not
enter the index.

Verified against all 13 archived runs: zero same-source-same-split collisions the
guard would no longer catch. It has in fact never caught a real rerun - its only
firing in the project's history is the false positive above.

**Untested live.** Run 15 used no dev screens at all, so the promotion path still
has not been exercised by an agent.

### Open, for whoever picks this up

- `harness/data.py` (main) and Zheng's `feature/kuairand-1k` solve the same
  variant-loading problem two ways. His edits `kuairand-starter-kit/data.py`
  directly, which forfeits the property that our copy of the kit is byte-identical
  to the official release - and deletes an organiser comment marking where
  students are meant to add features. The branch also removes the
  `filter on verdict` warning from this file. Resolve before merging.
- The training tensors sit in host memory and are indexed there every batch
  (`xp = Xtr_t[ps].to(device)`). Moving them to the device once is the available
  speed win, and it is larger than the one the GPU itself bought.

### UCT shakedown, 2026-08-30 — the adherence measurement

`uct-shakedown-1`, stopped by hand at 27 records / 23 scored. Deleted after
recording these numbers; it was a shakedown, not a record run, and its value was
answering one question: does the agent actually use the ranking?

**It does, and by a lot.**

```
chose the UCT-top node        83%   (19 of 23 turns)
                              40%   (13 greedy runs, ranking not shown)
rank of chosen parent         #1 x19   #2 x2   #3 x1   #6 x1
divergence appeared late      ranks at iteration 20+: 1,1,2,6,3,1,2,1
```

That matters because the previous thing we tried advising - the plateau
detector - was measurably ignored: runs 10 and 11 both ran with it and produced
the two longest exploit chains in the archive (16 and 12). Advice does not
automatically bind. This did.

Early turns all sat at rank #1 because greedy and UCT agree until the incumbent
has children; divergence only appears once a node has been expanded a few times,
which is exactly when the greedy failure mode starts.

**Score: undetermined, and the run was too short to say otherwise.**

```
at 20 scored   uct 0.605177   greedy best 0.605191  median 0.604746   2 of 11 ahead
at 23 scored   uct 0.605177   greedy best 0.605353  median 0.605012   3 of 11 ahead
```

Above the greedy median, inside the greedy range. One run landing in the top
quartile happens by chance about a quarter of the time, so this is not evidence
of an effect. Reading it against runs 13 and 15 alone made it look stronger than
it is - those are two of the weaker greedy runs.

**One thing to watch:** 4 failed experiments (memory, timeouts) against 0 in each
of runs 13 and 15. Possibly UCT steering toward more ambitious untried branches -
DIN, LambdaRank and listwise all appear - or possibly noise at n=1.
