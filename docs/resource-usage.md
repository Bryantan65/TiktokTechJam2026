# Resource usage

Deliverable 4. Two figures matter and they are different things: what the
**submitted run** cost, and what the **whole project** cost. Both are below.

Every number here is read back from `logs/*/events.jsonl` and
`logs/*/NNNN.json`, not from anyone's memory.

---

## The submitted run — `record-run-3`

```
iterations used            30  of the 50 allowed
stopped                    converged on the organisers' rule, not a cap
wall clock                 6 h 29 m   (see note below)
solution compute           5 h 56 m
LLM                        136 API calls
tokens                     2,293,488 in  /  169,094 out
                           1,676,544 of the input served from cache (73%)
cost                       $3.65
self-directed web searches 10
solution crashes           0
human interventions        0
```

**Zero within-run interventions.** The process was launched once and not
touched until it stopped itself. That is the number the autonomy criterion
asks for, and it is separate from the development figures below.

**On the 6 h 29 m.** This run predates the wall-clock guard and would today be
stopped by it — the organisers' 6 h ceiling was published on 2026-08-27, after
this run. Every run since respects it; runs 8, 9 and 10 finished in 1 h 38 m,
1 h 48 m and 1 h 58 m. This is disclosed rather than quietly omitted because
the alternative is a compliance claim we cannot make.

---

## The whole project — 15 runs

Includes the two shakedowns, the deliberately-voided run, the crashed run, the
model bake-off, and every record run.

```
runs                       15
experiments                272
wall clock                 23 h 43 m
solution compute           20 h 5 m
LLM                        1,034 API calls
tokens                     17,982,087 in  /  1,320,971 out
                           13,389,290 of the input served from cache (74%)
cost                       $33.10
self-directed web searches 72
solution crashes           8, all recovered from   (10 recovery events)
```

The 74% cache rate is not incidental. The loop resends the whole conversation
every tool round, so most input tokens are cache hits; billing them at full
rate would overstate spend roughly threefold. `AGENT_CACHED_COST_PER_M`
accounts for them separately, and per-provider rates are set explicitly
because DeepSeek charges cache hits at full price where OpenAI discounts them
~10x.

---

## Interventions, stated precisely

**Within the submitted run: zero.** Launched once, converged on its own.

**Between runs: many, and they are development.** The prompt, harness and
tooling were revised repeatedly across three days — the search policy, the
convergence floor, multi-seed scoring, the cost column, deterministic-seed
detection, the plateau detector, prediction caching. Each is recorded in git
with the measurement that motivated it.

Two restarts happened and both are on the record:

- `record-run-7` crashed at iteration 2 on a `UnicodeEncodeError` printing an
  arrow to a cp1252 stream. Fixed, relaunched as `record-run-8`. The
  organisers' Q&A of 2026-08-28 confirms a crash restart is not intervention
  provided no behaviour or parameter changes; the fix was to console output
  only.
- `void-run-1` was stopped deliberately after three experiments on noticing
  `converged()` was about to end a run prematurely. Documented as void and
  never presented as a result.

---

## Efficiency work, with what it was worth

Measured, including the two that did not work:

| change | effect |
| --- | --- |
| prompt caching accounting | 3x reduction in reported cost, and it is the honest figure |
| concurrent seeds | 2.10x on 12 cores; brought runs back inside the 6 h ceiling |
| history compaction | keeps input bounded as the ledger grows |
| cost column + batching guidance | agent can see what an experiment costs; unmeasured effect |
| adaptive seeding | **slower** — 69.6 s against 49.1 s. Default off, kept behind a flag |
| prediction caching | adopted from `experiment/new-directions`; blend experiments in seconds rather than retraining every member |

Adaptive seeding is listed because it failed. Seeds already run concurrently,
so screening one instead of three frees CPU without shortening the wall clock,
and the workload is memory-bandwidth bound so the freed cores buy nothing.
Predicted ~20% saving, measured 42% slower.
