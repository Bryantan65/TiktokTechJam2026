# Handoff

State, decisions, and what's next. **`CLAUDE.md` holds the task facts** — label,
metrics, splits, baselines, dead ends. This file holds the *decisions* and the
reasoning behind them, which is the part that's expensive to reconstruct.

Last updated: 2026-08-27, commit `1efdf9a`.

---

## Where things stand

```
✅ environment verified against the official numbers
✅ official baseline reproduced (numpy) and ported (PyTorch)
✅ harness built and its guardrails tested
⬜ agent loop
⬜ shakedown + hardening
⬜ record run
```

### What's built

| | |
| --- | --- |
| `kuairand-starter-kit/` | official kit. **Read-only, never modify.** `evaluate.py` is the sole authority on scoring. |
| `solutions/000_baseline.py` | untouched copy of the official FM |
| `solutions/001_torch_fm.py` | same model in PyTorch. **The substrate every experiment builds on.** |
| `harness/run.py` | runs one solution, scores it, logs it. Never raises. |
| `harness/ledger.py` | `logs/iterations/NNN.json` + `LEDGER.md`; owns the verdict rule |
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

Still to add, agent-side: `write_solution` must reject any path outside
`solutions/`. Otherwise the agent could edit `harness/` and the guardrails
become suggestions. Same for `kuairand-starter-kit/`.

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

1. **`agent/`** — `tools.py`, `loop.py`, `prompt.py`, ~250 lines. Needs an API
   key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`); the architecture is
   provider-agnostic behind a thin interface.
2. **Supervised shakedown** — run it, watch it break, **log every intervention**.
   That log is the hardening backlog.
3. **Harden** — turn each intervention into a mechanism.
4. **Record run** — fresh state, start it, walk away. Budget two or three
   attempts; the first clean run usually fails on something never seen in
   development. Don't schedule it for the last night.

First experiment once the agent exists: **pairwise or listwise loss.** The
organisers rank it #1, and the training trace agrees — valid peaks at epoch 7
and overfits after, which is a pointwise objective running out of road on a
ranking metric.

---

## Open

- ~~Is `log_random` trainable?~~ **Answered 2026-08-27: no.** Training on it
  means training on the evaluation period. Analysis and unbiased-evaluation
  experiments are fine; it must not fit the submitted model.
- **API key** — not obtained yet.
- **Branch strategy** — the agent will commit on every iteration. Two humans
  plus an agent on `main` will collide.
