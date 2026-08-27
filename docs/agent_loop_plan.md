# Implementation Plan: Agent Loop

## Requirements

Build the autonomous ML experiment agent that:
- Uses GPT-4o (OpenAI SDK) to propose hypotheses and write solution code
- Runs experiments via the existing harness, reads results, iterates
- Has 5 tools: `read_ledger`, `read_solution`, `write_solution`, `run_experiment`, `web_search`
- Supports two modes: **autonomous** (runs until convergence) and **supervised** (pauses after each iteration for human approval)
- `web_search` capped at 1 per iteration, primarily for beyond-the-7 directions, citation logged
- `write_solution` restricted to `solutions/` only
- Convergence = 3 consecutive ok experiments without improvement > 0.002
- Max iteration cap as a safety backstop

## Patterns to Mirror

| Category | Source | Pattern |
|---|---|---|
| Naming | `solutions/001_torch_fm.py` | `NNN_snake_name.py` -- zero-padded iteration number + descriptive slug |
| Error handling | `harness/run.py` | Never raises -- always returns a dict with `status`, `error` fields. Errors are data, not exceptions. |
| Data flow | `harness/run.py` -> `harness/ledger.py` | Harness owns scoring and logging. Solutions emit predictions only. |
| Config | `.env` | API key loaded from `.env` via `python-dotenv` |

## Files to Create

| File | Action | Purpose |
|---|---|---|
| `agent/__init__.py` | CREATE | Package marker |
| `agent/tools.py` | CREATE | 5 tool definitions as functions + OpenAI tool schemas |
| `agent/prompt.py` | CREATE | System prompt: task context, 7 directions, constraints, solution contract |
| `agent/loop.py` | CREATE | Main loop: call LLM -> execute tools -> log -> repeat. `--supervised` flag for pause mode |

## Dependencies

- `openai` -- the SDK
- `python-dotenv` -- load `.env`

Both need adding to `requirements.txt`.

## Task 1: `agent/__init__.py`

Empty package marker.

## Task 2: `agent/tools.py` (~100 lines)

Five tools, each a plain function + an OpenAI function-calling schema:

| Tool | What it does | Key constraint |
|---|---|---|
| `read_ledger()` | Returns `LEDGER.md` content + last 3 full JSON records | Read-only |
| `read_solution(path)` | Returns source code of a solution file | Restricted to `solutions/` |
| `write_solution(filename, code)` | Writes code to `solutions/{filename}` | Must start with `NNN_`, must be inside `solutions/`, validates Python syntax before writing |
| `run_experiment(solution, hypothesis, parent)` | Calls `harness.run.run_experiment()` and returns the result dict | Delegates entirely to existing harness |
| `web_search(query)` | Calls OpenAI's web search tool (or a lightweight arxiv/scholar search) | Returns top results with URLs for citation |

`write_solution` enforces:
- Path must resolve inside `solutions/`
- Filename must match `NNN_*.py` pattern
- Python syntax check before writing (same `ast.parse` the harness uses)
- Returns the written path on success, error message on failure

## Task 3: `agent/prompt.py` (~80 lines)

The system prompt tells the agent:
- **What it is:** an ML experiment agent optimising within-user video ranking
- **The goal:** beat valid primary 0.6035 (baseline 0.6015 + epsilon 0.002)
- **The metrics:** GAUC, nDCG@5, primary = mean of both
- **The solution contract:** `python solutions/NNN.py --data_dir DIR --split valid --out FILE.npy` -> one float per row
- **The 7 known directions** (ranked by likely payoff): loss function change, user sequences, multi-task, watch-time modelling, different models, time features, unbiased validation
- **Dead ends:** more features and more capacity are measured negatives
- **Key facts:** seed noise is 0.0008, nDCG ceiling is 0.729, user-side features contribute zero in isolation, `(user_id, video_id)` is not unique
- **The starting solution:** `solutions/001_torch_fm.py` -- a PyTorch FM with pointwise logloss
- **Workflow:** read ledger -> identify what to try -> write solution -> run experiment -> read result -> decide next
- **Search constraint:** "the 7 directions are your default. Search only when going beyond them, max 1 search per iteration. Log the citation."
- **What NOT to do:** never modify files outside `solutions/`, never compute your own metrics, never request test split

Also includes a function to build a dynamic user message each iteration with the current ledger state and best solution source.

## Task 4: `agent/loop.py` (~120 lines)

The main entry point:

```
python -m agent.loop              # autonomous mode
python -m agent.loop --supervised  # pause after each iteration
python -m agent.loop --max-iter 50 # safety cap (default: 100)
```

### Loop logic

1. Load API key from `.env`
2. Build system prompt
3. **While** not converged and iteration < max:
   a. Build user message (current ledger + best solution)
   b. Call GPT-4o with tools
   c. Execute tool calls, collect results
   d. Track `web_search` usage (enforce 1-per-iteration cap)
   e. Feed results back, let the model call more tools or finish its turn
   f. Log the iteration summary to stdout
   g. **If supervised:** print result, ask `[Enter] to continue, [q] to quit`
   h. Check `ledger.converged()`
4. Print final summary

### Tool execution loop (inner)

The model may call multiple tools per turn (e.g. read_ledger -> read_solution -> write_solution -> run_experiment). Loop until the model sends a text response (no more tool calls).

### Error handling

If the model produces an invalid tool call, return the error as a tool result and let it self-correct. Never crash the loop.

### Search cap

A counter resets each outer iteration. If the model calls `web_search` and the counter is already at 1, return "search limit reached for this iteration" instead of executing.

## Task 5: Update `requirements.txt`

Add `openai` and `python-dotenv`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Model writes solutions that crash | HIGH (expected) | Harness catches everything, returns error to agent, agent self-corrects |
| Model ignores prompt constraints | MEDIUM | Constraints enforced in code (path restriction, search cap), not just prompt |
| Model loops on the same failing idea | MEDIUM | Duplicate detection via source hash already in harness |
| OpenAI API rate limits | LOW | Add retry with backoff on 429 |
| Token cost spirals | LOW | Max iteration cap, and each iteration is bounded (one LLM turn with tools) |

## Validation

```bash
# 1. Check imports work
python -c "from agent import tools, prompt, loop"

# 2. Dry run - supervised mode, watch first iteration
python -m agent.loop --supervised --max-iter 1

# 3. Verify solution path restriction
# (agent shouldn't be able to write outside solutions/)

# 4. Verify search cap
# (in supervised mode, check that second search in same iteration is blocked)
```
