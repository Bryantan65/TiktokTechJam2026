# Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autonomous ML experiment agent that proposes hypotheses, writes solution code, runs experiments via the harness, and iterates until convergence.

**Architecture:** OpenAI GPT-4o via raw SDK (Chat Completions API for the main loop, Responses API for built-in web search). No framework. Five function-calling tools: `read_ledger`, `read_solution`, `write_solution`, `run_experiment`, `web_search`. Two modes: autonomous and supervised.

**Tech Stack:** Python 3.12, openai SDK, python-dotenv

**Spec:** `docs/agent_loop_plan.md`

## Global Constraints

- API key loaded from `.env` via `python-dotenv`
- Solutions restricted to `solutions/` directory only
- Agent only sees `valid` split, never `test`
- `web_search` capped at 1 call per iteration
- Convergence: 3 consecutive ok experiments without improvement > 0.002
- Max iteration safety cap (default 100)
- Solution contract: `python solutions/NNN.py --data_dir DIR --split valid --out FILE.npy`

---

### Task 1: Package Setup and Dependencies

**Files:**
- Create: `agent/__init__.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing
- Produces: `agent` package importable, `openai` and `python-dotenv` available

- [ ] **Step 1: Create agent package**

```python
# agent/__init__.py
# empty — package marker
```

- [ ] **Step 2: Add dependencies to requirements.txt**

Append to `requirements.txt`:

```
openai>=1.66.0
python-dotenv>=1.0.0
```

- [ ] **Step 3: Install dependencies**

Run:
```bash
pip install openai python-dotenv
```

- [ ] **Step 4: Verify imports**

Run:
```bash
python -c "import openai; from dotenv import load_dotenv; print('ok')"
```

Expected: `ok`

---

### Task 2: Tool Definitions (`agent/tools.py`)

**Files:**
- Create: `agent/tools.py`

**Interfaces:**
- Consumes: `harness/ledger.py` (functions: `recent(n)`, `best()`, `LEDGER` path), `harness/run.py` (function: `run_experiment(solution, hypothesis, parent, by)`)
- Produces: `TOOL_SCHEMAS: list[dict]` (OpenAI function-calling format), `TOOL_DISPATCH: dict[str, Callable]` (name -> handler), `do_web_search(client: OpenAI, query: str) -> str`

- [ ] **Step 1: Write test for path restriction**

Create `tests/test_tools.py`:

```python
"""Safety tests for agent tools."""
import ast
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_write_solution_rejects_path_traversal():
    from agent.tools import write_solution
    result = json.loads(write_solution('../harness/evil.py', 'print("pwned")'))
    assert 'error' in result
    assert 'solutions/' in result['error'] or 'NNN_name.py' in result['error']


def test_write_solution_rejects_bad_filename():
    from agent.tools import write_solution
    result = json.loads(write_solution('evil.py', 'print("hi")'))
    assert 'error' in result


def test_write_solution_rejects_syntax_error():
    from agent.tools import write_solution
    result = json.loads(write_solution('002_test.py', 'def f(\n'))
    assert 'error' in result
    assert 'SyntaxError' in result['error']


def test_write_solution_accepts_valid():
    from agent.tools import write_solution
    code = 'print("hello")\n'
    result = json.loads(write_solution('099_test_valid.py', code))
    assert result['status'] == 'ok'
    # clean up
    path = os.path.join(os.path.dirname(__file__), '..', 'solutions', '099_test_valid.py')
    if os.path.exists(path):
        os.remove(path)


def test_read_solution_rejects_outside_solutions():
    from agent.tools import read_solution
    result = json.loads(read_solution('../harness/run.py'))
    assert 'error' in result


if __name__ == '__main__':
    test_write_solution_rejects_path_traversal()
    test_write_solution_rejects_bad_filename()
    test_write_solution_rejects_syntax_error()
    test_write_solution_accepts_valid()
    test_read_solution_rejects_outside_solutions()
    print('All tool safety tests passed.')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python tests/test_tools.py`
Expected: `ModuleNotFoundError: No module named 'agent.tools'`

- [ ] **Step 3: Implement tools.py**

Create `agent/tools.py`:

```python
"""Tool definitions for the agent loop.

Each tool is a plain function returning a JSON string. TOOL_SCHEMAS holds the
OpenAI function-calling definitions. TOOL_DISPATCH maps names to handlers.
web_search is handled separately in the loop (it needs the OpenAI client).
"""
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTIONS_DIR = os.path.join(ROOT, 'solutions')

sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402


def read_ledger() -> str:
    content = ''
    if os.path.exists(ledger.LEDGER):
        with open(ledger.LEDGER) as f:
            content = f.read()
    return json.dumps({
        'ledger_md': content,
        'recent_records': ledger.recent(3),
        'best': ledger.best(),
    }, indent=2)


def read_solution(path: str) -> str:
    full = os.path.join(SOLUTIONS_DIR, path)
    real = os.path.realpath(full)
    if not real.startswith(os.path.realpath(SOLUTIONS_DIR) + os.sep):
        return json.dumps({'error': f'path must be inside solutions/: {path}'})
    if not os.path.isfile(real):
        return json.dumps({'error': f'file not found: {path}'})
    with open(real) as f:
        return f.read()


def write_solution(filename: str, code: str) -> str:
    if not re.match(r'^\d{3}_[a-zA-Z0-9_]+\.py$', filename):
        return json.dumps({'error': f'filename must match NNN_name.py: {filename}'})

    full = os.path.join(SOLUTIONS_DIR, filename)
    real = os.path.realpath(full)
    if not real.startswith(os.path.realpath(SOLUTIONS_DIR) + os.sep):
        return json.dumps({'error': f'path escapes solutions/: {filename}'})

    try:
        ast.parse(code, filename=filename)
    except SyntaxError as e:
        return json.dumps({'error': f'SyntaxError: {e.msg} (line {e.lineno})'})

    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(code)
    return json.dumps({'status': 'ok', 'path': f'solutions/{filename}'})


def run_experiment(solution: str, hypothesis: str = '',
                   parent: str | None = None) -> str:
    sys.path.insert(0, os.path.join(ROOT, 'harness'))
    from run import run_experiment as _run  # noqa: E402
    full = os.path.join(SOLUTIONS_DIR, solution)
    result = _run(full, hypothesis=hypothesis, parent=parent, by='agent')
    return json.dumps(result, indent=2)


def do_web_search(client, query: str) -> str:
    try:
        response = client.responses.create(
            model='gpt-4o-mini',
            tools=[{'type': 'web_search_preview'}],
            input=query,
        )
        return response.output_text
    except Exception as e:
        return json.dumps({'error': f'web search failed: {e}'})


TOOL_SCHEMAS = [
    {
        'type': 'function',
        'function': {
            'name': 'read_ledger',
            'description': (
                'Read the experiment ledger: all past experiments and results. '
                'Returns LEDGER.md content, the last 3 full records, and the '
                'best result so far.'
            ),
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_solution',
            'description': (
                'Read the source code of a solution file. '
                'Path is relative to solutions/ directory.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': "Filename, e.g. '001_torch_fm.py'",
                    },
                },
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_solution',
            'description': (
                'Write a new solution file. Filename must match NNN_name.py. '
                'Code is syntax-checked before writing. Write the COMPLETE '
                'file — it must be a standalone script that accepts --data_dir, '
                '--split, --out, --seed and writes predictions as .npy.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'filename': {
                        'type': 'string',
                        'description': "e.g. '002_bpr_loss.py'",
                    },
                    'code': {
                        'type': 'string',
                        'description': 'Complete Python source code',
                    },
                },
                'required': ['filename', 'code'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'run_experiment',
            'description': (
                'Run a solution through the harness: execute, score against '
                'the official evaluator, log the result. Returns the full '
                'experiment record with metrics and verdict.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'solution': {
                        'type': 'string',
                        'description': "Filename, e.g. '002_bpr_loss.py'",
                    },
                    'hypothesis': {
                        'type': 'string',
                        'description': 'One-line description of what this tests',
                    },
                    'parent': {
                        'type': 'string',
                        'description': 'Iteration number this builds on, or null',
                    },
                },
                'required': ['solution', 'hypothesis'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': (
                'Search the web for research papers, implementations, or '
                'techniques. Limited to 1 search per iteration. Use ONLY when '
                'going beyond the 7 known directions. Include the citation URL '
                'in your hypothesis.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query',
                    },
                },
                'required': ['query'],
            },
        },
    },
]

TOOL_DISPATCH = {
    'read_ledger': lambda **kw: read_ledger(),
    'read_solution': lambda **kw: read_solution(kw['path']),
    'write_solution': lambda **kw: write_solution(kw['filename'], kw['code']),
    'run_experiment': lambda **kw: run_experiment(
        kw['solution'], kw.get('hypothesis', ''), kw.get('parent')),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python tests/test_tools.py`
Expected: `All tool safety tests passed.`

- [ ] **Step 5: Commit**

```bash
git add agent/__init__.py agent/tools.py tests/test_tools.py
git commit -m "agent: add tool definitions with path restriction and syntax validation"
```

---

### Task 3: System Prompt (`agent/prompt.py`)

**Files:**
- Create: `agent/prompt.py`

**Interfaces:**
- Consumes: `harness/ledger.py` (functions: `best()`, `LEDGER` path), `agent/tools.py` (function: `read_solution(path)`)
- Produces: `system_prompt() -> str`, `build_user_message() -> str`

- [ ] **Step 1: Write test for prompt functions**

Append to `tests/test_tools.py`:

```python
def test_system_prompt_contains_key_elements():
    from agent.prompt import system_prompt
    sp = system_prompt()
    assert 'GAUC' in sp
    assert 'nDCG@5' in sp
    assert '0.6035' in sp
    assert 'solutions/' in sp
    assert 'web_search' in sp


def test_build_user_message_returns_string():
    from agent.prompt import build_user_message
    msg = build_user_message()
    assert isinstance(msg, str)
    assert len(msg) > 0
```

Add to the `if __name__` block:

```python
    test_system_prompt_contains_key_elements()
    test_build_user_message_returns_string()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python tests/test_tools.py`
Expected: `ModuleNotFoundError: No module named 'agent.prompt'`

- [ ] **Step 3: Implement prompt.py**

Create `agent/prompt.py`:

```python
"""System prompt and per-iteration user message for the agent loop."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402

SYSTEM_PROMPT = """\
You are an ML experiment agent. Your goal: beat the FM baseline on \
within-user video ranking for the KuaiRand-Pure dataset.

## Target
Valid primary (mean of GAUC and nDCG@5) >= 0.6035.
Baseline: 0.6015. Epsilon: 0.002. Seed noise: 0.0008.

## Metrics
- GAUC: per-user AUC, weighted by positive count. Users with all-positive \
or all-negative impressions excluded.
- nDCG@5: per-user nDCG at cutoff 5. Zero-positive users score 0.0 and ARE \
included. Oracle ceiling is 0.729, not 1.0.
- Primary = (GAUC + nDCG@5) / 2.

## Solution contract
Every solution must be a standalone Python script:
    python solutions/NNN_name.py --data_dir DIR --split valid --out FILE.npy \
--seed SEED
It writes one float per row as a .npy file, then exits 0.
Solutions emit PREDICTIONS ONLY — never compute metrics. The harness scores.

## Starting point
solutions/001_torch_fm.py — PyTorch FM with pointwise BCEWithLogitsLoss.
Valid primary: 0.6014. Peaks at epoch 7-11, overfits after.
Fields: [user_id, video_id, author_id, tab, dur_bucket], k=16, lr=0.001, \
batch=8192, patience=4.

## 7 known directions (ranked by likely payoff)
1. **Change the loss** — pointwise logloss -> pairwise (BPR) or listwise \
(softmax over user impressions). Aligns objective with ranking metrics. \
MOST LIKELY TO WORK.
2. **User behaviour sequences** — DIN/SIM-style attention over a user's \
history. Currently no sequence info used at all.
3. **Multi-task** — is_click, is_like, is_follow, is_comment, is_forward, \
play_time_ms as auxiliary tasks.
4. **Watch-time modelling** — censored regression (CWM-style).
5. **Different models** — DeepFM / DCN / xDeepFM. Ranked BELOW 1-4 since \
capacity is not the bottleneck.
6. **Time features** — hourmin, date, train-vs-test drift.
7. **Unbiased validation** — log_random as extra validation (never train on it).

## Dead ends — do NOT try these
- More static features (all 13 CWM fields): 0.5940 vs 0.5950 — no gain.
- More capacity (k=8/16/32): 0.5895/0.5902/0.5887 — flat.
- Pure user-side first-order features: contribute exactly zero to \
within-user ranking.

## Key facts
- (user_id, video_id) is NOT unique — 3.06% duplicated, up to 12x.
- long_view rate varies by tab: 4.1% (tab 0) to 48.3% (tab 4).
- tab 1 is 73.7% of rows at 37.9% positive rate.
- A correct run takes ~30-50 seconds. Iterations are cheap.
- log_random_4_22_to_5_08_pure.csv must NOT be trained on.

## Workflow
Each iteration:
1. Call read_ledger to see what has been tried.
2. Call read_solution to read the best or parent solution.
3. Decide what to try. Explain your hypothesis in 1-2 sentences.
4. Call write_solution with the complete new solution file.
5. Call run_experiment to score it.
6. Interpret the result. Plan the next iteration.

## web_search
- The 7 directions above are your default. Search ONLY when going beyond them.
- Maximum 1 search per iteration.
- When you use a search result, include the citation URL in your hypothesis.

## Rules
- NEVER request --split test. You develop on valid only.
- NEVER compute your own metrics. The harness scores.
- NEVER modify files outside solutions/.
- Write the COMPLETE solution file every time — it must run standalone.
- Whole file for a new idea. Targeted edit (copy parent, change one thing) \
for a bugfix or parameter tweak.
- Prefer many small experiments over one big change. Iterations are cheap.
"""


def system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_message() -> str:
    parts = []

    if os.path.exists(ledger.LEDGER):
        with open(ledger.LEDGER) as f:
            parts.append('## Current ledger\n\n' + f.read())
    else:
        parts.append('## Current ledger\n\nNo experiments yet.')

    best_rec = ledger.best()
    if best_rec:
        solution_file = best_rec.get('solution', '')
        fname = os.path.basename(solution_file)
        src_path = os.path.join(ROOT, 'solutions', fname)
        if os.path.isfile(src_path):
            with open(src_path) as f:
                parts.append(
                    f'## Best solution so far: {fname} '
                    f'(valid primary {best_rec["valid_primary"]})\n\n'
                    f'```python\n{f.read()}```'
                )
    else:
        src_path = os.path.join(ROOT, 'solutions', '001_torch_fm.py')
        if os.path.isfile(src_path):
            with open(src_path) as f:
                parts.append(
                    '## Starting solution: 001_torch_fm.py '
                    '(valid primary 0.6014)\n\n'
                    f'```python\n{f.read()}```'
                )

    parts.append(
        'What would you like to try next? '
        'Read the ledger, examine the current best solution, '
        'then propose and run your next experiment.'
    )
    return '\n\n'.join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python tests/test_tools.py`
Expected: `All tool safety tests passed.` (now including prompt tests)

- [ ] **Step 5: Commit**

```bash
git add agent/prompt.py tests/test_tools.py
git commit -m "agent: add system prompt with task context and 7 directions"
```

---

### Task 4: Main Loop (`agent/loop.py`)

**Files:**
- Create: `agent/loop.py`
- Create: `agent/__main__.py`

**Interfaces:**
- Consumes: `agent/tools.py` (`TOOL_SCHEMAS`, `TOOL_DISPATCH`, `do_web_search`), `agent/prompt.py` (`system_prompt()`, `build_user_message()`), `harness/ledger.py` (`converged()`, `best()`)
- Produces: CLI entry point `python -m agent [--supervised] [--max-iter N]`

- [ ] **Step 1: Implement loop.py**

Create `agent/loop.py`:

```python
"""Main agent loop: call LLM -> execute tools -> log -> repeat."""
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402

from agent.tools import TOOL_SCHEMAS, TOOL_DISPATCH, do_web_search  # noqa: E402
from agent.prompt import system_prompt, build_user_message  # noqa: E402

MODEL = 'gpt-4o'
MAX_TOOL_ROUNDS = 20  # safety cap on tool calls per iteration


def _execute_tool(client, name: str, args: dict,
                  search_count: int) -> tuple[str, int]:
    """Run one tool call, return (result_json, updated_search_count)."""
    if name == 'web_search':
        if search_count >= 1:
            return json.dumps({
                'error': 'search limit reached for this iteration (max 1)'
            }), search_count
        search_count += 1
        query = args.get('query', '')
        print(f'    [web_search] {query}')
        result = do_web_search(client, query)
        return result, search_count

    if name in TOOL_DISPATCH:
        try:
            print(f'    [{name}] {_summarize_args(name, args)}')
            result = TOOL_DISPATCH[name](**args)
            return result, search_count
        except Exception as e:
            return json.dumps({'error': f'{name} raised: {e}'}), search_count

    return json.dumps({'error': f'unknown tool: {name}'}), search_count


def _summarize_args(name: str, args: dict) -> str:
    if name == 'read_solution':
        return args.get('path', '')
    if name == 'write_solution':
        return args.get('filename', '')
    if name == 'run_experiment':
        return f"{args.get('solution', '')} — {args.get('hypothesis', '')}"
    return ''


def run_loop(supervised: bool = False, max_iter: int = 100) -> None:
    load_dotenv(os.path.join(ROOT, '.env'))
    client = OpenAI()

    messages = [{'role': 'system', 'content': system_prompt()}]

    print(f'=== Agent loop started (supervised={supervised}, '
          f'max_iter={max_iter}) ===\n')

    iteration = 0
    while iteration < max_iter:
        if ledger.converged():
            print('\nConverged: 3 consecutive ok experiments without '
                  'improvement > 0.002.')
            break

        iteration += 1
        search_count = 0
        t0 = time.time()
        print(f'\n--- Iteration {iteration} ---')

        user_msg = build_user_message()
        messages.append({'role': 'user', 'content': user_msg})

        tool_rounds = 0
        while tool_rounds < MAX_TOOL_ROUNDS:
            tool_rounds += 1
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=0.7,
                )
            except Exception as e:
                print(f'  API error: {e}')
                if 'rate_limit' in str(e).lower() or '429' in str(e):
                    print('  Retrying in 30s...')
                    time.sleep(30)
                    continue
                break

            choice = response.choices[0]
            messages.append(choice.message)

            if choice.finish_reason == 'stop':
                if choice.message.content:
                    print(f'\n  [Agent] {choice.message.content[:500]}')
                break

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result, search_count = _execute_tool(
                        client, tc.function.name, args, search_count)
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': result if isinstance(result, str)
                                   else json.dumps(result),
                    })
            else:
                break

        elapsed = time.time() - t0
        print(f'  [{elapsed:.0f}s elapsed]')

        best_rec = ledger.best()
        if best_rec:
            print(f'  Best so far: valid primary '
                  f'{best_rec["valid_primary"]} '
                  f'({best_rec["solution"]})')

        if supervised:
            try:
                resp = input('\n  [Supervised] Enter to continue, '
                             'q to quit: ')
                if resp.strip().lower() == 'q':
                    print('  Stopped by user.')
                    return
            except (EOFError, KeyboardInterrupt):
                print('\n  Stopped.')
                return

        # Trim message history to avoid context overflow.
        # Keep system + last 20 messages.
        if len(messages) > 40:
            messages = messages[:1] + messages[-20:]

    print('\n=== Agent loop finished ===')
    best_rec = ledger.best()
    if best_rec:
        print(f'Best result: valid primary {best_rec["valid_primary"]} '
              f'({best_rec["solution"]})')
    else:
        print('No successful experiments.')
```

- [ ] **Step 2: Create __main__.py entry point**

Create `agent/__main__.py`:

```python
"""Entry point: python -m agent [--supervised] [--max-iter N]"""
import argparse

from agent.loop import run_loop


def main():
    ap = argparse.ArgumentParser(
        description='Run the ML experiment agent')
    ap.add_argument('--supervised', action='store_true',
                    help='Pause after each iteration for human approval')
    ap.add_argument('--max-iter', type=int, default=100,
                    help='Maximum iterations before stopping (default: 100)')
    args = ap.parse_args()
    run_loop(supervised=args.supervised, max_iter=args.max_iter)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Verify the module is importable**

Run:
```bash
python -c "from agent.loop import run_loop; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Verify CLI help works**

Run:
```bash
python -m agent --help
```

Expected: help output showing `--supervised` and `--max-iter` flags.

- [ ] **Step 5: Commit**

```bash
git add agent/loop.py agent/__main__.py
git commit -m "agent: add main loop with autonomous and supervised modes"
```

---

### Task 5: Integration Validation

**Files:**
- No new files

**Interfaces:**
- Consumes: everything from Tasks 1-4
- Produces: confidence that the system works end-to-end

- [ ] **Step 1: Run all safety tests**

Run:
```bash
python tests/test_tools.py
```

Expected: `All tool safety tests passed.`

- [ ] **Step 2: Dry run in supervised mode (1 iteration)**

Run:
```bash
python -m agent --supervised --max-iter 1
```

Expected:
- Agent starts, reads ledger, reads best solution
- Proposes a hypothesis
- Writes a new solution file to solutions/
- Runs the experiment via the harness (~30-50s)
- Prints the result with metrics and verdict
- Pauses for input
- Type `q` to stop

Watch for:
- Does the agent propose something reasonable (likely loss function change)?
- Does the solution file actually appear in solutions/?
- Does the harness score it and log to LEDGER.md?
- Is the result recorded in logs/iterations/?

- [ ] **Step 3: Verify the experiment was logged**

Run:
```bash
cat LEDGER.md
```

Expected: a new row with the agent's experiment, verdict, and `agent` in the `by` column.

- [ ] **Step 4: Verify solution path restriction works**

Check that no files were created outside `solutions/`.

- [ ] **Step 5: Clean up test artifacts if needed**

Remove any test solution files the agent created if they're not worth keeping:
```bash
# Only if the solution scored worse than baseline
rm solutions/002_*.py  # or whatever it created
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Agent writes solutions that crash | HIGH (expected) | Harness catches everything, returns error, agent self-corrects |
| Agent ignores prompt constraints | MEDIUM | Constraints enforced in code (path restriction, search cap) |
| Agent loops on the same failing idea | MEDIUM | Source hash duplicate detection in harness |
| OpenAI API rate limits | LOW | Retry with backoff on 429 in loop |
| Token cost spirals | LOW | Max iteration cap + message history trimming |
| OpenAI Responses API unavailable for web search | LOW | Wrapped in try/except, returns error, agent continues without search |

## Acceptance

- [ ] All safety tests pass (`python tests/test_tools.py`)
- [ ] `python -m agent --help` shows both flags
- [ ] Supervised dry run completes 1 iteration without crashing
- [ ] Experiment logged to LEDGER.md with by=agent
- [ ] No files created outside solutions/
- [ ] Web search cap enforced (only 1 per iteration)
