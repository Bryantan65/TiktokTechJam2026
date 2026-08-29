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
BASE_SOLUTIONS_DIR = os.path.join(ROOT, 'solutions')
SOLUTIONS_DIR = BASE_SOLUTIONS_DIR

sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402


def init_solutions_dir(run_solutions_dir: str):
    """Redirect solution writes to a per-run folder.

    Reads still check both the run's solutions/ and root solutions/
    (for base solutions like 000_baseline.py, 001_torch_fm.py).
    """
    global SOLUTIONS_DIR
    SOLUTIONS_DIR = run_solutions_dir
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)


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


def _resolve_solution_read(path: str):
    """Find a solution file, checking run solutions/ then root solutions/."""
    for d in (SOLUTIONS_DIR, BASE_SOLUTIONS_DIR):
        full = os.path.join(d, path)
        real = os.path.realpath(full)
        if real.startswith(os.path.realpath(d) + os.sep) and os.path.isfile(real):
            return real
    return None


def read_solution(path: str) -> str:
    real = _resolve_solution_read(path)
    if real is None:
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

    if os.path.exists(full):
        # Overwriting silently loses the code behind an already-logged result,
        # and the ledger would then point at a file that no longer matches.
        return json.dumps({'error': f'{filename} already exists. Pick a new '
                                    f'number/name; never overwrite a solution '
                                    f'that has been run.'})

    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(code)
    return json.dumps({'status': 'ok', 'path': f'solutions/{filename}'})


def run_experiment(solution: str, hypothesis: str = '',
                   parent: str | None = None, split: str = 'valid') -> str:
    # 'dev' is the train-only holdout (harness/devdata.py); 'valid' is the real
    # thing. Anything else is refused here as well as in the harness - the model
    # cannot ask to be scored on test, and belt-and-braces is cheap for the one
    # guarantee the whole result rests on.
    if split not in ('valid', 'dev'):
        return json.dumps({'status': 'error',
                           'error': "split must be 'valid' or 'dev', got %r" % split})
    sys.path.insert(0, os.path.join(ROOT, 'harness'))
    from run import run_experiment as _run  # noqa: E402
    full = os.path.join(SOLUTIONS_DIR, solution)
    result = _run(full, hypothesis=hypothesis, parent=parent, by='agent',
                  split=split)
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
                    'split': {
                        'type': 'string',
                        'enum': ['valid', 'dev'],
                        'description': (
                            "'valid' (default) is the real experiment: it "
                            'counts toward convergence and can become the best '
                            'solution. "dev" screens against a train-only '
                            'holdout instead - cheaper to be wrong on, does not '
                            'count toward convergence, and its number is not '
                            'comparable to valid. Your solution must handle '
                            "--split dev; see 001_torch_fm.py."
                        ),
                    },
                    'hypothesis': {
                        'type': 'string',
                        'description': (
                            'What this tests, starting with the action: '
                            '"draft <direction>", "improve <n>" or "debug <n>". '
                            'One or two sentences after that.'
                        ),
                    },
                    'parent': {
                        'type': 'string',
                        'description': (
                            'Iteration number this expands. REQUIRED unless '
                            'this is the very first experiment. For improve and '
                            'debug it is the node being refined; for a draft it '
                            'is the node you branched the code from. Every '
                            'experiment branching from the same parent means '
                            'nothing was ever refined.'
                        ),
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
                'techniques. Use this on the FIRST iteration of any direction '
                'you have not tried before - a standard method implemented from '
                'memory is easy to get subtly wrong, and a wrong implementation '
                'records a false negative against a direction that works. Do '
                'not use it to tune a variant of something already working. '
                'Limited to 1 search per iteration. Put the citation URL in '
                'your hypothesis - nothing else records it.'
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
        kw['solution'], kw.get('hypothesis', ''), kw.get('parent'),
        kw.get('split', 'valid')),
}
