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
