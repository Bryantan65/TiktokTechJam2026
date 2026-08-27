"""Main agent loop: call LLM -> execute tools -> log -> repeat."""
import json
import os
import sys
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402

from agent.tools import TOOL_SCHEMAS, TOOL_DISPATCH, do_web_search  # noqa: E402
from agent.prompt import system_prompt, build_user_message  # noqa: E402

MAX_TOOL_ROUNDS = 20

# Model and pricing are configurable because model IDs and prices change often
# and a stale hardcoded value is worse than none: token spend is a reported
# deliverable (15% of the grade), so a wrong price silently misreports it.
# Discover valid ids with:  client.models.list()
# Set in .env:  AGENT_MODEL, AGENT_INPUT_COST_PER_M, AGENT_OUTPUT_COST_PER_M
MODEL = os.environ.get('AGENT_MODEL', 'gpt-4o')
INPUT_COST_PER_M = float(os.environ.get('AGENT_INPUT_COST_PER_M', 2.50))
OUTPUT_COST_PER_M = float(os.environ.get('AGENT_OUTPUT_COST_PER_M', 10.00))


def _role(m):
    """Messages are a mix of plain dicts and SDK objects."""
    return m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)


def _truncate(messages, keep_last=20):
    """Trim history without splitting a tool-call sequence.

    The API requires every 'tool' message to be preceded by the assistant
    message that requested it. Slicing blindly (messages[-20:]) can orphan one,
    and the request is then rejected outright - at a random iteration, probably
    unattended. Cutting only at a 'user' message is safe: that is the start of
    an iteration, so no tool sequence spans the boundary.
    """
    if len(messages) <= keep_last + 1:
        return messages
    for i in range(len(messages) - keep_last, len(messages)):
        if _role(messages[i]) == 'user':
            return messages[:1] + messages[i:]
    return messages          # no safe cut point; keep everything this round


class _Spinner:
    """Animated spinner with elapsed time for long-running tool calls."""

    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, label: str):
        self._label = label
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        elapsed = time.time() - self._t0
        sys.stdout.write(f'\r    {self._label} done ({elapsed:.0f}s)'
                         '                    \n')
        sys.stdout.flush()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(
                f'\r    {frame} {self._label} [{elapsed:.0f}s]'
                '          ')
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.15)


class TokenTracker:
    """Accumulates token usage and estimated cost across API calls."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.iter_input = 0
        self.iter_output = 0
        self.api_calls = 0

    def reset_iteration(self):
        self.iter_input = 0
        self.iter_output = 0

    def record(self, usage):
        if usage is None:
            return
        self.iter_input += usage.prompt_tokens
        self.iter_output += usage.completion_tokens
        self.total_input += usage.prompt_tokens
        self.total_output += usage.completion_tokens
        self.api_calls += 1

    def iter_cost(self) -> float:
        return (self.iter_input * INPUT_COST_PER_M
                + self.iter_output * OUTPUT_COST_PER_M) / 1_000_000

    def total_cost(self) -> float:
        return (self.total_input * INPUT_COST_PER_M
                + self.total_output * OUTPUT_COST_PER_M) / 1_000_000

    def iter_summary(self) -> str:
        return (f'tokens: {self.iter_input:,}in/{self.iter_output:,}out '
                f'(${self.iter_cost():.3f})')

    def total_summary(self) -> str:
        return (f'cumulative: {self.total_input:,}in/{self.total_output:,}out '
                f'(${self.total_cost():.3f}, {self.api_calls} API calls)')


def _execute_tool(client, name: str, args: dict,
                  search_count: int) -> tuple[str, int]:
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
            label = f'{name} {_summarize_args(name, args)}'
            is_long = name == 'run_experiment'
            if is_long:
                with _Spinner(label):
                    result = TOOL_DISPATCH[name](**args)
            else:
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
        h = args.get('hypothesis', '')
        return f"{args.get('solution', '')} — {h[:80]}"
    return ''


def run_loop(supervised: bool = False, max_iter: int = 100) -> None:
    load_dotenv(os.path.join(ROOT, '.env'))
    client = OpenAI()
    tokens = TokenTracker()

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
        ran_iterations = []
        tokens.reset_iteration()
        t0 = time.time()
        print(f'\n--- Iteration {iteration} ---')

        user_msg = build_user_message()
        messages.append({'role': 'user', 'content': user_msg})

        tool_rounds = 0
        while tool_rounds < MAX_TOOL_ROUNDS:
            tool_rounds += 1
            try:
                with _Spinner('thinking'):
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

            tokens.record(response.usage)
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
                    if tc.function.name == 'run_experiment':
                        try:
                            ran_iterations.append(json.loads(result)['iteration'])
                        except (ValueError, KeyError, TypeError):
                            pass
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': result if isinstance(result, str)
                                   else json.dumps(result),
                    })
            else:
                break

        elapsed = time.time() - t0

        # Fold this iteration's LLM cost into the experiment record(s) it
        # produced. Printing alone would lose it when the terminal closes, and
        # total token spend is a required submission deliverable.
        if ran_iterations:
            share_in = tokens.iter_input // len(ran_iterations)
            share_out = tokens.iter_output // len(ran_iterations)
            share_cost = tokens.iter_cost() / len(ran_iterations)
            for it in ran_iterations:
                ledger.annotate(it,
                                tokens_in=share_in,
                                tokens_out=share_out,
                                cost_usd=round(share_cost, 6),
                                model=MODEL,
                                agent_iteration=iteration,
                                wall_seconds=round(elapsed, 1))

        print(f'  [{elapsed:.0f}s elapsed] {tokens.iter_summary()}')
        print(f'  {tokens.total_summary()}')

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

        if len(messages) > 40:
            messages = _truncate(messages, keep_last=20)

    print('\n=== Agent loop finished ===')
    print(f'  {tokens.total_summary()}')
    t = ledger.totals()
    print(f'  logged totals: {t["tokens_in"]:,}in/{t["tokens_out"]:,}out '
          f'(${t["cost_usd"]:.3f}) over {t["iterations"]} experiments, '
          f'{t["compute_seconds"]:.0f}s compute')
    best_rec = ledger.best()
    if best_rec:
        print(f'  Best result: valid primary {best_rec["valid_primary"]} '
              f'({best_rec["solution"]})')
    else:
        print('  No successful experiments.')
