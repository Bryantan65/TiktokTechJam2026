"""Main agent loop: call LLM -> execute tools -> log -> repeat."""
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback

from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Must run before any import that reads os.environ at module level
# (ledger.py reads HARNESS_MIN_SCORED, loop.py reads AGENT_*).
load_dotenv(os.path.join(ROOT, '.env'))

sys.path.insert(0, os.path.join(ROOT, 'harness'))
import ledger  # noqa: E402

from agent.tools import TOOL_SCHEMAS, TOOL_DISPATCH, do_web_search  # noqa: E402
from agent.prompt import system_prompt, build_user_message  # noqa: E402

MAX_TOOL_ROUNDS = 20

# How many completed iterations of chat history to carry forward. The ledger is
# the agent's real memory - every iteration's user message is rebuilt with the
# current ledger and the current best solution, so older turns are duplicates
# that get resent on every tool round. Keeping one gives continuity ("that
# didn't work, try the variant") without paying for the whole run.
HISTORY_ITERATIONS = int(os.environ.get('AGENT_HISTORY_ITERATIONS', 1))

# Model and pricing are configurable because model IDs and prices change often
# and a stale hardcoded value is worse than none: token spend is a reported
# deliverable (15% of the grade), so a wrong price silently misreports it.
# Discover valid ids with:  client.models.list()
# Set in .env:  AGENT_MODEL, AGENT_INPUT_COST_PER_M, AGENT_OUTPUT_COST_PER_M
MODEL = os.environ.get('AGENT_MODEL', 'gpt-4o')
INPUT_COST_PER_M = float(os.environ.get('AGENT_INPUT_COST_PER_M', 2.50))
OUTPUT_COST_PER_M = float(os.environ.get('AGENT_OUTPUT_COST_PER_M', 10.00))
# Repeated prompt prefixes are served from cache at a large discount (OpenAI:
# 10x cheaper). The loop resends the whole conversation every tool round, so
# most input tokens are cache hits - billing them at full price overstates the
# reported spend, which is a graded deliverable. Defaults to a tenth of input.
CACHED_COST_PER_M = float(os.environ.get('AGENT_CACHED_COST_PER_M',
                                         INPUT_COST_PER_M / 10))

# The gpt-5.6 family (sol/terra/luna) refuses function tools on
# /v1/chat/completions unless reasoning_effort is 'none':
#   "Function tools with reasoning_effort are not supported ... use
#    /v1/responses or set reasoning_effort to 'none'."
# Set AGENT_REASONING_EFFORT=none to use them here. Left unset for models that
# accept tools natively (gpt-5.5 and earlier), which keeps reasoning on.
REASONING_EFFORT = os.environ.get('AGENT_REASONING_EFFORT') or None

# Backstops, not the intended terminating condition - a run should end on
# converged(), which is the rule the task specification names. What these
# actually guard against is a crash loop: converged() counts only experiments
# that SCORED, so an agent whose solutions all fail never converges and would
# otherwise run forever. MAX_EXPERIMENTS counts every ledger row, errors
# included, so it bounds that case. Set high enough that triggering it means
# something is genuinely wrong (80 experiments is ~$11 and ~2.3 hours).
MAX_EXPERIMENTS = int(os.environ.get('AGENT_MAX_EXPERIMENTS', 80))
MAX_COST_USD = float(os.environ.get('AGENT_MAX_COST_USD', 15.0))


# Transient API failures. A 500, a gateway timeout or a dropped connection says
# nothing about the experiment - retrying is the correct response, and over a
# 40-experiment run at least one will happen. Previously only rate limits were
# retried and everything else abandoned the iteration.
MAX_API_RETRIES = int(os.environ.get('AGENT_MAX_API_RETRIES', 5))
_RETRYABLE = ('rate_limit', '429', '500', '502', '503', '504',
              'timeout', 'timed out', 'connection', 'overloaded',
              'internal server error', 'bad gateway', 'service unavailable',
              'temporarily unavailable', 'apiconnectionerror')
# Retrying these just burns the budget on the same failure: a bad key, a
# malformed request or a prompt over the context limit will fail identically
# every time.
_FATAL = ('invalid_api_key', 'authentication', 'permission',
          'context_length', 'maximum context', 'invalid_request_error',
          'model_not_found', 'insufficient_quota', 'billing')


def _classify_error(exc):
    """'fatal' or 'retryable'. Checked fatal-first: an auth failure can mention
    'connection' in its message and must not be treated as transient."""
    text = ('%s %s' % (type(exc).__name__, exc)).lower()
    if any(k in text for k in _FATAL):
        return 'fatal'
    if any(k in text for k in _RETRYABLE):
        return 'retryable'
    return 'retryable'      # unknown: one run costs pennies, a lost run costs the night


def _backoff(attempt):
    """Exponential with jitter, capped. Jitter matters when a rate limit is
    shared - identical sleeps mean every retry collides again."""
    return min(60.0, 2.0 ** attempt) * (0.75 + 0.5 * random.random())


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


def _compact(messages, keep_iterations=HISTORY_ITERATIONS):
    """Drop all but the last n iterations of history.

    Each iteration begins with a 'user' message, so user messages are the
    iteration boundaries and cutting there never orphans a 'tool' message from
    the assistant turn that requested it (see _truncate).
    """
    if keep_iterations < 0:
        return messages
    if keep_iterations == 0:
        return messages[:1]          # not bounds[-0:], which is the whole list
    bounds = [i for i, m in enumerate(messages) if _role(m) == 'user']
    if len(bounds) <= keep_iterations:
        return messages
    return messages[:1] + messages[bounds[-keep_iterations]:]


def _console_supports_unicode():
    """Windows consoles often use cp1252, which cannot encode braille frames.
    Writing them raises UnicodeEncodeError inside the spinner thread and spams
    tracebacks through the whole run."""
    try:
        '⠋'.encode(sys.stdout.encoding or 'ascii')
        return True
    except (UnicodeEncodeError, LookupError, AttributeError):
        return False


class _Spinner:
    """Animated spinner with elapsed time for long-running tool calls."""

    FRAMES = (['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
              if _console_supports_unicode() else ['|', '/', '-', '\\'])

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
        try:
            done = self._fit(f'    {self._label} done ({elapsed:.0f}s)')
            sys.stdout.write('\r' + done.ljust(self._width()) + '\n')
            sys.stdout.flush()
        except Exception:          # noqa: BLE001  cosmetic only
            pass

    @staticmethod
    def _width():
        try:
            return max(20, shutil.get_terminal_size().columns - 1)
        except Exception:          # noqa: BLE001  not a tty, or no size
            return 79

    @classmethod
    def _fit(cls, line):
        """Clamp to the terminal width, leaving one column spare.

        A spinner redraws with '\\r', which returns to the start of the current
        VISUAL line. Once a write exceeds the terminal width the terminal wraps,
        '\\r' lands on the wrapped remainder, and every frame leaves the first
        half behind - the animation becomes a wall of near-identical lines. Seen
        with run_experiment, whose label carries 80 characters of hypothesis and
        reached 138 columns.

        Truncating from the left of the label would hide the filename, which is
        the part worth reading, so the tail is what goes.
        """
        limit = cls._width()
        return line if len(line) <= limit else line[:limit - 1] + '>'

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            try:
                elapsed = time.time() - self._t0
                frame = self.FRAMES[i % len(self.FRAMES)]
                line = f'    {frame} {self._label} [{elapsed:.0f}s]'
                # Pad to the full width so a shorter frame cannot leave
                # characters from a longer previous one on screen.
                sys.stdout.write('\r' + self._fit(line).ljust(self._width()))
                sys.stdout.flush()
            except Exception:      # noqa: BLE001  cosmetic only - never kill a run
                return
            i += 1
            self._stop.wait(0.15)


class TokenTracker:
    """Accumulates token usage and estimated cost across API calls."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.total_cached = 0
        self.iter_input = 0
        self.iter_output = 0
        self.iter_cached = 0
        self.api_calls = 0

    def reset_iteration(self):
        self.iter_input = 0
        self.iter_output = 0
        self.iter_cached = 0

    @staticmethod
    def _cached(usage):
        details = getattr(usage, 'prompt_tokens_details', None)
        return getattr(details, 'cached_tokens', 0) or 0

    def record(self, usage):
        if usage is None:
            return
        cached = min(self._cached(usage), usage.prompt_tokens)
        self.iter_input += usage.prompt_tokens
        self.iter_output += usage.completion_tokens
        self.iter_cached += cached
        self.total_input += usage.prompt_tokens
        self.total_output += usage.completion_tokens
        self.total_cached += cached
        self.api_calls += 1

    @staticmethod
    def _cost(inp, cached, out) -> float:
        return ((inp - cached) * INPUT_COST_PER_M
                + cached * CACHED_COST_PER_M
                + out * OUTPUT_COST_PER_M) / 1_000_000

    def iter_cost(self) -> float:
        return self._cost(self.iter_input, self.iter_cached, self.iter_output)

    def total_cost(self) -> float:
        return self._cost(self.total_input, self.total_cached, self.total_output)

    def iter_summary(self) -> str:
        hit = 100 * self.iter_cached / self.iter_input if self.iter_input else 0
        return (f'tokens: {self.iter_input:,}in ({hit:.0f}% cached)'
                f'/{self.iter_output:,}out (${self.iter_cost():.3f})')

    def total_summary(self) -> str:
        hit = 100 * self.total_cached / self.total_input if self.total_input else 0
        return (f'cumulative: {self.total_input:,}in ({hit:.0f}% cached)'
                f'/{self.total_output:,}out '
                f'(${self.total_cost():.3f}, {self.api_calls} API calls)')


def _execute_tool(client, name: str, args: dict,
                  search_count: int, budget=None) -> tuple[str, int]:
    if name == 'run_experiment' and budget is not None:
        stop = budget()
        if stop:
            # Refused as a normal tool result, so the agent sees a message and
            # can wrap up rather than the loop dying mid-turn.
            return json.dumps({'error': stop, 'status': 'budget_exhausted'}), search_count

    if name == 'web_search':
        if search_count >= 1:
            return json.dumps({
                'error': 'search limit reached for this iteration (max 1)'
            }), search_count
        search_count += 1
        query = args.get('query', '')
        print(f'    [web_search] {query}')
        result = do_web_search(client, query)
        # Log the query and result to the run log. The only rule for provenance
        # was a prompt line asking the agent to copy the citation URL into its
        # hypothesis, which nothing enforces - and history compaction now drops
        # the tool message after one iteration, so an uncited finding is gone.
        # "What the agent chose to try and why" is what Innovation is scored on;
        # a search it acted on is part of the why.
        ledger.log_event('web_search', query, result=str(result)[:2000])
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
        # ASCII separator: the label is written from the spinner thread,
        # and an em-dash raises UnicodeEncodeError on a cp1252 console,
        # which the thread's except clause swallows - killing the spinner
        # silently for the rest of the run.
        return f"{args.get('solution', '')} - {h[:80]}"
    return ''


def _setup_run_dir(run_name: str) -> str:
    """Create a new per-run folder and redirect all output paths to it."""
    from agent.tools import init_solutions_dir
    run_dir = ledger.next_run_dir(run_name)
    ledger.init_run_dir(run_dir)
    ledger.setup_control_row(run_dir)
    sol_dir = os.path.join(run_dir, 'solutions')
    init_solutions_dir(sol_dir)
    return run_dir


def run_loop(supervised: bool = False, max_iter: int = 100,
             run_name: str = 'run') -> None:
    run_dir = _setup_run_dir(run_name)
    print(f'=== Run folder: {os.path.relpath(run_dir, ROOT)}/ ===')

    client = OpenAI()          # .env is loaded at import, above
    tokens = TokenTracker()

    messages = [{'role': 'system', 'content': system_prompt()}]
    iteration = 0
    iter_events: list[dict] = []

    def _event(kind, detail, **extra):
        """Record an error or recovery event to disk AND to this iteration.

        Two destinations on purpose. logs/events.jsonl is the chronological run
        log, which is the only place a failure between experiments can live.
        iter_events is folded into the experiment record below, so a judge
        reading a single iteration sees what went wrong during it without
        cross-referencing timestamps.
        """
        rec = ledger.log_event(kind, detail, iteration=iteration, **extra)
        iter_events.append(rec)
        print('  [%s] %s' % (kind, rec['detail'][:200]))
        return rec

    def _budget_check():
        """Reason to stop running experiments, or None. Checked before each."""
        done = ledger.totals()['iterations']
        if done >= MAX_EXPERIMENTS:
            return ('experiment budget exhausted: %d of %d run. Stop proposing '
                    'experiments and summarise what you found.'
                    % (done, MAX_EXPERIMENTS))
        if tokens.total_cost() >= MAX_COST_USD:
            return ('cost budget exhausted: $%.2f of $%.2f spent. Stop '
                    'proposing experiments and summarise what you found.'
                    % (tokens.total_cost(), MAX_COST_USD))
        return None

    print(f'=== Agent loop started (supervised={supervised}, '
          f'max_iter={max_iter}, max_experiments={MAX_EXPERIMENTS}, '
          f'max_cost=${MAX_COST_USD:.2f}) ===\n')
    ledger.log_event('run_start', 'agent loop started', model=MODEL,
                     max_iter=max_iter, max_experiments=MAX_EXPERIMENTS,
                     max_cost_usd=MAX_COST_USD, supervised=supervised,
                     ledger_rows_at_start=ledger.totals()['iterations'])

    stop_reason = 'max_iter reached'
    try:
        while iteration < max_iter:
            stop = _budget_check()
            if stop:
                print('\nStopping: %s' % stop)
                stop_reason = 'budget: %s' % stop
                ledger.log_event('budget_stop', stop, iteration=iteration)
                break
            if ledger.converged():
                st = ledger.convergence_status()
                print('\nConverged: best improved by only %+.6f across the '
                      'last %d scored experiments (need %.3f).'
                      % (st['window_improvement'] or 0.0, st['n'],
                         st['epsilon']))
                stop_reason = 'converged'
                # The terminating condition the task specification asks for, so it
                # belongs in the run log rather than only on the terminal.
                ledger.log_event('converged',
                                 'best improved by %+.6f across the last %d '
                                 'scored experiments; epsilon is %s'
                                 % (st['window_improvement'] or 0.0, st['n'],
                                    st['epsilon']),
                                 iteration=iteration,
                                 window_improvement=st['window_improvement'])
                break

            iteration += 1
            search_count = 0
            ran_iterations = []
            fatal_api = None
            iter_events.clear()
            tokens.reset_iteration()
            t0 = time.time()
            print(f'\n--- Iteration {iteration} ---')

            # Before adding this iteration's message, drop the stale ones. The new
            # message carries a current ledger and current best solution, so older
            # turns are duplicates that would be resent on every tool round below.
            messages = _compact(messages)
            user_msg = build_user_message()
            messages.append({'role': 'user', 'content': user_msg})

            tool_rounds = 0
            while tool_rounds < MAX_TOOL_ROUNDS:
                tool_rounds += 1
                # No temperature: the gpt-5.6 family rejects anything but the
                # default (400 "does not support 0.7 with this model"). Diversity
                # across iterations comes from the ledger changing, not sampling.
                kwargs = {}
                if REASONING_EFFORT:
                    kwargs['reasoning_effort'] = REASONING_EFFORT

                response = None
                fatal_api = None
                for attempt in range(MAX_API_RETRIES + 1):
                    try:
                        with _Spinner('thinking' if attempt == 0
                                      else f'retrying ({attempt}/{MAX_API_RETRIES})'):
                            response = client.chat.completions.create(
                                model=MODEL, messages=messages,
                                tools=TOOL_SCHEMAS, **kwargs)
                        if attempt:
                            _event('api_recovered',
                                   'call succeeded on attempt %d' % (attempt + 1),
                                   attempts=attempt + 1)
                        break
                    except Exception as e:                     # noqa: BLE001
                        kind = _classify_error(e)
                        if kind == 'fatal':
                            # A bad key, a malformed request or an over-length
                            # prompt fails identically forever. Retrying wastes
                            # the budget; continuing to the next iteration
                            # wastes every remaining one.
                            fatal_api = '%s: %s' % (type(e).__name__, e)
                            _event('api_fatal', e, error_type=type(e).__name__)
                            break
                        if attempt >= MAX_API_RETRIES:
                            _event('api_gave_up', e, error_type=type(e).__name__,
                                   attempts=attempt + 1)
                            break
                        delay = _backoff(attempt)
                        _event('api_retry', e, error_type=type(e).__name__,
                               attempt=attempt + 1, retry_in_seconds=round(delay, 1))
                        time.sleep(delay)

                if response is None:
                    # Out of retries, or fatal. Abandoning the iteration is not
                    # abandoning the run: the ledger holds everything learned so
                    # far, so the next iteration rebuilds from disk and carries
                    # on. A fatal error is different - it will recur every time.
                    print('  Abandoning iteration %d after API failure.'
                          % iteration)
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
                            client, tc.function.name, args, search_count,
                            budget=_budget_check)
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

            if fatal_api:
                # Not transient: a bad key, a malformed request or an
                # over-length prompt fails identically on every remaining
                # iteration. Stop the run instead of burning the budget
                # reproducing the same error 39 more times.
                stop_reason = 'fatal API error: %s' % fatal_api
                print('\n  Fatal API error; stopping the run.')
                break

            # Fold this iteration's LLM cost into the experiment record(s) it
            # produced. Printing alone would lose it when the terminal closes, and
            # total token spend is a required submission deliverable.
            if ran_iterations:
                share_in = tokens.iter_input // len(ran_iterations)
                share_out = tokens.iter_output // len(ran_iterations)
                share_cached = tokens.iter_cached // len(ran_iterations)
                share_cost = tokens.iter_cost() / len(ran_iterations)
                for it in ran_iterations:
                    ledger.annotate(it,
                                    tokens_in=share_in,
                                    tokens_out=share_out,
                                    tokens_cached=share_cached,
                                    cost_usd=round(share_cost, 6),
                                    model=MODEL,
                                    agent_iteration=iteration,
                                    wall_seconds=round(elapsed, 1),
                                    # Section 2.4: each iteration records "any
                                    # error / recovery events". Empty list is
                                    # meaningful - it says the iteration was clean.
                                    recovery_events=list(iter_events))

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

    except KeyboardInterrupt:
        # A deliberate stop is not a crash, but it does end the run,
        # and an autonomy claim has to be able to say which happened.
        stop_reason = 'interrupted by user (KeyboardInterrupt)'
        print('\n  Interrupted.')
        ledger.log_event('interrupted', stop_reason,
                         iteration=iteration)
    except Exception as exc:                       # noqa: BLE001
        # Log the crash before re-raising so the run log records how
        # the run died. Without this the log simply stops, which is
        # indistinguishable from a closed terminal.
        stop_reason = 'crashed: %s: %s' % (type(exc).__name__, exc)
        ledger.log_event('crash', stop_reason, iteration=iteration,
                         error_type=type(exc).__name__,
                         traceback=traceback.format_exc()[-2000:])
        _finish(tokens, stop_reason)
        raise

    _finish(tokens, stop_reason)


def _finish(tokens, stop_reason):
    """Closing summary, printed and logged.

    Reached from a normal exit, a Ctrl-C, or an unhandled exception, so the run
    log always ends with a record of how the run stopped. A run log that simply
    stops is indistinguishable from one whose terminal was closed.
    """
    print('\n=== Agent loop finished ===')
    print(f'  stopped because: {stop_reason}')
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

    ledger.log_event('run_end', stop_reason,
                     tokens_in=tokens.total_input,
                     tokens_out=tokens.total_output,
                     tokens_cached=tokens.total_cached,
                     cost_usd=round(tokens.total_cost(), 4),
                     api_calls=tokens.api_calls,
                     experiments=t['iterations'],
                     best_valid_primary=(best_rec or {}).get('valid_primary'))
