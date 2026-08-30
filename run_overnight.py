"""Auto-restart wrapper for the ML experiment agent.

Runs the agent loop and restarts it if it crashes (exit code != 0).
Stops cleanly on a normal finish (converged / budget / max_iter).

Usage:
    python run_overnight.py --dataset 1k --run-name record-run [agent args...]

All flags are forwarded to `python -m agent` unchanged, so --dataset,
--run-name, --run-id, --max-iter, --supervised all work as normal.

Restart behaviour:
  - On crash (exit != 0): waits RESTART_DELAY seconds, then restarts.
  - On normal finish (exit 0): stops.
  - After MAX_RESTARTS consecutive crashes: gives up and exits.
  - RESTART_DELAY doubles after each crash (30s, 60s, 120s ...) up to 5 min,
    so a persistent error doesn't burn the whole API budget in a tight loop.
"""
import subprocess
import sys
import time
from datetime import datetime

MAX_RESTARTS = 5
RESTART_DELAY = 30       # seconds; doubles each attempt, capped at 300


def _stamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    agent_args = sys.argv[1:]   # forward everything verbatim
    delay = RESTART_DELAY

    for attempt in range(MAX_RESTARTS + 1):
        if attempt > 0:
            print(f'\n[{_stamp()}] Restart {attempt}/{MAX_RESTARTS} '
                  f'in {delay}s ...', flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)

        print(f'\n[{_stamp()}] Starting agent '
              f'(attempt {attempt + 1}/{MAX_RESTARTS + 1})', flush=True)
        print(f'  Command: python -m agent {" ".join(agent_args)}', flush=True)

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'agent'] + agent_args,
                # Don't capture stdout/stderr — let them flow to the terminal
                # so the user (or a log redirect) sees everything in real time.
            )
        except KeyboardInterrupt:
            print(f'\n[{_stamp()}] Interrupted by user. Stopping.', flush=True)
            sys.exit(0)

        code = result.returncode
        if code == 0:
            print(f'\n[{_stamp()}] Agent finished normally (exit 0). Done.',
                  flush=True)
            sys.exit(0)

        print(f'\n[{_stamp()}] Agent exited with code {code} '
              f'(crash or unhandled error).', flush=True)

    print(f'\n[{_stamp()}] Gave up after {MAX_RESTARTS} restarts. '
          f'Check the logs for the error.', flush=True)
    sys.exit(1)


if __name__ == '__main__':
    main()
