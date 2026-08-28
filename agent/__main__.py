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
    ap.add_argument('--run-name', default='run',
                    help='Run folder prefix (default: "run"). '
                         'Auto-numbers: --run-name record-run -> record-run-4')
    ap.add_argument('--run-id', default=None,
                    help='Exact run folder name, skipping auto-numbering. '
                         'e.g. --run-id record-run-5')
    args = ap.parse_args()
    run_loop(supervised=args.supervised, max_iter=args.max_iter,
             run_name=args.run_name, run_id=args.run_id)


if __name__ == '__main__':
    main()
