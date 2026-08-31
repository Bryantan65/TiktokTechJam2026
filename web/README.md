# The agent console

A local page for driving and replaying the agent. Stdlib only — no Flask, no
node, no build step, nothing added to `requirements.txt`.

```
.venv/Scripts/python.exe -m web.server
```

then open <http://127.0.0.1:8765>. `WEB_PORT` overrides the port.

Localhost only by design: it launches processes and edits the prompt, so it
binds `127.0.0.1` and is never exposed.

---

## The three tabs

**Runs** — replay any recorded run. Every iteration carries a `parent`, so a run
is a search tree rather than a list; press Play and it grows node by node,
coloured by verdict, with the best-so-far frontier highlighted. Clicking a node
shows that iteration's hypothesis, GAUC and nDCG@5 with the delta against its
parent, per-seed spread, the error text if it failed, and the code diff from
`diffs/NNNN.diff`. Where the two metrics moved in *opposite* directions the
panel says so — that is the thing the merged `primary` column hides, and the
project's best result came from noticing it.

**Agent** — edit the system prompt. Saves to `agent/prompt_override.txt`, never
to `agent/prompt.py`, so a bad edit is a bad string and can never be a Python
syntax error. The template is validated against every dataset config before it
is written; an unbalanced brace is refused inline instead of crashing a run 40
minutes in. Deleting the override (the *Revert to shipped* button) restores the
shipped prompt exactly.

**Launch** — settings, then a preflight screen, then the run. The settings map
one-to-one onto the environment variables and CLI flags the agent already
reads; there is no console-only setting. The preflight screen shows the real
assembled prompt — it calls the agent's own `system_prompt()` and
`build_user_message()`, so it cannot drift from what actually runs — plus the
budget ceilings, the tool list, and a warning when the prompt is an override or
the dataset is a bonus benchmark.

---

## What it writes and runs

It reads the run records read-only. It writes exactly one file,
`agent/prompt_override.txt`, and executes exactly one command:

```
python -m agent --run-id <name>-N --dataset <ds> --max-iter <n>
```

which is what a human would type. A run started here produces an ordinary run
folder that `harness/ledger.py`, `harness/gendiffs.py` and every other tool
already understand — there is no separate code path and no console-only
artifact.

The live view renders from the run's own `events.jsonl` and `NNNN.json` as they
land on disk, not from parsed stdout. If the two ever disagree, the files win,
because the files are the deliverable.

## Run identity

`run_start` in `events.jsonl` now records `prompt_source`, `prompt_hash` and
`prompt_chars` alongside the model and the budget ceilings. A run made with an
edited prompt says so permanently in its own record, so the log stays evidence
about a specific agent rather than about whatever the prompt happened to be
that day. Runs made before this change simply lack the fields.

## Known rough edges

- Killing the server does not kill an agent already running: the child is
  reparented and keeps writing its run folder. Use **Stop run** in the page
  (which kills the process tree) before shutting the server down.
- One run at a time. Starting a second while one is live is refused.
- `logs-1k/record-run-2` is flagged **label leak** in the run list. It reaches
  0.997 because the agent reconstructed `long_view` from raw columns rather
  than modelling anything. It is kept visible on purpose; it is not a result.
