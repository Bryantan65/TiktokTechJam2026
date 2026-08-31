# Architecture

How the agent, the harness and the run record fit together. Every diagram below
is drawn from the code named in it, not from a design document — if the two
disagree, the code is right and this file is stale.

Mermaid renders natively on GitHub and in Devpost's markdown, so these blocks
can be pasted straight into a writeup.

---

## 1. The whole system

The important line in this picture is the one between `agent/` and `harness/`.
The agent writes solutions and asks for them to be scored; it never holds a
label, never computes a metric, and never sees the test split. That separation
is mechanical, not prompted.

```mermaid
flowchart LR
    subgraph agent["agent/ — what the model controls"]
        direction TB
        P["prompt.py<br/>system prompt + ledger table + UCT ranking"]
        L["loop.py<br/>ask → tools → log → repeat"]
        T["tools.py<br/>5 tools, and the jail"]
        P --> L
        L <--> T
    end

    subgraph harness["harness/ — what the agent cannot reach"]
        direction TB
        R["run.py<br/>3 seeds, scores, never raises"]
        D["data.py<br/>picks the KuaiRand variant"]
        LG["ledger.py<br/>verdict, convergence, records"]
        E["evaluate.py<br/>official, unmodified"]
        R --> D
        R --> E
        R --> LG
    end

    subgraph disk["the record — the deliverable"]
        direction TB
        S["solutions/NNN_*.py"]
        J["logs/&lt;run&gt;/NNNN.json"]
        M["LEDGER.md"]
        EV["events.jsonl"]
        DF["diffs/NNNN.diff"]
    end

    T -->|"write_solution"| S
    T -->|"run_experiment"| R
    S -.->|"subprocess, no labels"| R
    LG --> J & M & EV
    J -.->|"gendiffs.py"| DF
    J -->|"read back every turn"| P

    WEB["web/server.py<br/>local console"]
    WEB -->|"reads"| J & EV & DF
    WEB -->|"spawns"| L
```

**The agent has exactly five tools** ([tools.py](../agent/tools.py)):
`read_ledger`, `read_solution`, `write_solution`, `run_experiment`, `web_search`.
There is no edit tool and no shell.

---

## 2. One iteration

The agent has no memory between iterations. Each turn is rebuilt from disk —
which is why a killed run can be restarted and resume knowing everything.

```mermaid
sequenceDiagram
    autonumber
    participant L as loop.py
    participant P as prompt.py
    participant M as model (gpt-5.5)
    participant T as tools.py
    participant H as harness/run.py
    participant LG as ledger.py

    L->>L: _budget_check() — iterations, experiments,<br/>wall clock, cost
    L->>LG: converged()?
    Note over L,LG: either check can end the run here

    L->>P: build_user_message()
    P->>LG: _load_all() → every record
    P-->>L: ledger table (GAUC and nDCG@5 split out)<br/>+ search shape + UCT ranking<br/>+ current best solution's source
    L->>M: system prompt + that one message + 5 tool schemas

    loop up to MAX_TOOL_ROUNDS (20)
        M-->>L: tool call
        alt web_search
            L->>T: query → result text
        else read_ledger / read_solution
            L->>T: read from disk
        else write_solution
            L->>T: full file, name must match NNN_name.py
        else run_experiment
            L->>H: solution + hypothesis + parent + split
            H-->>L: scores, or an error as TEXT
        end
        T-->>M: result
    end

    L->>LG: write the iteration record
    Note right of LG: NNNN.json + a LEDGER.md row + events.jsonl
    L->>L: _compact(messages) — drop the stale turn
```

**API failure is not run failure.** A retryable error backs off and retries up
to `MAX_API_RETRIES`; a fatal one (bad key, malformed request) stops the run
rather than burning the budget on a call that will fail identically forever.
Abandoning an *iteration* is not abandoning the *run* — the ledger holds
everything learned, so the next turn rebuilds from disk.

---

## 3. What `run_experiment` actually does

This is the part that has to be untrickable. Note the order: the guards come
before anything is executed.

```mermaid
flowchart TD
    A["run_experiment(solution, hypothesis, parent, split)"] --> B{"split is<br/>valid or dev?"}
    B -->|no| X1["refused — the agent<br/>cannot ask for test"]
    B -->|yes| C{"path inside<br/>solutions/?"}
    C -->|no| X2["refused"]
    C -->|yes| D{"source hash<br/>seen before?"}
    D -->|yes| X3["duplicate — not re-run"]
    D -->|no| E["seed 0 in a subprocess<br/>no labels in its env"]

    E --> F{"crashed, or<br/>timed out at 900s?"}
    F -->|yes| X4["error recorded as TEXT<br/>the agent reads it next turn"]
    F -->|no| G{"clearly below<br/>the incumbent?"}
    G -->|yes| H["stop after 1 seed<br/>ADAPTIVE_SEEDS"]
    G -->|no| I["seeds 1 and 2, concurrently"]

    H --> J["evaluate.py — official, unmodified<br/>GAUC + nDCG@5 per seed"]
    I --> J
    J --> K["mean across seeds<br/>+ primary_std"]

    K --> L{"primary above the<br/>oracle ceiling 0.8484?"}
    L -->|yes| X5["CHEATING — label leakage.<br/>Result invalid."]
    L -->|no| M{"same predictions hash,<br/>or same GAUC AND nDCG@5,<br/>as an earlier record?"}
    M -->|yes| X6["no-op — different code,<br/>same model. Nothing tested."]
    M -->|no| N["verdict()"]

    N --> O["KEPT / worse / noise / screen"]
    X3 & X4 & X5 & X6 & O --> P["ledger.write(record)"]
    P --> Q{"did the parent fail<br/>and this one score?"}
    Q -->|yes| R["solution_recovered event"]
    Q -->|no| S["done"]

    classDef bad fill:#3a1f24,stroke:#ef5f6b,color:#ffd7db
    classDef good fill:#17301f,stroke:#3fcf8e,color:#c9f2df
    class X1,X2,X5,X6 bad
    class O,R good
```

**One seed failing fails the experiment.** Averaging the seeds that happened to
work is not a measurement of anything, and it would hide a solution that breaks
one time in three.

---

## 4. Verdict vs convergence — two different questions

These are easy to conflate and they must not be. `verdict()` asks *"did this
beat the published baseline?"*; `converged()` asks *"has the search stopped
making progress?"*. Reusing the first for the second breaks the run: once the
agent clears the target, every result is `KEPT` forever and convergence can
never fire.

```mermaid
flowchart LR
    subgraph v["verdict() — measured against a FIXED 0.6015"]
        direction TB
        V1{"status"} -->|cheating| VA["CHEATING"]
        V1 -->|no-op| VB["no-op"]
        V1 -->|duplicate| VC["duplicate"]
        V1 -->|error| VD["failed"]
        V1 -->|ok| V2{"split"}
        V2 -->|dev| VE["screen — not comparable"]
        V2 -->|valid| V3{"primary − 0.6015"}
        V3 -->|"≥ +0.002"| VF["KEPT"]
        V3 -->|"≤ −0.002"| VG["worse"]
        V3 -->|"in between"| VH["noise"]
    end

    subgraph c["converged() — measured against the RUNNING best"]
        direction TB
        C1{"≥ 30 scored<br/>experiments?"} -->|no| CN["keep going —<br/>too early to call a plateau"]
        C1 -->|yes| C2["best of the last 3<br/>vs best of everything before"]
        C2 --> C3{"improvement<br/>under 0.002?"}
        C3 -->|yes| CY["converged — stop"]
        C3 -->|no| CN
    end
```

The 30-experiment floor is ours, not the organisers'. Taken literally the stated
rule fires at iteration 4, because three non-improvements in a row is simply
what the start of a search looks like — every run was stopping before it found
anything. The organisers confirmed a team may set its own ε, N and minimum floor
provided they are fixed before the run and recorded in the log. Ours are written
into every `run_start` event.

---

## 5. How a run ends

```mermaid
stateDiagram-v2
    [*] --> Running: python -m agent
    Running --> Running: iteration completes

    Running --> Converged: no gain > ε over 3,<br/>after ≥ 30 scored
    Running --> Capped: max_iter / max_experiments
    Running --> OverBudget: wall clock or cost ceiling
    Running --> Fatal: unrecoverable API error
    Running --> Interrupted: Ctrl-C
    Running --> Crashed: unhandled exception

    Converged --> [*]: run_end "converged"
    Capped --> [*]: run_end "max_iter reached"
    OverBudget --> [*]: run_end "budget: ..."
    Fatal --> [*]: run_end "fatal API error"
    Interrupted --> [*]: run_end "interrupted by user"
    Crashed --> [*]: run_end "crashed: ..."

    note right of Converged
        Only this one means the search
        finished. The others mean it
        was stopped.
    end note
```

Across the 26 recorded `run_end` events: **17 converged**, 4 hit `max_iter`,
2 were interrupted, 2 hit a budget ceiling, 1 crashed. `run_end` always carries
the reason, so a truncated run can never be mistaken for a finished one.

---

## 6. The search is a tree, not a list

Every record names a `parent`, and each hypothesis opens with `draft`,
`improve` or `debug` — so the record holds both the edge and the intent behind
it. This is what lets the agent abandon a line and return to an earlier one.

```mermaid
flowchart LR
    N8["8<br/>0.6035"] --> N11["11<br/>0.6040"]
    N11 --> N12["12 · 0.60398"]
    N11 --> N13["13 · 0.60402"]
    N11 --> N14["14 · 0.60400"]
    N11 --> N15["15 ✗ timeout"]
    N15 -.->|"debug — recovery"| N16["16 · 0.60354"]
    N11 --> N17["17<br/>0.60487"]
    N17 --> N18["18 · 0.60479"]

    classDef best fill:#17301f,stroke:#3fcf8e,color:#c9f2df,stroke-width:2px
    classDef dead fill:#3a1f24,stroke:#ef5f6b,color:#ffd7db
    classDef flat fill:#1c2532,stroke:#6b7d97,color:#9fb0c6
    class N17 best
    class N15 dead
    class N12,N13,N14,N16 flat
```

Real, from `logs/record-run-12`. Node 11 has five children. 12, 13 and 14 were
blend-weight tweaks that came in at −0.000058, −0.000018 and −0.000040 against
seed noise of 0.0008 — three iterations that resolved nothing. 15 was a new
mechanism that timed out; 16 rescued the idea but landed below 11. So when 17
was drafted, **11 was still the best node and that is where it branched from.**
A list-shaped search would have continued from 16 and carried a worse incumbent
forward.

Selection is UCT, scaled by the organisers' epsilon rather than by the observed
range — so seed noise cannot reorder the ranking. It is advisory: the agent
still names its own parent, and `run.py` records where that choice ranked.

---

## 7. What ends up on disk

```mermaid
flowchart TD
    RUN["logs/&lt;run&gt;/"] --> A["NNNN.json<br/>the full record — GAUC and nDCG@5<br/>separately, per-seed, tokens, cost,<br/>error, recovery_events"]
    RUN --> B["LEDGER.md<br/>one line per experiment.<br/>A RENDERING, not the source —<br/>it merges the two metrics"]
    RUN --> C["events.jsonl<br/>chronological. The only place a<br/>failure BETWEEN experiments lives"]
    RUN --> D["solutions/NNN_*.py<br/>what the agent wrote"]
    RUN --> E["diffs/NNNN.diff<br/>derived from D by gendiffs.py"]

    A -.-> B
    D -.-> E
```

Anything reasoning about *how* a result moved needs the JSON: GAUC and nDCG@5
routinely move in opposite directions and the merged `primary` hides it. The
agent's own best result came from noticing exactly that.

---

## Regenerating the diagrams

Nothing here is generated — they are hand-drawn from the code and go stale if
the code moves. The parts most worth re-checking after a change:

| diagram | check against |
| --- | --- |
| 2, one iteration | `run_loop()` in [loop.py](../agent/loop.py) |
| 3, the pipeline | `run_experiment()` in [run.py](../harness/run.py) |
| 4, verdict / convergence | `verdict()` and `converged()` in [ledger.py](../harness/ledger.py) |
| 5, how a run ends | the `run_end` details in any `events.jsonl` |
