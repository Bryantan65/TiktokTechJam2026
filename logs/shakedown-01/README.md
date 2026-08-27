# Shakedown run 01 — archived, not results

The first agent run, 2026-08-27. Kept as evidence of the shakedown; **not**
valid evidence about any technique. Do not seed a real run from this.

9 experiments, $3.34, 406s compute. Nothing beat the baseline.

What it actually found — four bugs in our own code:

| # | what happened | fix |
| --- | --- | --- |
| 2 | BPR loss never read the label; pairs were arbitrary rows. Scored 0.4970, below random | prompt rule: a score below random is a bug, not a finding |
| 4, 5, 7 | Recorded as `ok` at 0.601400 — bit-identical to the untouched baseline. Each warmed up with BCE, fine-tuned with a new loss, then kept the best checkpoint; the warmup always won, so the change was discarded | harness now detects no-ops by matching GAUC and nDCG@5 |
| — | history truncation could orphan a tool message and 400 the API | `_truncate()` cuts only at a user message |
| — | spinner wrote braille to a cp1252 console, spamming tracebacks | ASCII fallback |

Also measured here: **$0.371 per experiment**, input is **77%** of cost
(514,087 in vs 25,718 out). That is the number the caps were sized against.

The lesson worth keeping: five of nine "results" tested nothing, and the ledger
recorded them as evidence that the organisers' top-ranked directions did not
help. Two separate detectors now exist because of this run.
