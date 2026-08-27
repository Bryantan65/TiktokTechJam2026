# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`delta` is validation primary against the reproduced official baseline
(**0.6015**). A result counts only at **>= +0.002** (the official epsilon).

| # | parent | hypothesis | valid | delta | verdict | by |
|---|---|---|---|---|---|---|
| 1 | - | baseline: FM with pointwise logloss, autograd port | 0.6014 | -0.0001 | noise | human |
| 2 | 1 | ~~Implement BPR loss for pairwise ranking~~ **INVALID — bug, not a finding: label never used, pairs are arbitrary rows. Below random. Says nothing about pairwise ranking.** | 0.4970 | -0.1045 | invalid | agent |
