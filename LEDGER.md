# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`delta` is validation primary against the reproduced official baseline
(**0.6015**). A result counts only at **>= +0.002** (the official epsilon).

| # | parent | hypothesis | valid | delta | verdict | by |
|---|---|---|---|---|---|---|
| 1 | - | baseline: official FM ported to PyTorch, pointwise BCE | 0.6014 | -0.0001 | noise | human |
| 2 | 001 | Replace pointwise BCE with true within-user BPR: sample (positive, negative) pairs from th | 0.6031 | +0.0016 | noise | agent |
| 3 | 002 | Since 1-negative same-user BPR improved to 0.6031, sample two same-user negatives per posi | 0.6033 | +0.0018 | noise | agent |
| 4 | 003 | Two same-user negatives per positive slightly improved BPR; test four negatives per positi | 0.6030 | +0.0015 | noise | agent |
