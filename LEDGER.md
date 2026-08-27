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
| 5 | 003 | BPR-2neg gives the best primary via GAUC but loses a little nDCG@5; add a small BCE auxili | 0.6036 | +0.0021 | KEPT | agent |
| 6 | 005 | The 0.10 BCE auxiliary improved pure BPR-2neg, so try aux_weight=0.20 to see whether stron | 0.6035 | +0.0020 | KEPT | agent |
| 7 | 005 | The current best uses uniformly sampled same-user negatives, which optimizes average pair  | 0.5981 | -0.0034 | worse | agent |
| 8 | 005 | For each positive with two same-user sampled negatives, replace independent BPR terms with | 0.6039 | +0.0024 | KEPT | agent |
| 9 | 008 | Increase sampled softmax from 2 to 3 same-user negatives per positive while keeping BCE au | 0.6037 | +0.0022 | KEPT | agent |
| 10 | 008 | Sample distinct negatives inside the 2-negative same-user softmax set whenever possible, a | 0.6037 | +0.0022 | KEPT | agent |
| 11 | 008 | Reduce the BCE auxiliary in the current best sampled-softmax FM from 0.10 to 0.05; lighter | 0.6039 | +0.0024 | KEPT | agent |
| 12 | 011 | Reducing BCE auxiliary from 0.10 to 0.05 slightly improved the sampled-softmax FM, suggest | 0.6040 | +0.0025 | KEPT | agent |
| 13 | 012 | The loss direction has saturated, so switch to time drift features: add coarse hour/date b | -- | -- | failed | agent |
