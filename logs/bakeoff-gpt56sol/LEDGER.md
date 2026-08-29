# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`valid` is the **mean** validation primary across seeds; `+/-` is its standard
deviation across those seeds. `delta` is against the reproduced official
baseline (**0.6015**); a result counts only at **>= +0.002** (the official
epsilon). Two rows closer together than their `+/-` have not been told apart.

| # | parent | hypothesis | valid | +/- | delta | verdict | by |
|---|---|---|---|---|---|---|---|
| 1 | - | control: official FM ported to PyTorch, pointwise BCE. 3-seed mean. | 0.6014 | 0.0002 | -0.0001 | noise | human |
| 2 | 1 | draft loss: train the FM with BPR's -log sigmoid score difference on balanced positive-neg | -- | -- | -- | failed | agent |
| 3 | 2 | debug 2: encoded user IDs are returned as a Python sequence, so convert them to a NumPy ar | 0.6009 | 0.0002 | -0.0006 | noise | agent |
| 4 | 3 | improve 3: pure BPR loses pointwise prevalence information, so jointly optimize balanced-p | 0.6009 | 0.0003 | -0.0006 | noise | agent |
| 5 | 1 | draft user behaviour sequences: add causal, smoothed positive-history affinities for each  | 0.6018 | 0.0005 | +0.0003 | noise | agent |
| 6 | 5 | improve 5: node 5 used only global target popularity and was not actually personalized; ad | 0.6022 | 0.0005 | +0.0007 | noise | agent |
| 7 | 6 | improve 6: persistent lifetime affinities dilute changing user interests; exponentially de | 0.6023 | 0.0003 | +0.0008 | noise | agent |
