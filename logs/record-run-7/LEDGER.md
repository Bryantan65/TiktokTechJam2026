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
| 2 | 1 | draft loss: replace pointwise BCE with same-user BPR pairs, following Rendle et al.'s BPR  | 0.6018 | 0.0004 | +0.0003 | noise | agent |
| 3 | 2 | improve 2: BPR from scratch was slightly positive but noisy; initialise with 5 epochs of B | 0.5993 | 0.0004 | -0.0022 | worse | agent |
