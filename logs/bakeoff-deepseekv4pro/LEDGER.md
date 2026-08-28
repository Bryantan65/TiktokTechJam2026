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
| 2 | 1 | draft 1 (change the loss): replace pointwise BCE with a listwise softmax over each user's  | 0.5960 | 0.0009 | -0.0055 | worse | agent |
| 3 | 2 | debug 2: the listwise softmax changed both the objective AND the batch structure (full use | 0.6088 | 0.0002 | +0.0073 | screen | agent |
| 4 | 2 | debug 2 -> promote: BPR screen on dev was strong (0.6088, nDCG@5 0.5524, +/- 0.0002). Now  | -- | -- | -- | duplicate | agent |
