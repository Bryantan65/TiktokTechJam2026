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
| 2 | 1 | draft 1: replace pointwise BCE with same-user BPR pairs, matching the pairwise implicit-fe | -- | -- | -- | failed | agent |
| 3 | 2 | debug 2: 002 failed before training because encode() supplies users as a list; convert use | 0.6033 | 0.0005 | +0.0018 | noise | agent |
| 4 | 3 | improve 3: one sampled negative per positive is a noisy approximation to BPR; sample three | 0.6027 | 0.0009 | +0.0012 | noise | agent |
| 5 | 3 | improve 3: test the other standard ranking-loss variant under direction 1, a same-user lis | 0.5959 | 0.0004 | -0.0056 | worse | agent |
| 6 | 3 | improve 3: BPR improves ranking but may lose some calibrated pointwise signal; blend stand | 0.6036 | 0.0002 | +0.0021 | KEPT | agent |
| 7 | 6 | improve 6: node 6's blend works and BPR is the higher-variance component; average two inde | 0.6039 | 0.0003 | +0.0024 | KEPT | agent |
| 8 | 7 | draft 6: add temporal context to the current best FM ensemble, using hour/day categorical  | 0.6046 | 0.0005 | +0.0031 | KEPT | agent |
| 9 | 8 | improve 8: node 8 shows temporal signal; add half-hour plus coarser daypart/weekend bucket | 0.6043 | 0.0001 | +0.0028 | KEPT | agent |
