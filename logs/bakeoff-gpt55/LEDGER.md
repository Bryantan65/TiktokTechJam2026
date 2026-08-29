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
| 2 | 1 | draft loss: replace pointwise BCE with within-user BPR pairs, sampling one same-user negat | 0.6029 | 0.0004 | +0.0014 | noise | agent |
| 3 | 2 | improve 2: BPR helped, but one sampled negative per positive gives sparse/noisy pair super | 0.6031 | 0.0005 | +0.0016 | noise | agent |
| 4 | 3 | improve 3: random BPR pairs help but still optimise sampled pair order, not the whole user | 0.5964 | 0.0002 | -0.0051 | worse | agent |
| 5 | 3 | improve 3: BPR is the only improving mechanism and random negatives may be too easy after  | 0.6011 | 0.0002 | -0.0004 | noise | agent |
| 6 | 3 | improve 3: random BPR is best but may discard BCE's complementary pointwise signal; test a | 0.6035 | 0.0005 | +0.0020 | noise | agent |
| 7 | 6 | improve 6: the 70/30 rank blend reached the target but still has +/-0.00046 from BPR sampl | 0.6040 | 0.0003 | +0.0025 | KEPT | agent |
| 8 | 7 | improve 7: add explicit linear-only (user,author), (user,video), and (user,tab) crosses to | 0.6041 | 0.0005 | +0.0026 | KEPT | agent |
| 9 | 8 | draft user behaviour sequences: approximate history attention with smoothed user-author/us | 0.6019 | 0.0008 | +0.0004 | noise | agent |
