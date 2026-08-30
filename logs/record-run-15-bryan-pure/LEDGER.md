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
| 2 | 1 | draft 1: train the same FM with same-user BPR pairs instead of pointwise BCE, matching the | 0.6027 | 0.0005 | +0.0012 | noise | agent |
| 3 | 2 | improve 2: sample three same-user negatives per positive instead of one so each epoch sees | 0.6035 | 0.0006 | +0.0020 | noise | agent |
| 4 | 3 | improve 3: replace averaging three random negatives with online hard negative selection fr | 0.5797 | 0.0006 | -0.0218 | worse | agent |
| 5 | 4 | debug 4: max-selected hard negatives were too aggressive; use detached softmax weights ove | 0.6030 | 0.0006 | +0.0015 | noise | agent |
| 6 | 3 | improve 3: node 3 is the best mechanism but seed-noisy; average three independently seeded | 0.6039 | 0.0002 | +0.0024 | KEPT | agent |
| 7 | 6 | improve 6: keep the node-6 BPR members unchanged but fuse their per-user z-scored predicti | 0.6040 | 0.0002 | +0.0025 | KEPT | agent |
| 8 | 7 | improve 7: fuse unchanged BPR seed-bag members with 70% per-user z-score averaging and 30% | 0.6038 | 0.0003 | +0.0023 | KEPT | agent |
| 9 | 7 | draft 4: use watch-time as a confidence weight on the successful same-user BPR pairs, foll | 0.5972 | 0.0001 | -0.0043 | worse | agent |
| 10 | 9 | debug 9: the large drop likely came from assuming raw CSV prefix alignment and over-strong | 0.6038 | 0.0002 | +0.0023 | KEPT | agent |
| 11 | 10 | improve 10: the debugged watch-time BPR is slightly weaker standalone but has marginally h | 0.6042 | 0.0003 | +0.0027 | KEPT | agent |
| 12 | 11 | improve 11: the 30% watch blend moved both GAUC and nDCG up, so try a still-readable 40% w | 0.6043 | 0.0004 | +0.0028 | KEPT | agent |
| 13 | 12 | improve 12: the watch member's contribution has improved monotonically from 30% to 40%, so | 0.6043 | 0.0003 | +0.0028 | KEPT | agent |
| 14 | 13 | draft 1: add a sampled-softmax/listwise FM trained to choose one positive among same-user  | 0.6044 | 0.0002 | +0.0029 | KEPT | agent |
| 15 | 14 | improve 14: node 14 improved nDCG but only gave the listwise member 30%; keep member train | 0.6041 | 0.0001 | +0.0026 | KEPT | agent |
| 16 | 14 | draft 5: add a diverse LightGBM LambdaRank member with leakage-safe leave-one-out target/c | 0.6001 | 0.0011 | -0.0014 | noise | agent |
| 17 | 16 | debug 16: the large drop may be LambdaRank/group construction rather than target-encoding  | 0.6017 | 0.0009 | +0.0002 | noise | agent |
| 18 | 14 | draft 6: KuaiRand includes temporal context useful for sequential/ranking models (dataset  | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 19 | 18 | improve 18: the time member clearly helped; replace coarse hour4+dow with a richer time en | 0.6047 | 0.0003 | +0.0032 | KEPT | agent |
| 20 | 19 | improve 19: node 18 and 19 time members have similar primary but different GAUC/nDCG tilt; | 0.6048 | 0.0003 | +0.0033 | KEPT | agent |
| 21 | 20 | draft 2: add a DIN-inspired user history signal (paper https://ojs.aaai.org/index.php/AAAI | 0.5997 | 0.0002 | -0.0018 | noise | agent |
| 22 | 21 | debug 21: the hand-weighted sequence overlap likely imposed the wrong signs/scales; learn  | 0.6050 | 0.0002 | +0.0035 | KEPT | agent |
| 23 | 22 | improve 22: learned sequence-context improved both metrics at 30% weight, so keep the memb | 0.6050 | 0.0002 | +0.0035 | KEPT | agent |
| 24 | 22 | draft 3: add an auxiliary-feedback multi-task sequence FM at 30% blend, following MTL reco | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 25 | 22 | improve 22: the learned sequence member helped while time members helped separately; add a | 0.6052 | 0.0001 | +0.0037 | KEPT | agent |
| 26 | 25 | improve 25: node25's time-aware sequence member raised GAUC with little nDCG loss, so keep | 0.6051 | 0.0002 | +0.0036 | KEPT | agent |
| 27 | 25 | improve 25: uniform 40% overweights the time-aware sequence member, so keep all cached mem | 0.6052 | 0.0001 | +0.0037 | KEPT | agent |
| 28 | 24 | debug 24: auxiliary MTL hurt on the plain sequence rows and may have dominated BPR; apply  | 0.6051 | 0.0002 | +0.0036 | KEPT | agent |
| 29 | 25 | improve 25: the time-sequence member helped; add leakage-safe label-free prior exposure/re | 0.6055 | 0.0002 | +0.0040 | KEPT | agent |
| 30 | 29 | improve 29: exposure context improved both metrics at 30%, so keep the trained members unc | 0.6054 | 0.0001 | +0.0039 | KEPT | agent |
