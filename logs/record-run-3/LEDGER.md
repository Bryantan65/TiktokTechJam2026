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
| 2 | 1 | draft direction 1: replace pointwise BCE with same-user BPR pairs, optimizing -log sigmoid | 0.6029 | 0.0004 | +0.0014 | noise | agent |
| 3 | 2 | improve 2: BPR showed signal, so sample two same-user negatives per positive per epoch to  | 0.6028 | 0.0006 | +0.0013 | noise | agent |
| 4 | 2 | improve 2: BPR improved GAUC but not enough nDCG, so add a small pointwise BCE term on the | 0.6028 | 0.0005 | +0.0013 | noise | agent |
| 5 | 2 | improve 2: BPR has saturated but loss direction is not exhausted; try a true within-user L | 0.5995 | 0.0006 | -0.0020 | worse | agent |
| 6 | 2 | improve 2: BPR is the only node with signal; initialize it from a 7-epoch BCE FM warmup, t | 0.6002 | 0.0007 | -0.0013 | noise | agent |
| 7 | 2 | improve 2: BPR showed the only positive signal, so refine its negative sampler to draw 75% | 0.5943 | 0.0004 | -0.0072 | worse | agent |
| 8 | 2 | improve 2: the uniform BPR method is stable but just under target; average three independe | 0.6037 | 0.0003 | +0.0022 | KEPT | agent |
| 9 | 8 | improve 8: the 3-member BPR ensemble crossed target, so push the same mechanism by using f | 0.6043 | 0.0002 | +0.0028 | KEPT | agent |
| 10 | 9 | improve 9: the five-member within-user rank ensemble gave a clear gain, so test whether th | 0.6045 | 0.0001 | +0.0030 | KEPT | agent |
| 11 | 10 | improve 10: the identical-member ensemble is flattening, so add diversity by blending its  | 0.6042 | 0.0003 | +0.0027 | KEPT | agent |
| 12 | 10 | draft direction 2: add a train-only user behavior history signal by blending the best BPR  | 0.6044 | 0.0001 | +0.0029 | KEPT | agent |
| 13 | 10 | improve 10: convergence is at risk, so make the highest-probability move: extend the only  | 0.6045 | 0.0002 | +0.0030 | KEPT | agent |
| 14 | 10 | draft direction 6: add explicit temporal context to the best BPR ensemble by reading raw h | 0.6049 | 0.0001 | +0.0034 | KEPT | agent |
| 15 | 14 | debug 14: the first time-context draft warned that almost every row missed raw alignment,  | 0.6049 | 0.0001 | +0.0034 | KEPT | agent |
| 16 | 14 | debug 14: per-key raw matching still missed nearly all rows, so follow the documented file | 0.5829 | 0.0005 | -0.0186 | worse | agent |
| 17 | 14 | debug 14: node 14's lookup included author_id even though raw logs lack it, forcing an hou | 0.6051 | 0.0002 | +0.0036 | KEPT | agent |
| 18 | 17 | draft direction 3: add multi-task auxiliary feedback heads sharing the FM embeddings while | 0.6048 | 0.0001 | +0.0033 | KEPT | agent |
| 19 | 17 | improve 17: time features are the only new signal that clearly moved the best node, so add | 0.6044 | 0.0002 | +0.0029 | KEPT | agent |
| 20 | 18 | improve 18: the multi-task draft is not exhausted; its rare-task pos_weight and shared sca | 0.6050 | 0.0003 | +0.0035 | KEPT | agent |
| 21 | 20 | improve 20: the gentle multi-task model is close but not better than the plain time model, | 0.6049 | 0.0004 | +0.0034 | KEPT | agent |
| 22 | 17 | draft direction 4: use CWM-style watch-time supervision as an additional ranking signal, m | 0.6049 | 0.0002 | +0.0034 | KEPT | agent |
| 23 | 22 | improve 22: the pure watch-ratio member improved nDCG but hurt GAUC because it ranked by a | 0.6052 | 0.0002 | +0.0037 | KEPT | agent |
| 24 | 23 | improve 23: GAUC-weighted BPR over-represents users with many positives while nDCG@5 is un | 0.6055 | 0.0001 | +0.0040 | KEPT | agent |
| 25 | 24 | improve 24: the added user-balanced BPR member gave the clearest recent gain, so average t | 0.6054 | 0.0003 | +0.0039 | KEPT | agent |
| 26 | 24 | improve 24: node 12's history draft was not refined, and the FM can only store repeat user | 0.6054 | 0.0001 | +0.0039 | KEPT | agent |
| 27 | 24 | draft direction 5: add a small DeepFM member to the node-24 ensemble, following Guo et al. | 0.6055 | 0.0002 | +0.0040 | KEPT | agent |
| 28 | 27 | improve 27: node 27's DeepFM member only shifted the GAUC/nDCG tradeoff, so make the deep  | 0.6054 | 0.0001 | +0.0039 | KEPT | agent |
| 29 | 24 | improve 24: the strongest ensemble still trains most label members on easy cross-tab negat | 0.6054 | 0.0001 | +0.0039 | KEPT | agent |
| 30 | 27 | improve 27: with convergence imminent, refine the current best by adding a very weak empir | 0.6054 | 0.0002 | +0.0039 | KEPT | agent |
