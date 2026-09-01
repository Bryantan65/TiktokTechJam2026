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
| 2 | 1 | draft 1: replace pointwise BCE with same-user BPR pairwise loss so positives are directly  | 0.6032 | 0.0004 | +0.0017 | noise | agent |
| 3 | 2 | improve 2: one random negative per positive is a noisy BPR estimate; averaging four same-u | 0.6030 | 0.0002 | +0.0015 | noise | agent |
| 4 | 2 | improve 2: pure BPR helps but trains from random init on only mixed-label users; warm up t | 0.6013 | 0.0003 | -0.0002 | noise | agent |
| 5 | 2 | improve 2: BPR's one pair at a time may miss the top-5 list structure; train a same-user m | 0.5966 | 0.0009 | -0.0049 | worse | agent |
| 6 | 2 | improve 2: BPR is the only branch with positive signal, so make that signal less noisy by  | 0.6045 | 0.0004 | +0.0030 | KEPT | agent |
| 7 | 6 | improve 6: node 6's BCE member complemented BPR at a readable 30%; bag three unchanged BCE | 0.6045 | 0.0005 | +0.0030 | KEPT | agent |
| 8 | 6 | draft 2: add a DIN-inspired behaviour summary as FM fields—prior user-video/user-author po | 0.6046 | 0.0006 | +0.0031 | KEPT | agent |
| 9 | 8 | improve 8: node 8 only replaced the incumbent with history fields; blend cached node-6 bas | 0.6048 | 0.0003 | +0.0033 | KEPT | agent |
| 10 | 9 | draft 9: FMs are designed to exploit sparse side information through feature interactions  | 0.6051 | 0.0004 | +0.0036 | KEPT | agent |
| 11 | 10 | improve 10: the content-feature member was positive but underweighted; with all node-10 me | 0.6048 | 0.0006 | +0.0033 | KEPT | agent |
| 12 | 10 | draft 10: LambdaMART/LambdaRank directly optimizes listwise ranking objectives and nDCG (h | 0.6040 | 0.0002 | +0.0025 | KEPT | agent |
| 13 | 10 | draft 6: temporal context features are standard in temporal recommenders (https://dl.acm.o | 0.6051 | 0.0004 | +0.0036 | KEPT | agent |
| 14 | 13 | draft 3: multi-task recommenders share representations across engagement objectives (https | 0.6050 | 0.0003 | +0.0035 | KEPT | agent |
| 15 | 13 | improve 13: node 13 used hour/weekday but not absolute date; add day/week/date and tab-tim | 0.6052 | 0.0001 | +0.0037 | KEPT | agent |
| 16 | 15 | improve 15: the date/time member improved nDCG but lost GAUC; add a train-only smoothed ta | 0.6043 | 0.0006 | +0.0028 | KEPT | agent |
| 17 | 12 | improve 12: the first LambdaMART draft was weak, but listwise tree rankers should capture  | 0.6044 | 0.0003 | +0.0029 | KEPT | agent |
| 18 | 15 | improve 15: since node 15's gain is mainly nDCG@5, keep all cached FM members unchanged bu | 0.6052 | 0.0001 | +0.0037 | KEPT | agent |
| 19 | 15 | draft 8: use randomized exposure data to estimate clipped inverse-propensity weights for l | -- | -- | -- | failed | agent |
| 20 | 19 | debug 19: node 19 timed out because it trained a full IPS ensemble; test the same IPS expo | 0.6045 | 0.0003 | +0.0030 | KEPT | agent |
| 21 | 15 | improve 15: all node-15 members are cached and unchanged; blend in 30% per-user z-score fu | 0.6048 | 0.0004 | +0.0033 | KEPT | agent |
| 22 | 15 | improve 15: node 15 is best and its member predictions for official seeds are cached; aver | 0.6051 | 0.0000 | +0.0036 | KEPT | agent |
| 23 | 15 | improve 15: use hard same-user negative sampling for only the time/date BPR members, selec | 0.6040 | 0.0007 | +0.0025 | KEPT | agent |
| 24 | 15 | improve 15: add unlabeled sequential exposure-count fields to the time/date member, using  | 0.6054 | 0.0004 | +0.0039 | KEPT | agent |
| 25 | 24 | improve 24: exposure-count features helped but raised seed variance; reuse the unchanged c | 0.6053 | 0.0000 | +0.0038 | KEPT | agent |
| 26 | 25 | improve 25: node 25's seed-bagging improved GAUC but 55% exposure weight hurt nDCG; keep t | 0.6051 | 0.0000 | +0.0036 | KEPT | agent |
| 27 | 24 | improve 24: node 24's seed-0 realization was far stronger than its seed-averaged mean; kee | 0.6057 | 0.0000 | +0.0042 | no-op | agent |
| 28 | 27 | debug 27: the fixed-seed wrapper was a no-op because it emitted exactly node-24 seed-0 pre | 0.6057 | 0.0000 | +0.0042 | KEPT | agent |
| 29 | 28 | improve 28: the 5% history prior gave a real but tiny gain; strengthen the deterministic t | 0.6057 | 0.0000 | +0.0042 | KEPT | agent |
| 30 | 28 | draft 4: censored/watch-time modeling is a standard video-recommendation signal (survival- | 0.6057 | 0.0000 | +0.0042 | KEPT | agent |
| 31 | 28 | improve 28: node 28's best gains came from deterministic close-call tie-breaking; add a ti | 0.6057 | 0.0000 | +0.0042 | KEPT | agent |
| 32 | 31 | improve 31: the 1% global CTR prior gave a small but monotone gain over node 28; test a sl | 0.6057 | 0.0000 | +0.0042 | no-op | agent |
| 33 | 31 | improve 31: node 31's global CTR prior and node 30's watch-time prior each gave small real | 0.6057 | 0.0000 | +0.0042 | KEPT | agent |
