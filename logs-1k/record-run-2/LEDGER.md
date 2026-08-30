# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`valid` is the **mean** validation primary across seeds; `+/-` is its standard
deviation across those seeds. `delta` is against the reproduced official
baseline (**0.6015**); a result counts only at **>= +0.002** (the official
epsilon). Two rows closer together than their `+/-` have not been told apart.

| # | parent | hypothesis | valid | +/- | delta | verdict | by |
|---|---|---|---|---|---|---|---|
| 1 | 001 | draft direction 1: replace pointwise BCE with within-user BPR loss, sampling one same-user | 0.6228 | 0.0008 | -0.0223 | worse | agent |
| 2 | 1 | improve 1: one-negative BPR helped, so sample two independent same-user negatives per posi | 0.6250 | 0.0024 | -0.0201 | worse | agent |
| 3 | 001 | draft direction 1 from 001: use LightGBM LambdaRank with user groups to optimise NDCG rank | -- | -- | -- | failed | agent |
| 4 | 3 | debug 3: fix the feature allocation bug (6 categorical + 24 numeric columns) so the Lambda | -- | -- | -- | failed | agent |
| 5 | 3 | debug 3: LightGBM rejects user groups over 10k rows, so split oversized users into same-us | 0.4636 | 0.0343 | -0.1815 | screen | agent |
| 6 | 5 | debug 5: the sampled LambdaRank result was near-random, so keep the same feature construct | 0.5231 | 0.0181 | -0.1220 | screen | agent |
| 7 | 2 | improve 2: two-uniform-negative BPR improved over one negative but likely wastes pairs on  | 0.6245 | 0.0010 | -0.0206 | worse | agent |
| 8 | 2 | improve 2: use heterogeneous implicit feedback as BPR confidence weighting (BPR paper: htt | -- | -- | -- | failed | agent |
| 9 | 8 | debug 8: the watch-time weighting idea timed out during raw CSV alignment, so stream-align | 0.6272 | 0.0005 | -0.0179 | screen | agent |
| 10 | 9 | debug 9: 010 showed the weighted model reduced to unweighted BPR because no raw rows align | 0.6272 | 0.0005 | -0.0179 | no-op | agent |
| 11 | 9 | debug 9: raw logs lack author_id and duration_ms, so align by available ordered keys (date | 0.6249 | 0.0005 | -0.0202 | screen | agent |
| 12 | 2 | draft direction 6: add raw-log temporal context (hourmin buckets and tab-hour crosses) to  | 0.6480 | 0.0033 | +0.0029 | screen | agent |
| 13 | 12 | improve 12: the dev screen showed temporal context gives a large lift, so run the same hou | 0.6456 | 0.0017 | +0.0005 | noise | agent |
| 14 | 13 | improve 13: temporal context helped, so add explicit sparse affinity crosses (user-author, | -- | -- | -- | failed | agent |
| 15 | 13 | improve 13: the same-tab negative raised nDCG before but hurt GAUC; with time features now | 0.6491 | 0.0019 | +0.0040 | screen | agent |
| 16 | 15 | improve 15: the two-uniform plus one same-tab hard negative variant screened above node 13 | 0.6482 | 0.0007 | +0.0031 | KEPT | agent |
| 17 | 16 | improve 16: node 14's broad sparse crosses timed out, so test only cheap user-context cros | 0.6495 | 0.0039 | +0.0044 | screen | agent |
| 18 | 17 | improve 17: the cheap user-context crosses screened above node 16 on dev without timing ou | -- | -- | -- | duplicate | agent |
| 19 | 17 | improve 17: the cheap user-context crosses screened above node 16 on dev without timing ou | 0.6534 | 0.0035 | +0.0083 | KEPT | agent |
| 20 | 19 | improve 19: add a cheap smoothed item/author/time statistical ranker as a 25% per-user z-s | 0.6417 | 0.0006 | -0.0034 | screen | agent |
| 21 | 19 | improve 19: node 19's user-context crosses lift nDCG but hurt GAUC, so replace one uniform | 0.6499 | 0.0014 | +0.0048 | screen | agent |
| 22 | 21 | improve 21: the same-hour hard negative variant screened slightly above node 17/19-style d | -- | -- | -- | duplicate | agent |
| 23 | 21 | improve 21: the same-hour hard negative variant screened slightly above node 17/19-style d | 0.6510 | 0.0027 | +0.0059 | KEPT | agent |
| 24 | 19 | improve 19: revisit the watch-time confidence idea on the current best time/user-context m | 0.6493 | 0.0017 | +0.0042 | screen | agent |
| 25 | 19 | improve 19: node 19 has high seed variance mostly in nDCG, so average two independently se | 0.6566 | 0.0018 | +0.0115 | KEPT | agent |
| 26 | 25 | improve 25: keep the cached node-25 members unchanged and test a readable fusion change, b | 0.6560 | 0.0012 | +0.0109 | KEPT | agent |
| 27 | 25 | improve 25: node 25's two-member average still has nDCG seed wobble, so add a third indepe | 0.6577 | 0.0016 | +0.0126 | KEPT | agent |
| 28 | 27 | improve 27: the third same-code seed improved both metrics, so test whether variance reduc | 0.6582 | 0.0008 | +0.0131 | KEPT | agent |
