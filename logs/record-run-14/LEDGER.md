# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`valid` is the **mean** validation primary across seeds; `+/-` is its standard
deviation across those seeds. `delta` is against the reproduced official
baseline (**0.6015**); a result counts only at **>= +0.002** (the
official epsilon). Two rows closer together than their `+/-` have not been told
apart.

| # | parent | hypothesis | valid | +/- | delta | verdict | by |
|---|---|---|---|---|---|---|---|
| 1 | - | control: official FM ported to PyTorch, pointwise BCE. 3-seed mean. | 0.6014 | 0.0002 | -0.0001 | noise | human |
| 2 | 1 | draft loss: replace pointwise BCE with within-user BPR pair sampling, following Rendle et  | 0.6019 | 0.0004 | +0.0004 | noise | agent |
| 3 | 2 | improve 2: BPR from scratch may waste capacity learning the baseline popularity/calibratio | 0.6009 | 0.0003 | -0.0006 | noise | agent |
| 4 | 2 | improve 2: replace random positive-negative BPR pairs with a ListNet-style per-user softma | 0.5991 | 0.0004 | -0.0024 | worse | agent |
| 5 | 2 | draft model: branch from best BPR/FM direction but test a readable new mechanism, LightGBM | 0.5766 | 0.0044 | -0.0249 | worse | agent |
| 6 | 2 | improve 2: the BPR draft helped slightly but sampled easy random negatives; make the ranki | 0.6018 | 0.0003 | +0.0003 | noise | agent |
| 7 | 6 | improve 6: since the play_time_ms join failed and hard negatives alone traded nDCG for GAU | 0.5980 | 0.0005 | -0.0035 | worse | agent |
| 8 | 1 | draft historical features: Kaggle OTTO recommender writeups emphasize user-item interactio | 0.6018 | 0.0002 | +0.0003 | noise | agent |
| 9 | 8 | improve 8: the hand-smoothed historical scores were weak; expose user-video/user-author/us | 0.5981 | 0.0003 | -0.0034 | worse | agent |
| 10 | 1 | draft time features: time-of-day and recency/order buckets are standard CTR ranking featur | 0.6026 | 0.0005 | +0.0011 | noise | agent |
| 11 | 10 | debug 10: node 10's CSV key join missed every row, so attach hour/order by the documented  | 0.6008 | 0.0003 | -0.0007 | noise | agent |
| 12 | 10 | improve 10: after debugging, real hour/order fields were noisy; keep node 10's successful  | 0.6012 | 0.0005 | -0.0003 | noise | agent |
| 13 | 10 | improve 10: node 10 stdout shows the BPR member alone peaked at 0.603036 before the BCE me | 0.6030 | 0.0002 | +0.0015 | noise | agent |
| 14 | 13 | improve 13: the BPR-only time model is seed-noisy but each seed is strong; average five in | 0.6040 | 0.0005 | +0.0025 | KEPT | agent |
| 15 | 14 | improve 14: add a genuinely different BPR member that weights positive-negative pairs by l | 0.6040 | 0.0005 | +0.0025 | no-op | agent |
| 16 | 14 | improve 14: node 14 is GAUC-strong but nDCG is still the bottleneck, so train a complement | 0.6041 | 0.0007 | +0.0026 | KEPT | agent |
| 17 | 16 | improve 16: use the cached old and balanced BPR members but change only fusion: add per-us | 0.6043 | 0.0005 | +0.0028 | KEPT | agent |
| 18 | 17 | improve 17: the rank blend helped both metrics, so add reciprocal-rank fusion at 15% to em | 0.6043 | 0.0005 | +0.0028 | KEPT | agent |
| 19 | 18 | draft LambdaRank: Burges LambdaMART weights pairwise logistic gradients by metric deltas f | 0.6040 | 0.0006 | +0.0025 | KEPT | agent |
| 20 | 19 | debug 19: LambdaRank from scratch was only ~0.597, so warm-start with balanced BPR and the | 0.6039 | 0.0005 | +0.0024 | KEPT | agent |
| 21 | 18 | draft multi-task: auxiliary feedback heads can regularize sparse CTR representations (e.g. | 0.6044 | 0.0003 | +0.0029 | KEPT | agent |
| 22 | 21 | improve 21: the multitask member helped despite being seed-noisy; keep the exact cached me | 0.6046 | 0.0003 | +0.0031 | KEPT | agent |
| 23 | 22 | draft sequence: DIN weights a user's behavior history by relevance to the candidate item ( | -- | -- | -- | failed | agent |
| 24 | 23 | debug 23: the DIN draft crashed because Module.parameters iterators were passed directly i | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 25 | 24 | improve 24: the DIN member overfit immediately from random init; warm-start it from the st | 0.6046 | 0.0005 | +0.0031 | KEPT | agent |
| 26 | 25 | improve 25: warm-DIN was stronger standalone but less complementary; instead reduce the ra | 0.6046 | 0.0004 | +0.0031 | KEPT | agent |
| 27 | 26 | improve 26: the 5-seed DIN block is now cached, so test whether its complementarity is con | 0.6047 | 0.0003 | +0.0032 | KEPT | agent |
| 28 | 27 | draft watch-time weighting: implicit-feedback ranking commonly weights pairwise positives  | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 29 | 28 | improve 28: watch-time weighting itself looked weak, so replace that added member with a m | 0.6040 | 0.0005 | +0.0025 | KEPT | agent |
| 30 | 27 | improve 27: the ensemble is seed-noisy and the third measured seed-family of node 27 was m | 0.6050 | -- | +0.0035 | KEPT | agent |
| 31 | 30 | improve 30: add a tiny train-only empirical-rate prior as a deterministic tie-breaker for  | 0.6051 | -- | +0.0036 | KEPT | agent |
| 32 | 31 | improve 31: the empirical prior helped, so make it more ranking-like: add sparse user-vide | -- | -- | -- | failed | agent |
| 33 | 32 | debug 32: node 32 crashed only because the imported base lacked percentile/RRF helpers; de | 0.6047 | -- | +0.0032 | KEPT | agent |
