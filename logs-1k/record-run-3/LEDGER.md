# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`valid` is the **mean** validation primary across seeds; `+/-` is its standard
deviation across those seeds. `delta` is against the reproduced official
baseline (**0.6451**); a result counts only at **>= +0.002** (the
official epsilon). Two rows closer together than their `+/-` have not been told
apart.

| # | parent | hypothesis | valid | +/- | delta | verdict | by |
|---|---|---|---|---|---|---|---|
| 1 | 001 | draft 1: replace pointwise BCE with a ListNet-style per-user softmax ranking loss, -log pr | 0.6165 | 0.0006 | -0.0286 | worse | agent |
| 2 | 1 | improve 1: the softmax FM improved GAUC but likely under-optimizes top-5; try LambdaRank/N | -- | -- | -- | failed | agent |
| 3 | 2 | debug 2: LightGBM crashed because some per-user query groups exceed its 10000-row limit; s | 0.4975 | 0.0471 | -0.1476 | screen | agent |
| 4 | 3 | debug 3: the LambdaRank split-groups screen was near-random and highly unstable; hold the  | 0.4001 | 0.1502 | -0.2450 | screen | agent |
| 5 | 4 | debug 4: binary LightGBM with the same stats was also bad, so test the stat construction d | 0.6094 | -- | -0.0357 | screen | agent |
| 6 | 1 | improve 1: after the softmax ranking loss gave a real gain, try standard same-user BPR sam | 0.6066 | 0.0038 | -0.0385 | worse | agent |
| 7 | 5 | improve 5: the direct historical-rate predictor confirmed stat alignment on dev; run the s | 0.6162 | 0.0000 | -0.0289 | worse | agent |
| 8 | 1 | improve 1: the softmax FM and historical stats have opposite metric strengths, so blend th | 0.6269 | 0.0018 | -0.0182 | worse | agent |
| 9 | 8 | improve 8: reuse the cached 009 members but shift the per-user-z blend toward the higher-G | 0.6270 | 0.0012 | -0.0181 | worse | agent |
| 10 | 9 | improve 9: Kaggle RecSys writeups such as OTTO emphasize rank/count fusion (https://www.ka | 0.6278 | 0.0017 | -0.0173 | worse | agent |
| 11 | 10 | improve 10: add a readable 40% recency-decayed history-stat rank member, motivated by RecS | 0.6275 | 0.0019 | -0.0176 | worse | agent |
| 12 | 11 | improve 11: recency rates lifted GAUC but hurt top-5, so replace them with a sequence-like | 0.6239 | 0.0015 | -0.0212 | worse | agent |
| 13 | 10 | improve 10: add a readable 25% label-free target-exposure rank member to node 10; KuaiRand | 0.6189 | 0.0001 | -0.0262 | worse | agent |
| 14 | 10 | improve 10: with added heuristic members hurting node 10, use cached node-10 members only  | 0.6277 | 0.0014 | -0.0174 | worse | agent |
| 15 | 14 | improve 14: node 14 improved nDCG by globally tilting toward softmax but lost GAUC, so kee | 0.6209 | 0.0019 | -0.0242 | worse | agent |
| 16 | 2 | debug 2: earlier LambdaRank was either crashed or near-random, so retry the standard Light | 0.6202 | 0.0023 | -0.0249 | worse | agent |
| 17 | 14 | draft 6: KuaiRand explicitly contains chronological logs and feedback histories (paper DOI | 0.6251 | 0.0004 | -0.0200 | worse | agent |
| 18 | 10 | draft 3: multi-task/auxiliary feedback ranking uses clicks, likes and dwell-style signals  | 0.6156 | 0.0025 | -0.0295 | worse | agent |
| 19 | 18 | debug 18: node 18 may have failed because auxiliary signals were used as a noisy standalon | 0.6291 | 0.0011 | -0.0160 | worse | agent |
| 20 | 19 | improve 19: node 19 showed auxiliary supervision helps; make it more ranking-aligned by we | 0.6273 | 0.0041 | -0.0178 | worse | agent |
| 21 | 19 | improve 19: auxiliary heads in node 19 regularized the main model but were discarded; use  | 0.6291 | 0.0018 | -0.0160 | worse | agent |
| 22 | 21 | improve 21: replace the fixed node-21 fusion with a supervised stacker trained on leave-on | 0.6217 | 0.0021 | -0.0234 | worse | agent |
| 23 | 21 | improve 21: node 21's auxiliary rank improved top-5 but diluted whole-list GAUC, so keep t | 0.6299 | 0.0014 | -0.0152 | worse | agent |
| 24 | 23 | improve 23: the top-gated auxiliary score helped, so make the gate sharper and stronger (b | 0.6299 | 0.0010 | -0.0152 | worse | agent |
| 25 | 24 | draft 2: add a readable collaborative-filtering sequence/graph member to node 24, using tr | 0.6308 | 0.0021 | -0.0143 | worse | agent |
| 26 | 25 | improve 25: the positive-only SVD helped nDCG but ignores explicit negative impressions; a | 0.6279 | 0.0025 | -0.0172 | worse | agent |
| 27 | 25 | improve 25: signed SVD hurt, but node 25's positive-only CF raised nDCG; reuse the unchang | 0.6238 | 0.0023 | -0.0213 | worse | agent |
| 28 | 25 | improve 25: node 25 is the best but its seed spread is large and seed 2 is much stronger,  | 0.6311 | -- | -0.0140 | worse | agent |
| 29 | 28 | improve 28: bagging raised GAUC but averaged away node-25 seed 2's strongest top-5, so fix | 0.6330 | -- | -0.0121 | worse | agent |
| 30 | 29 | improve 29: node 29 showed CF is useful mainly for top-5, so sharpen the same cached seed- | 0.6317 | -- | -0.0134 | worse | agent |
| 31 | 29 | improve 29: implicit-feedback MF commonly weights observed interactions by confidence/stre | 0.6306 | -- | -0.0145 | worse | agent |
| 32 | 29 | improve 29: node 29's best gain is top-5 personalization, so add a TRAIN-only repeat-histo | 0.6314 | -- | -0.0137 | worse | agent |
| 33 | 29 | improve 29: the last variants added new high-variance members and hurt, so keep node-29's  | 0.6309 | -- | -0.0142 | worse | agent |
| 34 | 29 | draft 8: change mechanism to exposure debiasing using the label-free random log, inspired  | 0.6270 | -- | -0.0181 | worse | agent |
