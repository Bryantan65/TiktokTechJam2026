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
| 1 | - | control: official FM ported to PyTorch, pointwise BCE. 3-seed mean on KuaiRand-1k. | 0.6434 | 0.0006 | -0.0017 | noise | human |
| 2 | 1 | draft loss: replace pointwise BCE with same-user BPR pairs, using Rendle et al.'s -log sig | 0.6015 | 0.0020 | -0.0436 | worse | agent |
| 3 | 2 | debug 2: pure BPR discarded the calibrated pointwise signal and collapsed nDCG, so keep ba | 0.6383 | 0.0011 | -0.0068 | worse | agent |
| 4 | 1 | draft model: try LightGBM LambdaMART with categorical features and smoothed train-only CTR | -- | -- | -- | failed | agent |
| 5 | 4 | debug 4: encoded categorical values include -1 for missing/unseen values, so shift codes t | -- | -- | -- | failed | agent |
| 6 | 5 | debug 5: robustly factorize every base and crossed categorical feature into dense non-nega | -- | -- | -- | failed | agent |
| 7 | 6 | debug 6: users from encode() are a Python sequence, so convert to np.asarray before applyi | -- | -- | -- | failed | agent |
| 8 | 7 | debug 7: LightGBM lambdarank refuses query groups above 10000 rows, so split only oversize | 0.4762 | 0.0529 | -0.1689 | screen | agent |
| 9 | 8 | debug 8: isolate the below-random LambdaMART failure by keeping the same categorical/stat  | 0.3072 | 0.0003 | -0.3379 | screen | agent |
| 10 | 9 | debug 9: the pointwise LightGBM with historical stats is even below random, so remove all  | 0.6462 | 0.0009 | +0.0011 | screen | agent |
| 11 | 10 | improve 10: the minimal LightGBM binary classifier recovered on dev, so validate that same | 0.6499 | 0.0044 | +0.0048 | KEPT | agent |
| 12 | 11 | improve 11: keep the same minimal LightGBM mechanism but reduce GBDT sampling variance by  | 0.6484 | 0.0038 | +0.0033 | KEPT | agent |
| 13 | 11 | improve 11: add explicit categorical crosses (user-author/user-video/user-tab and item/aut | 0.6466 | 0.0035 | +0.0015 | noise | agent |
| 14 | 11 | improve 11: node 11 is selected by validation logloss, so switch LightGBM early stopping t | 0.6495 | 0.0044 | +0.0044 | KEPT | agent |
| 15 | 14 | draft time features: starting from node 14, add non-label time signals (hourmin from raw l | 0.6419 | 0.0033 | -0.0032 | worse | agent |
| 16 | 11 | draft watch-time modelling: from best node 11, keep the minimal LightGBM but reweight posi | 0.6461 | 0.0028 | +0.0010 | noise | agent |
| 17 | 14 | debug 14: the earlier LambdaMART failures mixed ranking loss with broken stat/cross featur | 0.6337 | 0.0040 | -0.0114 | worse | agent |
| 18 | 11 | improve 11: node 11's mean is strong but nDCG swings heavily by LightGBM seed, so bag the  | -- | -- | -- | failed | agent |
| 19 | 18 | debug 18: the 3-member bag exceeded the 900s limit, so average only two exact node-11 Ligh | 0.6546 | -- | +0.0095 | KEPT | agent |
| 20 | 19 | improve 19: seed-bagging helped, and the next cheap test is per-user z-score normalization | 0.6547 | -- | +0.0096 | KEPT | agent |
| 21 | 20 | improve 20: test another cheap fusion on the same cached LightGBM members by averaging wit | 0.6510 | -- | +0.0059 | KEPT | agent |
| 22 | 20 | improve 20: add a readable third member at 30% weight after per-user z-score fusion, but m | 0.6567 | -- | +0.0116 | KEPT | agent |
| 23 | 22 | improve 22: the fast member added complementary signal at 30%, so keep all cached member p | 0.6549 | -- | +0.0098 | KEPT | agent |
| 24 | 22 | improve 22: 30% fast-member weight improved over the 2-member z-fusion but 45% hurt, so ke | 0.6571 | -- | +0.0120 | KEPT | agent |
| 25 | 24 | draft feature-engineered member: change mechanism by adding a K-fold out-of-fold target/co | 0.6572 | -- | +0.0121 | KEPT | agent |
| 26 | 25 | draft user behaviour sequences: change mechanism from static LightGBM blending to a DIN-in | 0.6604 | -- | +0.0153 | KEPT | agent |
| 27 | 26 | improve 26: the history member moved nDCG strongly, so enrich the same mechanism with rece | -- | -- | -- | failed | agent |
| 28 | 26 | debug 27: the richer history implementation timed out, so keep node 26's cached sequence m | 0.6605 | -- | +0.0154 | KEPT | agent |
| 29 | 28 | draft multi-task: change mechanism by training a separate history LightGBM member on soft  | 0.6604 | -- | +0.0153 | KEPT | agent |
| 30 | 29 | improve 29: the auxiliary member lifted GAUC but hurt nDCG, so keep the same cached member | 0.6610 | -- | +0.0159 | KEPT | agent |
| 31 | 30 | draft user-history heuristic: change mechanism from learned history LightGBM to an explici | 0.6575 | -- | +0.0124 | KEPT | agent |
| 32 | 31 | debug 31: the explicit history-affinity member caused a large nDCG drop, so test whether i | 0.6549 | -- | +0.0098 | KEPT | agent |
| 33 | 30 | draft candidate-context features: change mechanism from more user history to non-label imp | 0.6749 | 0.0000 | +0.0298 | KEPT | agent |
| 34 | 33 | improve 33: candidate-context caused a large, stable gain but node 33 diluted it with the  | 0.6786 | 0.0012 | +0.0335 | KEPT | agent |
| 35 | 34 | improve 34: increasing candidate-context weight gave a clear gain, so push it to a majorit | 0.6819 | 0.0001 | +0.0368 | KEPT | agent |
| 36 | 35 | improve 35: node 35 improved when candidate-context became dominant, so test the next poin | 0.6809 | 0.0015 | +0.0358 | KEPT | agent |
| 37 | 35 | improve 35: node 36 showed 90% candidate-context is too high, so test the intermediate 80% | 0.6817 | 0.0004 | +0.0366 | KEPT | agent |
| 38 | 37 | improve 37: inspired by recommender feature-engineering practice around context/repeated-i | 0.6826 | 0.0007 | +0.0375 | KEPT | agent |
| 39 | 38 | improve 38: candidate-context is now carrying most of the score, so add a new readable can | -- | -- | -- | failed | agent |
| 40 | 39 | debug 39: the richer candidate-context member timed out, so keep node 38 intact and test t | 0.6827 | 0.0006 | +0.0376 | KEPT | agent |
