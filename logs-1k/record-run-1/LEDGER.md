# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`valid` is the **mean** validation primary across seeds; `+/-` is its standard
deviation across those seeds. `delta` is against the reproduced official
baseline (**0.6015**); a result counts only at **>= +0.002** (the official
epsilon). Two rows closer together than their `+/-` have not been told apart.

| # | parent | hypothesis | valid | +/- | delta | verdict | by |
|---|---|---|---|---|---|---|---|
| 1 | 001 | draft 1: replace pointwise BCE with same-user BPR pairwise loss, L=-log sigmoid(s_pos-s_ne | 0.6187 | 0.0013 | -0.0264 | worse | agent |
| 2 | 1 | improve 1: node 1's BPR works, so make each positive compete with 4 same-user sampled nega | 0.6240 | 0.0030 | -0.0211 | worse | agent |
| 3 | 2 | improve 2: multi-negative BPR improved nDCG, so try the standard sampled-softmax/listwise  | 0.6240 | 0.0037 | -0.0211 | worse | agent |
| 4 | 2 | draft 2: DIN uses target-conditioned history attention for CTR (https://arxiv.org/abs/1706 | 0.6343 | 0.0016 | -0.0108 | screen | agent |
| 5 | 4 | improve 4: the dev screen for causal same-author/video history buckets was healthy and imp | -- | -- | -- | duplicate | agent |
| 6 | 4 | improve 4: the dev screen for causal same-author/video history buckets was healthy and imp | 0.6270 | 0.0029 | -0.0181 | worse | agent |
| 7 | 6 | improve 6: history counts helped but were sparse, so add causal user-tab preference and gl | 0.6219 | 0.0038 | -0.0232 | worse | agent |
| 8 | 6 | draft 4: implicit-feedback ranking often weights observations by confidence (Hu/Koren/Voli | 0.6340 | 0.0030 | -0.0111 | screen | agent |
| 9 | 8 | improve 8: the watch-time weighted BPR passed the dev disaster screen, so run the same imp | -- | -- | -- | duplicate | agent |
| 10 | 8 | improve 8: node 8 was only a dev screen; this non-identical copy promotes the same watch-t | 0.6301 | 0.0020 | -0.0150 | worse | agent |
| 11 | 10 | draft 3: KuaiRand was proposed for multi-task learning with click/like/view-time feedback  | 0.6381 | 0.0011 | -0.0070 | screen | agent |
| 12 | 11 | improve 11: the MTL raw-feedback model passed the dev disaster screen, so promote the same | 0.6434 | 0.0006 | -0.0017 | noise | agent |
| 13 | 12 | improve 12: node 12 used auxiliary labels only for training; use a small fixed inference b | 0.6429 | 0.0002 | -0.0022 | worse | agent |
| 14 | 12 | improve 12: auxiliary heads help as regularization but direct inference blend hurt, so kee | 0.6434 | 0.0006 | -0.0017 | no-op | agent |
| 15 | 14 | debug 14: the aux_weight=0.05 change no-oped, so make the auxiliary term unmistakably acti | 0.6434 | 0.0006 | -0.0017 | no-op | agent |
| 16 | 15 | debug 15: diagnostics showed the raw CSVs were not under --data_dir, so robustly search ne | 0.6381 | 0.0011 | -0.0070 | no-op | agent |
| 17 | 12 | draft 6: Kaggle CTR practice commonly adds date/hour cyclic and drift features (e.g. Kaggl | 0.6373 | 0.0008 | -0.0078 | screen | agent |
| 18 | 17 | improve 17: dense date drift was weak; Kaggle CTR feature engineering often uses count/tar | 0.6281 | 0.0091 | -0.0170 | screen | agent |
| 19 | 12 | draft 5: test a non-FM tree ranker directly optimized for nDCG; CTR leaderboard practice u | -- | -- | -- | failed | agent |
| 20 | 19 | debug 19: node 19 crashed only because LightGBM caps query length at 10000, so split overs | 0.4446 | 0.0301 | -0.2005 | screen | agent |
| 21 | 20 | debug 20: the LambdaRank version was far below random, so isolate whether the bug is query | 0.3308 | 0.0632 | -0.3143 | screen | agent |
| 22 | 20 | debug 20: LightGBM binary also failed, so test the aggregate feature construction directly | 0.6173 | 0.0005 | -0.0278 | screen | agent |
| 23 | 21 | debug 21: node 22 proved the aggregate rates are aligned, so remove LightGBM categorical i | 0.3343 | 0.0001 | -0.3108 | screen | agent |
| 24 | 23 | debug 23: numeric LightGBM still stopped at iteration 1 using global AUC/logloss, so force | 0.3022 | 0.0061 | -0.3429 | screen | agent |
| 25 | 12 | improve 12: node 22's aggregate-rate heuristic is weaker but mechanistically different, so | 0.6387 | 0.0019 | -0.0064 | screen | agent |
| 26 | 12 | draft 8: KuaiRand's random-exposure log is intended for debiasing (https://arxiv.org/abs/2 | 0.6381 | 0.0011 | -0.0070 | no-op | agent |
| 27 | 26 | debug 26: node 26 no-oped because the random log was not found, so recursively search ance | 0.6381 | 0.0011 | -0.0070 | no-op | agent |
| 28 | 12 | improve 12: combine the best raw-feedback MTL regularizer with the ranking objective rathe | 0.6349 | 0.0011 | -0.0102 | screen | agent |
| 29 | 12 | improve 12: KuaiRand-Pure exposes many feedback signals for MTL (dataset page https://kuai | 0.6381 | 0.0011 | -0.0070 | no-op | agent |
| 30 | 29 | debug 29: node 29 no-oped because the raw logs were not under --data_dir; robustly search  | 0.6381 | 0.0011 | -0.0070 | no-op | agent |
| 31 | 12 | draft 5: DeepFM explicitly combines FM low-order terms with a DNN for high-order CTR inter | 0.6402 | 0.0034 | -0.0049 | screen | agent |
| 32 | 31 | improve 31: the full DeepFM tower increased variance, so make it a small residual on only  | 0.6367 | 0.0026 | -0.0084 | screen | agent |
| 33 | 12 | improve 12: node 12 is the strongest standalone model but has seed noise, so average three | -- | -- | -- | failed | agent |
| 34 | 33 | debug 33: the full three-member MTL ensemble timed out, so average the top checkpoints fro | 0.6422 | 0.0002 | -0.0029 | worse | agent |
| 35 | 12 | improve 12: tune the one component that clearly helped, the raw-feedback auxiliary regular | 0.6434 | 0.0006 | -0.0017 | no-op | agent |
| 36 | 35 | debug 35: node 35 no-oped because raw feedback logs were not under --data_dir; search like | 0.6381 | 0.0011 | -0.0070 | no-op | agent |
| 37 | 12 | draft 1: since metrics are per-user while BCE is row-weighted, train the FM with 1/sqrt(us | 0.6352 | 0.0010 | -0.0099 | screen | agent |
| 38 | 12 | draft 6: add leakage-safe smoothed target/count encoding buckets for video/author interact | 0.6350 | 0.0058 | -0.0101 | screen | agent |
| 39 | 12 | improve 12: node 12 is GAUC-heavy, so select checkpoints by validation nDCG@5 instead of p | 0.6434 | 0.0006 | -0.0017 | no-op | agent |
| 40 | 12 | draft 5: add explicit crossed categorical tokens to the best MTL FM so linear terms can me | -- | -- | -- | failed | agent |
