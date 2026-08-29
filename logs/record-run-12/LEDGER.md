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
| 2 | 1 | draft 1: replace pointwise BCE with within-user BPR pairs, sampling positives against same | 0.6030 | 0.0005 | +0.0015 | noise | agent |
| 3 | 2 | improve 2: remove the BCE anchor so the model's selected checkpoint is trained only by sam | 0.6028 | 0.0004 | +0.0013 | noise | agent |
| 4 | 2 | improve 2: keep the anchored BPR loss but weight each positive pair by clipped log play_ti | 0.6030 | 0.0005 | +0.0015 | no-op | agent |
| 5 | 4 | debug 4: iteration 004 was a no-op, so the watch-time join probably fell back to unit weig | 0.6059 | 0.0001 | +0.0044 | screen | agent |
| 6 | 4 | debug 4: the dev screen proved non-unit watch-time weights reach training; run the row-ord | 0.6007 | 0.0007 | -0.0008 | noise | agent |
| 7 | 2 | draft ensemble from 2: test a readable 30% contribution from a pointwise BCE FM blended wi | 0.6033 | 0.0006 | +0.0018 | noise | agent |
| 8 | 7 | improve 7: node 7 improved both GAUC and nDCG but still has seed wobble; average the same  | 0.6036 | -- | +0.0021 | KEPT | agent |
| 9 | 8 | improve 8: change only the deterministic seed-bag blend weight to 60/40 BPR/BCE, reusing c | 0.6035 | -- | +0.0020 | KEPT | agent |
| 10 | 8 | improve 8: change only the deterministic seed-bag blend weight to 75/25 BPR/BCE, reusing c | 0.6036 | -- | +0.0021 | KEPT | agent |
| 11 | 8 | improve 8: keep the same cached seed-bagged BPR/BCE members but fuse per-user percentile r | 0.6040 | -- | +0.0025 | KEPT | agent |
| 12 | 11 | improve 11: with rank fusion outperforming z-score fusion, change only the rank blend weig | 0.6040 | -- | +0.0025 | KEPT | agent |
| 13 | 11 | improve 11: change only the per-user rank-fusion weight to 65/35 BPR/BCE, reusing unchange | 0.6040 | -- | +0.0025 | KEPT | agent |
| 14 | 11 | improve 11: change only the rank-fusion weight to 68/32 BPR/BCE, reusing cached members; 6 | 0.6040 | -- | +0.0025 | KEPT | agent |
| 15 | 11 | draft 3: add a multi-task auxiliary-feedback member (shared FM factors with separate BCE h | -- | -- | -- | failed | agent |
| 16 | 15 | debug 15: node 15 timed out from training three auxiliary BPR members per harness seed, so | 0.6035 | 0.0001 | +0.0020 | KEPT | agent |
| 17 | 11 | draft 9: append categorical video side information (music/tag/type/upload/size/music_type) | 0.6049 | -- | +0.0034 | KEPT | agent |
| 18 | 17 | improve 17: content side features gave the first clear jump, but they may smooth away usef | 0.6048 | -- | +0.0033 | KEPT | agent |
| 19 | 17 | improve 17: keep the content-feature member models fixed and change only fusion from perce | 0.6043 | -- | +0.0028 | KEPT | agent |
| 20 | 17 | draft 6: add raw CSV hourmin/date time context as a train-only smoothed prior blended at 2 | 0.6045 | -- | +0.0030 | KEPT | agent |
| 21 | 20 | debug 20: node 20 missed every raw hourmin row because tuple-key matching used incompatibl | 0.6039 | -- | +0.0024 | KEPT | agent |
| 22 | 17 | improve 17: node 21 fixed hourmin alignment but showed a hand-smoothed prior is the wrong  | 0.6048 | -- | +0.0033 | KEPT | agent |
| 23 | 17 | draft 5: test a new mechanism rather than another content-FM variant: DeepFM high-order in | 0.6049 | 0.0001 | +0.0034 | KEPT | agent |
| 24 | 23 | improve 23: DeepFM raised GAUC but hurt nDCG, so bag its three cached seeds and gate its b | 0.6051 | -- | +0.0036 | KEPT | agent |
| 25 | 24 | draft 14: add a learned LambdaRank tabular ranker at a readable 25% blend as a stacking-st | 0.6003 | 0.0001 | -0.0012 | noise | agent |
| 26 | 24 | draft 2: add a small train-only personalized behaviour prior (user-video/author/tag/music/ | 0.6052 | -- | +0.0037 | KEPT | agent |
| 27 | 26 | improve 26: the history prior helped, but flat 12% may over-trust sparse matches; gate the | 0.6050 | -- | +0.0035 | KEPT | agent |
| 28 | 26 | improve 26: keep node 26's effective 12% history-prior weight, but split one third of it t | 0.6051 | -- | +0.0036 | KEPT | agent |
| 29 | 26 | improve 26: node 26's personalized prior helps, so add a small 4% global item-quality prio | 0.6051 | -- | +0.0036 | KEPT | agent |
| 30 | 26 | improve 26: node 26 is a rank-average ensemble; test a readable 30% top-heavy reciprocal-r | 0.6045 | -- | +0.0030 | KEPT | agent |
| 31 | 26 | draft 8: train a readable 30% IPS-weighted BPR content-FM member using random-log vs stand | 0.6050 | -- | +0.0035 | KEPT | agent |
| 32 | 26 | draft 12: add a readable 30% LambdaRank-style BPR member that reweights same-user positive | -- | -- | -- | failed | agent |
| 33 | 32 | debug 32: the LambdaRank member crashed from a discount-array off-by-one, so fix rank clip | 0.6048 | -- | +0.0033 | KEPT | agent |
| 34 | 26 | draft transductive candidate-set prior from 26: use only the unlabeled target candidate li | 0.6000 | -- | -0.0015 | noise | agent |
