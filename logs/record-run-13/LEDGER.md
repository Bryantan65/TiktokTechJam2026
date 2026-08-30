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
| 2 | 1 | draft loss: replace pointwise BCE with same-user BPR pairs so positives are optimized abov | 0.6026 | 0.0007 | +0.0011 | noise | agent |
| 3 | 2 | improve 2: BPR improved ranking but had higher seed spread, so add a small BCE term on sam | 0.6025 | 0.0007 | +0.0010 | noise | agent |
| 4 | 2 | improve 2: uniform BPR negatives may waste updates on easy negatives; sample a mixture wit | 0.5735 | 0.0017 | -0.0280 | worse | agent |
| 5 | 4 | debug 4: the large drop likely came from mining mostly hard negatives from an untrained/ea | 0.5966 | 0.0014 | -0.0049 | worse | agent |
| 6 | 2 | improve 2: BPR is the best but noisy while BCE is stable and learns a different pointwise  | 0.6031 | 0.0008 | +0.0016 | noise | agent |
| 7 | 6 | improve 6: node 6 showed BCE and BPR complement after per-user z-scoring; keep the same ca | 0.6033 | 0.0008 | +0.0018 | noise | agent |
| 8 | 7 | improve 7: keep the same cached BCE/BPR members but add a 30% per-user rank-percentile fus | 0.6035 | 0.0006 | +0.0020 | noise | agent |
| 9 | 8 | improve 8: node 8's rank-percentile term was the only cheap change still moving both metri | 0.6035 | 0.0005 | +0.0020 | noise | agent |
| 10 | 8 | improve 8: node 8 is strong but seed-noisy; bag node-8 predictions across member seeds 0/1 | 0.6041 | 0.0000 | +0.0026 | KEPT | agent |
| 11 | 10 | improve 10: the 3-seed bag won by variance reduction; average the same unchanged node-8 me | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 12 | 11 | improve 11: five-seed bagging gave a clear gain, so extend the same unchanged node-8 seed- | 0.6044 | 0.0000 | +0.0029 | KEPT | agent |
| 13 | 11 | improve 11: node 12 showed extra seeds can dilute, while node 11's gain came from adding s | 0.6047 | 0.0000 | +0.0032 | KEPT | agent |
| 14 | 13 | draft time features: KuaiRand includes timestamp/hourmin fields (official repo https://git | 0.6109 | 0.0000 | +0.0094 | screen | agent |
| 15 | 14 | improve 14: the dev screen verified that raw hourmin joins hit every row, so spend the cou | -- | -- | -- | duplicate | agent |
| 16 | 14 | improve 14: the dev screen verified that raw hourmin joins hit every row, so spend the cou | 0.6046 | 0.0000 | +0.0031 | KEPT | agent |
| 17 | 13 | draft user behaviour sequences: DIN weights historical behaviours by relevance to the targ | 0.6035 | 0.0000 | +0.0020 | KEPT | agent |
| 18 | 17 | improve 17: the first history proxy was diluted by global popularity backoffs and a heavy  | 0.6035 | 0.0000 | +0.0020 | noise | agent |
| 19 | 13 | draft watch-time modelling: ABPR-style confidence weighting for heterogeneous implicit fee | 0.6044 | 0.0000 | +0.0029 | KEPT | agent |
| 20 | 19 | debug 19: node 19 printed zero play-time hits, so the watch-time member was not actually w | 0.6046 | 0.0000 | +0.0031 | KEPT | agent |
| 21 | 13 | draft loss: listwise softmax over each user's impressions directly optimizes within-user p | 0.6042 | 0.0001 | +0.0027 | KEPT | agent |
| 22 | 13 | draft multi-task: ESMM-style auxiliary supervision for user actions (e.g. https://arxiv.or | 0.6043 | 0.0001 | +0.0028 | KEPT | agent |
| 23 | 17 | improve 17: the previous history proxy was too diluted; use explicit train-only user-condi | 0.6040 | 0.0000 | +0.0025 | KEPT | agent |
| 24 | 16 | improve 16: time features were slightly below node 13 standalone but may make different er | 0.6051 | 0.0000 | +0.0036 | KEPT | agent |
| 25 | 24 | improve 24: the 40% time-aware blend improved both metrics, so move further along the same | 0.6051 | 0.0000 | +0.0036 | KEPT | agent |
| 26 | 25 | draft different models: add a standalone LightGBM LambdaRank member at 30% weight, using c | 0.6039 | 0.0002 | +0.0024 | KEPT | agent |
| 27 | 23 | improve 23: the history/target-encoding proxy was only tested away from the best time ense | 0.6043 | 0.0000 | +0.0028 | KEPT | agent |
| 28 | 25 | improve 25: the time-aware blend is the current best and only coarse 4-hour buckets were t | 0.6050 | 0.0000 | +0.0035 | KEPT | agent |
| 29 | 25 | improve 25: node 28 suggests adding sharper time categories trades GAUC for nDCG; instead  | 0.6054 | 0.0000 | +0.0039 | KEPT | agent |
| 30 | 29 | improve 29: date-free time clearly improved the blend, and node 25's 50% setting was inher | 0.6055 | 0.0000 | +0.0040 | KEPT | agent |
| 31 | 30 | improve 30: tab has very different base positive rates by product surface, so hour effects | 0.6048 | 0.0000 | +0.0033 | KEPT | agent |
| 32 | 30 | improve 30: node 31 showed crosses hurt, but node 28 hinted sharper time helped nDCG; repl | 0.6053 | 0.0000 | +0.0038 | KEPT | agent |
