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
| 2 | 1 | draft loss: replace pointwise BCE with BPR over same-user positive/negative pairs, followi | 0.6032 | 0.0004 | +0.0017 | noise | agent |
| 3 | 2 | improve 2: BPR helped, so replace one-vs-one pairs with a sampled within-user softmax wher | 0.6035 | 0.0010 | +0.0020 | noise | agent |
| 4 | 3 | improve 3: sampled softmax hit the target mean but had high seed variance; remove negative | 0.5966 | 0.0009 | -0.0049 | worse | agent |
| 5 | 3 | improve 3: sampled softmax improved ranking but was noisy; increase same-user negatives fr | 0.6036 | 0.0005 | +0.0021 | KEPT | agent |
| 6 | 5 | improve 5: add a small BCE auxiliary term to the sampled-softmax candidate set, so the bes | 0.6035 | 0.0001 | +0.0020 | KEPT | agent |
| 7 | 5 | draft time features: add hour-of-day/day-of-week/hour×tab categorical context from raw log | 0.6043 | 0.0001 | +0.0028 | KEPT | agent |
| 8 | 7 | improve 7: time context gave a clear stable gain; add finer 10-minute bucket plus hour×dow | 0.6043 | 0.0001 | +0.0028 | KEPT | agent |
| 9 | 7 | draft multi-task: following MMoE-style shared representation for related recommender feedb | 0.6044 | 0.0002 | +0.0029 | KEPT | agent |
| 10 | 9 | improve 9: sparse action heads barely moved the model; add a dense standardized log(play_t | 0.6042 | 0.0001 | +0.0027 | KEPT | agent |
| 11 | 9 | draft user behaviour sequences: following DIN target-attention over user history (https:// | 0.6041 | 0.0002 | +0.0026 | KEPT | agent |
| 12 | 11 | improve 11: DIN attention may have destabilized the strong FM score; debug the sequence si | 0.6011 | 0.0005 | -0.0004 | noise | agent |
| 13 | 9 | improve 9: refine the strongest loss node by sampling half the within-user negatives from  | 0.6039 | 0.0006 | +0.0024 | KEPT | agent |
| 14 | 9 | improve 9: with convergence at risk, average two independently trained copies of the curre | 0.6043 | 0.0004 | +0.0028 | KEPT | agent |
| 15 | 9 | improve 9: node 9 optimizes one listwise row per positive, but primary includes unweighted | 0.6032 | 0.0003 | +0.0017 | noise | agent |
| 16 | 11 | improve 11: the learned DIN-style history branch was flat/unstable, so debug the same user | 0.6044 | 0.0001 | +0.0029 | KEPT | agent |
| 17 | 16 | improve 16: history affinities gave the only recent stable gain, so increase author/video  | 0.6039 | 0.0002 | +0.0024 | KEPT | agent |
| 18 | 16 | improve 16: node 16 trains/checkpoints on the FM score but deploys FM+history; choose earl | 0.6044 | 0.0001 | +0.0029 | no-op | agent |
| 19 | 16 | improve 16: node 17 showed stronger all-history weights hurt, so keep node 16's robust ble | 0.6045 | 0.0000 | +0.0030 | KEPT | agent |
| 20 | 19 | improve 19: the recent-history residual moved both metrics up but is nearly saturated; add | 0.6041 | 0.0001 | +0.0026 | KEPT | agent |
| 21 | 19 | draft watch-time modelling: following censored/watch-time ranking ideas (search: Censored  | 0.6045 | 0.0000 | +0.0030 | no-op | agent |
| 22 | 21 | debug 21: the watch-time pairwise loss was no-op, likely too weak or discarded by checkpoi | 0.6041 | 0.0002 | +0.0026 | KEPT | agent |
| 23 | 19 | draft different models: add a DeepFM-style MLP branch over the same field embeddings to ca | 0.6040 | 0.0005 | +0.0025 | KEPT | agent |
| 24 | 23 | improve 23: DeepFM had one strong seed but high variance, so stabilize it with a zero-star | 0.6038 | 0.0006 | +0.0023 | KEPT | agent |
| 25 | 19 | improve 19: the best node is still the listwise FM with history; make the final attempt a  | 0.6045 | 0.0001 | +0.0030 | KEPT | agent |
| 26 | 25 | improve 25: neg16 gave a small but stable gain, so test whether the same sampled-softmax a | 0.6042 | 0.0002 | +0.0027 | KEPT | agent |
| 27 | 25 | improve 25: neg32 saturated the loss, so spend the final high-risk pass on the strongest r | 0.6045 | 0.0001 | +0.0030 | KEPT | agent |
| 28 | 27 | improve 27: author-tab was flat, but it suggested residual scores can trade GAUC/nDCG; add | 0.6042 | 0.0003 | +0.0027 | KEPT | agent |
| 29 | 25 | improve 25: the strongest gains came from same-user sampled softmax; make the negative sam | 0.6044 | 0.0003 | +0.0029 | KEPT | agent |
| 30 | 27 | improve 27: with training-time changes saturated, use a label-free target-split exposure-f | 0.6045 | 0.0001 | +0.0030 | KEPT | agent |
| 31 | 25 | improve 25: exploit row order as a time proxy by adding a tiny train-learned user trend ti | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 32 | 31 | improve 31: the row-order user-trend residual improved nDCG; add a second tiny trend term  | 0.6045 | 0.0001 | +0.0030 | KEPT | agent |
