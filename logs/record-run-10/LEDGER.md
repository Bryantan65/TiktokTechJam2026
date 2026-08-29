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
| 2 | 1 | draft 1: replace pointwise BCE with same-user BPR pairwise loss so training directly order | 0.6025 | 0.0004 | +0.0010 | noise | agent |
| 3 | 2 | improve 2: pure BPR improved GAUC but may lose pointwise priors useful for top-5, so add a | 0.6019 | 0.0003 | +0.0004 | noise | agent |
| 4 | 2 | improve 2: try the standard ListNet/ListMLE-style user-level softmax objective, -log(sum e | 0.5966 | 0.0017 | -0.0049 | worse | agent |
| 5 | 2 | improve 2: node 2's random BPR has signal, so focus BPR on top-k mistakes with online same | 0.6023 | 0.0005 | +0.0008 | noise | agent |
| 6 | 3 | improve 3: mixed-gradient BCE+BPR underperformed, so keep the two objectives as separate e | 0.6033 | 0.0006 | +0.0018 | noise | agent |
| 7 | 6 | improve 6: the separate BCE+BPR rank blend is close to target; change only the blend to 60 | 0.6036 | 0.0007 | +0.0021 | KEPT | agent |
| 8 | 7 | draft 6: add a shared day-of-week categorical time feature to the current best BCE+BPR ran | 0.6035 | 0.0007 | +0.0020 | noise | agent |
| 9 | 8 | improve 8: weekday alone was too weak, so read raw CSV hourmin and add hour-of-day/4-hour  | 0.6037 | 0.0003 | +0.0022 | KEPT | agent |
| 10 | 9 | draft 3: replace the BCE member in the best rank blend with a shared-embedding multi-task  | 0.6033 | 0.0005 | +0.0018 | noise | agent |
| 11 | 9 | improve 9: hour features helped slightly, so add only low-cardinality context crosses tab× | 0.6031 | 0.0004 | +0.0016 | noise | agent |
| 12 | 9 | draft 2: replace the BCE member with a DIN-style attention model over recent same-user pos | -- | -- | -- | failed | agent |
| 13 | 12 | debug 12: node 12 crashed because frozen eval histories were stored in a plain dict and un | 0.6036 | 0.0005 | +0.0021 | KEPT | agent |
| 14 | 9 | draft 4: replace the BCE member with a censored log-normal watch-time FM trained from raw  | 0.6016 | 0.0007 | +0.0001 | noise | agent |
| 15 | 9 | draft 5: replace the BCE member with a LightGBM LambdaRank model grouped by user_id, while | 0.6041 | 0.0005 | +0.0026 | KEPT | agent |
| 16 | 15 | improve 15: LambdaRank helped as an independent member; append the trained BPR score as a  | 0.6032 | 0.0005 | +0.0017 | noise | agent |
| 17 | 15 | improve 15: LambdaRank is the best new mechanism; keep the BPR member and 60/40 rank blend | 0.6030 | 0.0005 | +0.0015 | noise | agent |
| 18 | 15 | improve 15: LambdaRank helped but only used raw categorical ids; add smoothed train-only l | 0.5728 | 0.0005 | -0.0287 | worse | agent |
| 19 | 15 | improve 15: the target-stat LambdaRank variant in 18 was a bad branch, so return to node 1 | 0.6046 | 0.0002 | +0.0030 | KEPT | agent |
| 20 | 19 | debug 18: the huge drop likely came from replacing raw categorical LambdaRank features wit | 0.5951 | 0.0020 | -0.0064 | worse | agent |
| 21 | 19 | improve 19: add non-leaky user behaviour summary features to the LambdaRank member only: p | 0.6045 | 0.0003 | +0.0030 | KEPT | agent |
| 22 | 19 | improve 19: keep node 19 fixed except swap the tree member from LambdaRank to LightGBM ran | 0.6046 | 0.0005 | +0.0031 | KEPT | agent |
| 23 | 22 | improve 22: train the XENDCG tree member on graded relevance from play_time_ms/duration qu | 0.6026 | 0.0002 | +0.0011 | noise | agent |
| 24 | 22 | improve 22: combine the two strongest tree objectives, rank_xendcg and lambdarank, as sepa | 0.6049 | 0.0004 | +0.0034 | KEPT | agent |
| 25 | 24 | improve 24: the dual-tree blend improved, so average two independently seeded LightGBM mod | 0.6046 | 0.0004 | +0.0031 | KEPT | agent |
| 26 | 24 | improve 24: dual rankers add complementary signal, so shift weight from the weaker BCE slo | 0.6046 | 0.0003 | +0.0031 | KEPT | agent |
| 27 | 24 | improve 24: the useful hour features may be missing absolute train-valid drift, so keep no | 0.6042 | 0.0003 | +0.0027 | KEPT | agent |
| 28 | 24 | improve 24: node 24's equal-spaced per-user rank blend optimizes broad ordering; square ea | 0.6048 | 0.0007 | +0.0033 | KEPT | agent |
| 29 | 24 | improve 24: replace hand blend with a readable learned combiner trained only on a temporal | 0.6037 | 0.0003 | +0.0022 | KEPT | agent |
| 30 | 24 | improve 24: node 28's rank^2 transform raised nDCG but lost robustness, so try a milder ra | 0.6048 | 0.0005 | +0.0033 | KEPT | agent |
| 31 | 24 | improve 24: the dual-tree members are the main complementary signal, so increase only Ligh | 0.6047 | 0.0001 | +0.0032 | KEPT | agent |
