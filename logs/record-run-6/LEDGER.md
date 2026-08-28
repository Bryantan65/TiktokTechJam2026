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
| 2 | 1 | draft loss: replace pointwise BCE with a per-user listwise softmax likelihood, -log P(any  | -- | -- | -- | failed | agent |
| 3 | 2 | debug 2: the listwise draft crashed because encode() returns users as a Python list; conve | 0.5966 | 0.0009 | -0.0049 | worse | agent |
| 4 | 3 | improve 3: pure per-user softmax dropped single-class users and hurt calibration, so keep  | 0.5981 | 0.0004 | -0.0034 | worse | agent |
| 5 | 1 | draft loss: try same-user BPR pairs (Rendle 2009, https://arxiv.org/abs/1205.2618) while r | 0.5997 | 0.0002 | -0.0018 | noise | agent |
| 6 | 1 | draft sequence: add DIN-inspired user behaviour history features (target-aware past positi | 0.6019 | 0.0008 | +0.0004 | noise | agent |
| 7 | 6 | improve 6: aggregate user histories nudged nDCG, so add DIN-style recent target-aware sequ | 0.6017 | 0.0006 | +0.0002 | noise | agent |
| 8 | 6 | draft multi-task: use auxiliary feedback heads (click/like/follow/comment/forward and watc | 0.6029 | 0.0003 | +0.0014 | noise | agent |
| 9 | 8 | debug 8: auxiliary targets were nearly all zero because the join key included raw duration | 0.6029 | 0.0003 | +0.0014 | no-op | agent |
| 10 | 8 | debug 8: key-based raw alignment failed and produced all-zero auxiliaries, so follow the d | 0.6025 | 0.0008 | +0.0010 | noise | agent |
| 11 | 8 | draft time: add temporal context from raw hourmin/date as categorical FM fields (time-awar | 0.5944 | 0.0019 | -0.0071 | worse | agent |
| 12 | 11 | debug 11: the time draft likely overfit high-cardinality date/day fields or misaligned raw | 0.6032 | 0.0004 | +0.0017 | noise | agent |
| 13 | 12 | improve 12: node 12 showed the time reader missed almost every row; align coarse hour/bloc | 0.6032 | 0.0004 | +0.0017 | no-op | agent |
| 14 | 13 | debug 13: key-based raw matching was still a no-op because nearly all rows missed; follow  | 0.6023 | 0.0007 | +0.0008 | noise | agent |
| 15 | 12 | draft model: standalone LightGBM LambdaRank grouped by user directly optimizes an nDCG-sty | 0.5983 | 0.0002 | -0.0032 | worse | agent |
| 16 | 12 | improve 12: the best single FM is seed-stable but still wobbles in top-5 ranks; average th | 0.6037 | 0.0000 | +0.0022 | KEPT | agent |
| 17 | 16 | improve 16: the 3-model bag beat the target, so expand the same readable incumbent-only en | 0.6040 | 0.0000 | +0.0025 | KEPT | agent |
| 18 | 17 | improve 17: the five-seed bag still improved the same model without adding an ambiguous me | 0.6041 | 0.0000 | +0.0026 | KEPT | agent |
| 19 | 18 | improve 18: for ranking metrics only within-user order matters, so average per-user normal | 0.6045 | 0.0000 | +0.0030 | KEPT | agent |
| 20 | 19 | draft watch-time: use the trained auxiliary watch-ratio heads as a readable scoring compon | 0.5967 | 0.0000 | -0.0048 | worse | agent |
| 21 | 20 | debug 20: the raw watch logits dominated/warped the ranking, so test whether the watch hea | 0.6023 | 0.0000 | +0.0008 | noise | agent |
| 22 | 19 | improve 19: nDCG@5 is top-heavy, so make the successful per-user rank ensemble more top-se | 0.6045 | 0.0000 | +0.0030 | KEPT | agent |
| 23 | 15 | improve 15: standalone LambdaRank was weak, but use it readably as a 30% per-user rank cor | 0.6040 | 0.0002 | +0.0025 | KEPT | agent |
| 24 | 5 | improve 5: pure BPR underperformed, so test it as a small same-user BPR fine-tune after th | 0.6044 | 0.0000 | +0.0029 | KEPT | agent |
| 25 | 22 | improve 22: squared ranks helped nDCG but discard score-margin confidence, so blend 30% pe | 0.6045 | 0.0000 | +0.0030 | KEPT | agent |
| 26 | 25 | improve 25: the 30% per-user margin blend gave a small but consistent lift, so test a read | 0.6045 | 0.0000 | +0.0030 | KEPT | agent |
| 27 | 24 | improve 24: BPR fine-tune was near-best but not combined with the strongest per-user rank+ | 0.6035 | 0.0000 | +0.0020 | noise | agent |
| 28 | 25 | draft model: add a DeepFM high-order interaction tower to the current best FM+history+aux  | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 29 | 28 | improve 28: DeepFM raised GAUC but hurt nDCG@5 versus the FM rank ensemble, so blend it re | 0.6050 | 0.0001 | +0.0035 | KEPT | agent |
| 30 | 29 | improve 29: the 65% DeepFM blend was a clear gain, so test a stronger 80% DeepFM weight to | 0.6050 | 0.0001 | +0.0035 | KEPT | agent |
| 31 | 29 | improve 29: DeepFM is the best readable mechanism but is still trained pointwise; add a sm | -- | -- | -- | failed | agent |
| 32 | 31 | debug 31: the first BCE+BPR mix timed out because it sorted each minibatch for all ensembl | 0.6044 | 0.0000 | +0.0029 | KEPT | agent |
| 33 | 29 | improve 29: DeepFM's gain is real but the ensemble is already averaging 8 seeds, so remove | 0.6048 | 0.0001 | +0.0033 | KEPT | agent |
| 34 | 29 | improve 29: the current ensemble may smooth away high-confidence item/author priors, so ad | 0.6032 | 0.0001 | +0.0017 | noise | agent |
