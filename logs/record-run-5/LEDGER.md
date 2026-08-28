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
| 2 | 1 | draft loss: replace pointwise BCE with within-user BPR positive-negative pairs, following  | 0.6029 | 0.0006 | +0.0014 | noise | agent |
| 3 | 2 | improve 2: pure BPR improved ranking but has higher seed spread; initialize with 5 epochs  | 0.6013 | 0.0007 | -0.0002 | noise | agent |
| 4 | 2 | improve 2: test the other standard ranking loss from direction 1, a same-user listwise sof | 0.5989 | 0.0003 | -0.0026 | worse | agent |
| 5 | 2 | improve 2: BPR has the only positive signal, so use sampled hard negatives within the same | 0.5851 | 0.0013 | -0.0164 | worse | agent |
| 6 | 5 | debug 5: the hard max likely made early noisy negatives adversarial; average BPR over 4 in | 0.6034 | 0.0004 | +0.0019 | noise | agent |
| 7 | 6 | improve 6: K=4 random same-user negatives improved over single-pair BPR and fixed hard-neg | 0.6032 | 0.0003 | +0.0017 | noise | agent |
| 8 | 6 | improve 6: pure multi-negative BPR is best but ignores the stable calibration signal from  | 0.6028 | 0.0002 | +0.0013 | noise | agent |
| 9 | 6 | draft user behaviour sequences: following DIN's candidate-conditioned history idea (https: | 0.6022 | 0.0002 | +0.0007 | noise | agent |
| 10 | 6 | improve 6: the best multi-negative BPR is just below target and has meaningful seed wobble | 0.6037 | 0.0002 | +0.0022 | KEPT | agent |
| 11 | 10 | draft time features: add a leakage-safe weekday embedding to the kept 3-model BPR ensemble | 0.6041 | 0.0002 | +0.0026 | KEPT | agent |
| 12 | 11 | improve 11: weekday helped, so recover hourmin from the raw log CSVs and add hour-of-day a | -- | -- | -- | failed | agent |
| 13 | 12 | debug 12: node 12 crashed because log_standard lacks author_id, so key raw rows only by sh | 0.6050 | 0.0003 | +0.0035 | KEPT | agent |
| 14 | 13 | improve 13: hour gave a clear gain, so add finer half-hour and weekday×half-hour reusable  | 0.6048 | 0.0006 | +0.0033 | KEPT | agent |
| 15 | 13 | improve 13: half-hour over-sparsified, but tab has very different positive rates, so add a | 0.6052 | 0.0002 | +0.0037 | KEPT | agent |
| 16 | 15 | draft multi-task: following multi-task recommendation practice such as MMoE/shared-bottom  | 0.6048 | 0.0003 | +0.0033 | KEPT | agent |
| 17 | 9 | improve 9: the DIN draft underperformed, but sequence/user-history is not exhausted; add l | 0.6041 | 0.0002 | +0.0026 | KEPT | agent |
| 18 | 16 | improve 16: the first multi-task draft used sparse action labels and slightly regularized  | 0.6047 | 0.0005 | +0.0032 | KEPT | agent |
| 19 | 15 | improve 15: node 15 optimizes equal-per-positive BPR, which matches GAUC but underweights  | 0.6035 | 0.0005 | +0.0020 | noise | agent |
| 20 | 15 | improve 15: tab×hour was the best time cross, and a very small tab×weekday cross may captu | 0.6049 | 0.0004 | +0.0034 | KEPT | agent |
| 21 | 15 | improve 15: time features are the strongest added signal, so weight BPR positives toward r | 0.6029 | 0.0002 | +0.0014 | noise | agent |
| 22 | 15 | draft watch/user-history features: add leakage-safe per-user history signals (user-author  | -- | -- | -- | failed | agent |
| 23 | 22 | debug 22: node 22 timed out, so test whether the history mechanism itself has signal with  | 0.6049 | 0.0003 | +0.0034 | KEPT | agent |
| 24 | 15 | draft watch-time modelling: use raw play_time_ms as a bounded completion-based weight on p | 0.6041 | 0.0003 | +0.0026 | KEPT | agent |
| 25 | 15 | draft different model/features: add leakage-safe smoothed video/author target-CTR bins wit | 0.6042 | 0.0002 | +0.0027 | KEPT | agent |
| 26 | 15 | improve 15: replace one of the three identical tab-hour ensemble members with the nearly-a | 0.6054 | 0.0001 | +0.0039 | KEPT | agent |
| 27 | 26 | improve 26: the 33% weekday+hour member improved node 15 and reduced variance, so test whe | 0.6053 | 0.0001 | +0.0038 | KEPT | agent |
| 28 | 26 | improve 26: node 23's compact history model was close to node 15 and may be more complemen | 0.6051 | 0.0000 | +0.0036 | KEPT | agent |
