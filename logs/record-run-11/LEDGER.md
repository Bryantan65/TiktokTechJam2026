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
| 2 | 1 | draft 1: replace pointwise BCE with same-user BPR pairs so training directly compares posi | 0.6026 | 0.0002 | +0.0011 | noise | agent |
| 3 | 2 | improve 2: the BPR draft used only one sampled negative per positive; use 3 same-user nega | 0.6021 | 0.0007 | +0.0006 | noise | agent |
| 4 | 2 | improve 2: BPR helped but more sampled pairs did not; switch to a same-user listwise softm | 0.5993 | 0.0006 | -0.0022 | worse | agent |
| 5 | 2 | improve 2: pure BPR is still the best node; add a small BCE term during BPR epochs to reta | 0.6030 | 0.0003 | +0.0015 | noise | agent |
| 6 | 5 | improve 5: node 5's BCE anchor was computed on balanced pair rows, so it may not preserve  | 0.6024 | 0.0010 | +0.0009 | noise | agent |
| 7 | 5 | improve 5: node 5 is the strongest loss direction; replace uniform same-user negatives wit | 0.5877 | 0.0008 | -0.0138 | worse | agent |
| 8 | 7 | debug 7: the large drop likely came from always training on top-scoring negatives, includi | 0.6033 | 0.0003 | +0.0018 | noise | agent |
| 9 | 8 | improve 8: semi-hard sampling mainly raises nDCG@5, so add a light LambdaRank-style weight | 0.6033 | 0.0001 | +0.0018 | noise | agent |
| 10 | 9 | draft 6: add contextual weekday/hour/daypart embeddings from the raw log hourmin column on | 0.6039 | 0.0003 | +0.0024 | KEPT | agent |
| 11 | 10 | improve 10: node 10 showed time context helps; add half-hour and tab×half-hour embeddings  | 0.6031 | 0.0011 | +0.0016 | noise | agent |
| 12 | 10 | draft 3: add shared auxiliary engagement heads for click/like/follow/comment/forward on to | 0.6040 | 0.0001 | +0.0025 | KEPT | agent |
| 13 | 12 | improve 12: node 12's auxiliary heads helped GAUC, so add play_time_ms as a lightly weight | 0.6040 | 0.0003 | +0.0025 | KEPT | agent |
| 14 | 13 | draft 2: add DIN-inspired user behaviour history as candidate-specific prior-interaction f | 0.6038 | 0.0002 | +0.0023 | KEPT | agent |
| 15 | 14 | improve 14: the history buckets were too coarse and trained as noisy FM fields; replace th | 0.6035 | 0.0003 | +0.0020 | noise | agent |
| 16 | 13 | draft 5: change mechanism from pure FM capacity to DeepFM, adding a small MLP over field e | 0.6043 | 0.0004 | +0.0028 | KEPT | agent |
| 17 | 16 | improve 16: DeepFM helped but is noisy; make the MLP a learned small residual initialized  | 0.6043 | 0.0003 | +0.0028 | KEPT | agent |
| 18 | 17 | draft 4: use a censored log-normal watch-time auxiliary head and add its survival margin o | 0.6044 | 0.0005 | +0.0029 | KEPT | agent |
| 19 | 18 | improve 18: CWM only gave a small noisy gain; fit a frozen-score context calibrator on tra | 0.6044 | 0.0004 | +0.0029 | KEPT | agent |
| 20 | 19 | improve 19: the logistic calibrator only changes coarse offsets; train a LightGBM LambdaRa | 0.6046 | 0.0007 | +0.0031 | KEPT | agent |
| 21 | 20 | improve 20: LambdaMART helped nDCG but its global residual scale is noisy; center and scal | 0.6044 | 0.0005 | +0.0029 | KEPT | agent |
| 22 | 4 | debug 4: listwise softmax was a large negative, likely because the slate construction was  | 0.6038 | 0.0002 | +0.0023 | KEPT | agent |
| 23 | 14 | improve 14: the first history draft was weak as coarse FM fields; retest behaviour memory  | 0.6043 | 0.0003 | +0.0028 | KEPT | agent |
| 24 | 10 | improve 10: time features helped but all dates are still treated equally; test a stronger  | 0.6019 | 0.0007 | +0.0004 | noise | agent |
| 25 | 20 | improve 20: the last unresolved source of movement is seed variance (+/-0.000688); train t | 0.6046 | 0.0002 | +0.0031 | KEPT | agent |
| 26 | 20 | improve 20: node 20's LambdaMART residual uses hour/daypart but omits weekday/date even th | 0.6045 | 0.0004 | +0.0030 | KEPT | agent |
| 27 | 25 | improve 25: node 25 only ensembles seed variance of node20; node23's history residual was  | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 28 | 27 | improve 27: history residual complementarity helped; add node26's weekday/date residual as | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 29 | 27 | improve 27: node27 combines two LambdaMART residuals by z-scored margins; because nDCG@5 o | 0.6044 | 0.0001 | +0.0029 | KEPT | agent |
| 30 | 27 | improve 27: the best ensemble's carrying extra part is the history residual; refine it by  | 0.6045 | 0.0001 | +0.0030 | KEPT | agent |
