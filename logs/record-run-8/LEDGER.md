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
| 2 | 1 | draft loss: replace pointwise BCE with within-user BPR, sampling one same-user negative pe | 0.6035 | 0.0002 | +0.0020 | KEPT | agent |
| 3 | 2 | improve 2: keep BPR as the main objective but add a small balanced BCE term on the same po | 0.6036 | 0.0001 | +0.0021 | KEPT | agent |
| 4 | 3 | improve 3: replace one-negative BPR with a sampled listwise softmax over one positive and  | 0.6030 | 0.0003 | +0.0015 | noise | agent |
| 5 | 3 | draft multi-task: add auxiliary click/like/follow/comment/forward heads sharing the FM emb | 0.6029 | 0.0002 | +0.0014 | noise | agent |
| 6 | 5 | debug 5: auxiliary alignment missed almost every row because it keyed on video-feature fie | 0.6029 | 0.0002 | +0.0014 | noise | agent |
| 7 | 3 | draft time features: append hour/hour-bucket/tab-hour features from raw log hourmin, follo | 0.6027 | 0.0004 | +0.0012 | noise | agent |
| 8 | 3 | improve 3: focus the successful BPR+BCE loss on harder same-user mistakes by sampling 35%  | 0.6012 | 0.0008 | -0.0003 | noise | agent |
| 9 | 3 | improve 3: add explicit user-video/user-author/user-tab cross fields to the best BPR+BCE F | 0.6005 | 0.0005 | -0.0010 | noise | agent |
| 10 | 3 | draft user behaviour sequences: following DIN's attention-over-history idea (Deep Interest | 0.6023 | 0.0002 | +0.0008 | noise | agent |
| 11 | 3 | improve 3: change the BPR+BCE sampler from positive-count-weighted to partially user-balan | 0.6011 | 0.0002 | -0.0004 | noise | agent |
| 12 | 3 | improve 3: average three independently seeded copies of the current best BPR+BCE FM logits | 0.6042 | 0.0002 | +0.0027 | KEPT | agent |
| 13 | 12 | improve 12: train five BPR+BCE members and average within-user percentile ranks instead of | 0.6047 | 0.0003 | +0.0032 | KEPT | agent |
| 14 | 13 | draft different models: add a standalone LightGBM LambdaRank/NDCG@5 ranker as a 30% within | -- | -- | -- | failed | agent |
| 15 | 14 | debug 14: node 14 crashed before training because the starter-kit path used os.path.dirnam | 0.6036 | 0.0002 | +0.0021 | KEPT | agent |
| 16 | 13 | improve 13: BPR optimizes AUC while nDCG@5 cares most about the very top; keep the five BP | 0.6046 | 0.0003 | +0.0031 | KEPT | agent |
| 17 | 15 | improve 15: the independent LambdaRank ranker degraded the strong FM ensemble; use the FM  | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 18 | 17 | draft watch-time modelling: use raw play_time_ms as a censored-engagement signal by traini | 0.6039 | 0.0003 | +0.0024 | KEPT | agent |
| 19 | 18 | debug 18: the watch-time join had zero hits because log_standard lacks or differs on autho | 0.6018 | 0.0005 | +0.0003 | noise | agent |
| 20 | 17 | improve 17: the time-feature FM draft was weak, but hour/date drift may be useful as a tre | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 21 | 20 | improve 20: LightGBM residual with time features moved nDCG@5, so focus the LambdaRank gra | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 22 | 20 | improve 20: the residual time tree is useful but is built on five seed-only FM members; di | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 23 | 20 | improve 20: the FM ensemble is stable, but the single residual LambdaRank tree is stochast | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 24 | 20 | improve 20: add leave-one-out personalized history rates (user-video/author/tab/duration/a | 0.6048 | 0.0002 | +0.0033 | KEPT | agent |
| 25 | 24 | improve 24: node 24's aggregate user-history rates helped slightly; add compact sequence/r | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 26 | 24 | improve 24: expose the five individual FM rank members plus their disagreement to the Lamb | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 27 | 24 | improve 24: the best node is still variance-limited by the FM rank ensemble; increase the  | 0.6047 | 0.0002 | +0.0032 | KEPT | agent |
| 28 | 24 | improve 24: switch the personalized residual tree from pairwise LambdaRank to LightGBM's X | 0.6046 | 0.0001 | +0.0031 | KEPT | agent |
| 29 | 24 | improve 24: multi-task heads were weak, but raw auxiliary feedback may help as smoothed us | 0.6046 | 0.0003 | +0.0031 | KEPT | agent |
| 30 | 24 | improve 24: node 24 is the best but its residual LambdaRank tree is a single stochastic mo | 0.6047 | 0.0002 | +0.0031 | KEPT | agent |
| 31 | 24 | debug 24: node 24 may overfit train labels because global video/author/tab target encoding | 0.6047 | 0.0003 | +0.0032 | KEPT | agent |
