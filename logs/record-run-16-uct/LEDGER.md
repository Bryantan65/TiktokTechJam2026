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
| 2 | 1 | draft 1: replace pointwise BCE with same-user BPR pairs, following Rendle et al.'s BPR obj | 0.6035 | 0.0002 | +0.0020 | KEPT | agent |
| 3 | 2 | improve 2: sample three same-user negatives per positive instead of one, so BPR sees a bro | 0.6031 | 0.0004 | +0.0016 | noise | agent |
| 4 | 2 | improve 2: replace sampled BPR with full same-user multi-positive softmax CE, n_pos*logsum | 0.6031 | 0.0002 | +0.0016 | noise | agent |
| 5 | 4 | improve 4: lower the same-user softmax temperature (T=0.5) so the listwise objective focus | 0.6024 | 0.0007 | +0.0009 | noise | agent |
| 6 | 3 | improve 3: keep the 3-negative BPR mechanism but weight each positive's pairs by capped pl | 0.6031 | 0.0004 | +0.0016 | no-op | agent |
| 7 | 2 | improve 2: sample five same-user negatives but backprop only through the currently highest | 0.5792 | 0.0024 | -0.0223 | worse | agent |
| 8 | 7 | debug 7: pure max hard-negative BPR likely collapsed because one noisy early negative got  | 0.6033 | 0.0002 | +0.0018 | noise | agent |
| 9 | 6 | debug 6: the watch-time weighted BPR run was a no-op, likely because the raw CSV join used | 0.6030 | 0.0002 | +0.0015 | noise | agent |
| 10 | 8 | improve 8: soft-hard BPR is close to node 2 but may rank different users' top items; follo | 0.6038 | 0.0002 | +0.0023 | KEPT | agent |
| 11 | 10 | improve 10: keep the two mechanisms and 60/40 per-user-z fusion fixed, but bag member seed | 0.6042 | -- | +0.0027 | KEPT | agent |
| 12 | 11 | improve 11: the seed-bagged score average improved both metrics; add a 30% per-user rank-p | 0.6046 | -- | +0.0031 | KEPT | agent |
| 13 | 12 | improve 12: keep the cached six-member ensemble but replace linear rank-percentile fusion  | 0.6042 | -- | +0.0027 | KEPT | agent |
| 14 | 13 | improve 13: RRF showed hand-crafted rank fusion is sensitive; train a small LightGBM Lambd | 0.6041 | -- | +0.0026 | KEPT | agent |
| 15 | 12 | draft 2: add a DIN-inspired user behaviour history signal (DIN paper https://arxiv.org/abs | 0.6027 | -- | +0.0012 | noise | agent |
| 16 | 14 | improve 14: node 014's stacker may overfit because it trained on in-sample TRAIN member pr | 0.6045 | -- | +0.0030 | KEPT | agent |
| 17 | 12 | improve 12: node 012's 30% rank-percentile fusion was the largest clean gain after ensembl | 0.6047 | -- | +0.0032 | KEPT | agent |
| 18 | 17 | improve 17: rank-percentile weight improved from 30% to 40%, so keep all cached members fi | 0.6047 | -- | +0.0032 | KEPT | agent |
| 19 | 18 | improve 18: add a readable 10% train-only historical residual prior for repeated user-vide | 0.6048 | -- | +0.0033 | KEPT | agent |
| 20 | 19 | improve 19: the train-only residual prior helped, so make it stronger and add smoothed glo | 0.6041 | -- | +0.0026 | KEPT | agent |
| 21 | 16 | improve 16: expand the untouched OOF LambdaRank stacker by training/applying it on the str | 0.6045 | -- | +0.0030 | KEPT | agent |
| 22 | 19 | improve 19: node 020 showed stronger history/global priors hurt, so keep node-019's 10% us | 0.6047 | -- | +0.0032 | KEPT | agent |
| 23 | 19 | improve 19: the rank/history knobs appear locally peaked, so test a different high-leverag | 0.6044 | -- | +0.0029 | KEPT | agent |
| 24 | 22 | improve 22: node 022's stronger 60% rank blend hurt slightly, likely because plain percent | 0.6048 | -- | +0.0033 | KEPT | agent |
| 25 | 24 | improve 24: p^2 rank fusion improved node 022 by top-emphasizing ranks; increase the rank- | 0.6046 | -- | +0.0030 | KEPT | agent |
| 26 | 24 | improve 24: target/user-history encodings are common in ranking blends (e.g. Kaggle TE gui | 0.6045 | -- | +0.0030 | KEPT | agent |
| 27 | 24 | draft item-CF: change mechanism rather than another near-tie blend tweak; following item-b | 0.6032 | -- | +0.0017 | noise | agent |
| 28 | 27 | debug 27: the 20% positive item-CF author co-occurrence blend caused a large GAUC drop, so | 0.6042 | -- | +0.0027 | KEPT | agent |
| 29 | 24 | draft time features: branch from the best despite UCT because the convergence watch needs  | -- | -- | -- | failed | agent |
| 30 | 29 | debug 29: the time-feature draft timed out in the raw CSV hourmin join, so keep the time h | 0.6046 | -- | +0.0031 | KEPT | agent |
| 31 | 30 | improve 30: tuple-only weekday/date residual was weak, so test the intended time feature w | 0.6048 | -- | +0.0033 | no-op | agent |
| 32 | 31 | debug 31: the hourmin residual was a no-op, so diagnose the join by normalizing CSV tuple  | 0.6047 | -- | +0.0032 | KEPT | agent |
| 33 | 32 | improve 32: the hour join is now live but the positive residual hurt, so test whether the  | 0.6047 | -- | +0.0032 | KEPT | agent |
