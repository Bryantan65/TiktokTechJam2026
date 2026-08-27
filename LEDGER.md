# Experiment ledger

One line per experiment, read in full by the agent every iteration. Full
records in `logs/iterations/NNN.json`; the code for each is `solutions/NNN_*.py`.

`delta` is validation primary against the reproduced official baseline
(**0.6015**). A result counts only at **>= +0.002** (the official epsilon).

| # | parent | hypothesis | valid | delta | verdict | by |
|---|---|---|---|---|---|---|
| 1 | - | baseline: FM with pointwise logloss, autograd port | 0.6014 | -0.0001 | noise | human |
| 2 | 1 | ~~Implement BPR loss for pairwise ranking~~ **INVALID — bug, not a finding: label never used, pairs are arbitrary rows. Below random. Says nothing about pairwise ranking.** | 0.4970 | -0.1045 | invalid | agent |
| 3 | 1 | Correct the invalid pairwise attempt by explicitly using labels to construct BPR pairs as  | -- | -- | failed | agent |
| 4 | 3 | Bugfix of the corrected BPR experiment: convert encoder user outputs to NumPy arrays befor | 0.6014 | -0.0001 | noise | agent |
| 5 | 1 | A per-user listwise softmax fine-tune is more directly aligned with within-user ranking th | 0.6014 | -0.0001 | noise | agent |
| 6 | 1 | Add a small train-only smoothed target-encoding residual to a fixed near-peak FM score. Th | 0.6009 | -0.0006 | noise | agent |
| 7 | 1 | Auxiliary engagement actions can regularize the FM embeddings toward broader user-video pr | 0.6014 | -0.0001 | noise | agent |
| 8 | 1 | Explicit train-history aggregates may add user-behavior signal that FM embeddings smooth a | 0.6014 | -0.0001 | noise | agent |
| 9 | 8 | The user-history residual at weight 0.05 was slightly positive, so doubling the blend to 0 | 0.6014 | -0.0001 | noise | agent |
