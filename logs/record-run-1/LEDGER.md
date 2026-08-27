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
| 2 | 1 | Replace pointwise BCE with same-user BPR, optimizing -log sigmoid(s_pos-s_neg) on positive | 0.6029 | 0.0007 | +0.0014 | noise | agent |
| 3 | 2 | Try a same-user listwise softmax loss: for each user, minimize logsumexp(all_scores)-mean( | 0.5989 | 0.0003 | -0.0026 | worse | agent |
| 4 | 2 | Add temporal context features (date and hour) to the current BPR-FM, motivated by temporal | 0.6026 | 0.0003 | +0.0011 | noise | agent |
| 5 | 2 | Train the BPR-FM with shared embeddings plus auxiliary heads for raw feedback labels (clic | 0.6027 | 0.0004 | +0.0012 | noise | agent |
| 6 | 2 | Add a DIN-style target-aware attention term over each user’s prior positive video history  | 0.6029 | 0.0006 | +0.0014 | noise | agent |
| 7 | 2 | Try a watch-time/CWM-inspired pairwise objective using raw play_time_ms versus duration_ms | 0.5640 | 0.0005 | -0.0375 | worse | agent |
| 8 | 2 | Try DeepFM with same-user BPR: the DeepFM paper combines an FM component with a deep compo | 0.6031 | 0.0005 | +0.0016 | noise | agent |
