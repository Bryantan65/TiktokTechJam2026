# Question to organisers — does a multi-run ensemble count as one submission?

Sent: (draft, 2026-08-29)

---

Subject: Track 2 — is an ensemble of independent agent runs within the compute rules?

Hi,

One question on the compute limits, which we would rather ask than assume.

The rules specify **50 iterations and a 6-hour wall clock per benchmark run**.
Our agent converges on the ε/N rule well inside both, typically around 30
experiments and 2-3 hours.

We have found that independent runs of the same agent produce genuinely
decorrelated models — their within-user rankings correlate 0.89-0.96 across
runs, against 0.98-0.99 between branches of a single run. Rank-averaging the
winners of five independent runs scores measurably better than any single run
(+0.0014 on valid).

We would like to know whether the following is within the rules:

> A system that launches N independent agent runs, each with fresh context and
> each individually inside the 50-iteration and 6-hour limits, and then
> rank-averages all N winners by a rule fixed in advance. No human selects
> anything; every run contributes, including the weak ones.

Our reading is that this is an architectural choice, in the same way an
ensemble is a modelling choice, and that each constituent run respects the
stated limits. But it does consume N times the total compute of a single run,
and if the intent is that a submission corresponds to exactly one benchmark
run, we would rather know now than submit something outside the spirit of the
rules.

If it is not permitted, we will submit the single-run result, which is what we
have been treating as the deliverable throughout.

Thanks,
Bryan Tan
