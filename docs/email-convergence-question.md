# Draft email — convergence rule questions

**Subject:** Track 2: convergence rule (ε = 0.002, N = 3)

---

Hi,

A question on the convergence rule before our final runs.

Read literally, N = 3 counts every scored iteration — including ones spent
refining an idea that already exists. Our latest run converged while it was still
improving: the last three iterations scored +0.0021, +0.0024 and +0.0031 over
baseline, but no single step was as large as ε, so the rule declared a plateau
and ended the run at 9 iterations. Steady incremental gain and being stuck look
identical to it.

Three questions:

1. Do iterations that crash or fail to score count toward N? We currently
   exclude them.
2. Do refinement iterations count toward N — i.e. is a run converged if the last
   three iterations were all spent improving one method, even when each scored
   better than the last?
3. Is a minimum number of iterations before the rule applies acceptable? Ours
   nearly terminated after three experiments, so we now require 8 scored
   iterations before convergence can be declared.



Thanks,
Bryan
Team ZuMianBao
