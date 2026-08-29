# Working assumption: may a Pure submission train on KuaiRand-1k?

**Decided 2026-08-29. Written before acting on it, not after.**

We asked the organisers whether a KuaiRand-Pure submission may use KuaiRand-1k
as auxiliary training data. The reply (Phyllis Chua, 17:26) was that engineer
capacity is exhausted and any answer would come after the deadline, with the
guidance to *"read the problem statement carefully, then make an assessment on
the suitability of your approach and working assumptions."*

This file is that assessment.

## The governing text

From `track2-problem-statement.md`, under **Resource policy**:

> There is one hard rule: **no external training data.** Training must rely only
> on the KuaiRand datasets listed below - no augmenting, joining, or
> pre-training on any other dataset, and no pretrained model whose weights were
> trained on these benchmarks' test labels. This single rule is what keeps the
> hidden-test ranking fair; everything else is unrestricted.

The table immediately below lists **one** dataset entry, KuaiRand, described as
*"Three released variants: KuaiRand-Pure is required, while KuaiRand-1k and
KuaiRand-27k are bonus."*

## Our reading

**Permitted.** The prohibited operations - augmenting, joining, pre-training -
are prohibited *on any other dataset*. KuaiRand-1k is not another dataset; it is
one of the three listed variants of the dataset we were given. The sentence
closes with an explicit catch-all, *"everything else is unrestricted."*

We considered the opposite reading, which infers separateness from the fact that
1k and 27k are called *bonus benchmarks* with their own submissions, and from the
Primary metric being a delta over a baseline computed on Pure alone. That
inference is real but it is an inference. The resource policy is explicit text on
exactly this question, and explicit text governs.

**With one hard exclusion.** The same rule forbids weights trained on *"these
benchmarks' test labels."* KuaiRand-1k ships two standard logs:

```
log_standard_4_08_to_4_21_1k.csv   dates 20220408..20220421   = Pure's train window
log_standard_4_22_to_5_08_1k.csv   covers Pure's valid AND test windows
```

Measured on the real files: 993 of 1k's 1,000 users are also Pure users, and 879
of them appear in Pure's **test** split. The second file therefore carries
evaluation-period labels for users we are scored on. **It is excluded**, as is
every 27k file covering 04-22 onward.

## What we will and will not do

| | |
| --- | --- |
| use `log_standard_4_08_to_4_21_1k.csv` for training | **yes** - Pure's own train window, no evaluation-period rows |
| use `log_standard_4_22_to_5_08_1k.csv` | **no** - valid and test labels for scored users |
| use `log_random_*` from any variant | **no** - spans the evaluation window |
| train on Pure's own `valid` split | **no** - unchanged from every previous run |

## Why this is conservative where it matters

The rule's stated purpose is *"what keeps the hidden-test ranking fair."* The
exclusion above is drawn at the evaluation window rather than at the dataset
boundary, which is the line the rule actually cares about. No model we submit
will have seen a label from 04-22 or later, on any variant, which is the same
discipline every run so far has followed.

## What it is worth

Bounded by coverage. The 1k users are 3.7% of Pure's 27,077, so even a large
improvement on them is capped:

```
for the 963 shared users, in Pure's train window
   Pure gives them   median     30 rows
   1k   gives them   median  3,593 rows      120x
```

Expected effect on the primary metric: **+0.001 to +0.002**. Real, and roughly a
third of the agent's entire eleven-run gain, but not transformative - the
constraint is that 96.3% of Pure's users gain nothing, because 1k does not
contain them. KuaiRand-27k would cover all of them and is the version that would
actually matter; at 322M interactions it is out of reach on this hardware and
timeline.

## If the organisers rule otherwise

The Pure submission of record is `logs/record-run-3/solutions/027_deepfm_member.py`,
trained on Pure alone, valid 0.605493 and test 0.598508. It is unaffected by any
of this and remains submittable as-is.
