# MeLearn — three-minute demo narration

The PowerPoint contains the same narration in its speaker notes. Timing is a
target rather than a hard cut; record at a natural pace and use the remaining
seconds for transitions.

## Slide 1 — Hook (0:00–0:15)

MeLearn is machine learning that learns how to improve itself. With one
command, it ran thirty experiments, beat the required KuaiRand baseline, cost
three dollars and sixty-five cents, and required zero human interventions.

## Slide 2 — The problem (0:15–0:40)

Recommendation research is an iterative loop: form a hypothesis, write the
code, train, evaluate, and decide what comes next. Here, the agent had to rank
watch completion, beat the official factorization-machine baseline by point
zero zero two, and stay inside fifty experiments and six hours. Because seed
noise can consume most of that margin, a single promising score is not enough.

## Slide 3 — Demo (0:40–1:25)

We start MeLearn with one command. It reads every previous experiment from the
ledger, then uses tree search to select a promising parent. It states a
hypothesis, writes a complete Python solution, and hands that code to the
harness. The harness evaluates three seeds and returns only the validation
result. MeLearn records the score, keeps or rejects the idea, and starts again.
This footage compresses hours into seconds; after twenty-seven experiments, the
highlighted branch produced the winning model.

## Slide 4 — Trustworthy autonomy (1:25–1:55)

Autonomy is only useful if the evaluation is trustworthy. The ledger is
persistent memory, and UCT lets MeLearn revisit older branches instead of
endlessly refining its latest idea. The harness—not the model—owns the data,
official evaluator, seed policy, and time limits. Test labels are never visible,
invalid or duplicate solutions are rejected, and failures return as text so the
agent can diagnose and recover on its next turn.

## Slide 5 — Results (1:55–2:30)

On the required KuaiRand-Pure dataset, the official baseline scored point six
zero one six. MeLearn reached point six zero five four nine three: a gain of
point zero zero three nine four four, almost twice the required margin. On the
bonus one-thousand-user dataset, it improved the primary score by point zero
three seven eight nine four. The record run used thirty of fifty experiments,
two point four six million tokens, three dollars and sixty-five cents, no GPUs,
and no human intervention.

## Slide 6 — Discovery and close (2:30–3:00)

The largest result also revealed the central insight. In Pure, only three point
three eight percent of validation rows contain a user-creator pair seen during
training. In one-K, that overlap is thirty-three point seven percent—about ten
times larger. The performance gain grows by almost the same factor. MeLearn did
not just find a better model. It discovered where the dataset's learnable
signal lives: an autonomous researcher that measures, learns, and recovers.

