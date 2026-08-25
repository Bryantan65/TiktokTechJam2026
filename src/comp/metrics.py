"""Competition metrics: NDCG@K and Recall@K over a binary relevance label.

Both take `labels` already ordered by descending predicted score for one user.
Users with no positives are ambiguous under both metrics (IDCG=0, and a zero
denominator for recall), so both return None and let the aggregator apply the
configured policy.
"""
import numpy as np


def ndcg_at_k(labels, k):
    """labels: 1/0 array for one user, ordered by descending prediction."""
    labels = np.asarray(labels, dtype=np.float64)
    if labels.sum() == 0:
        return None

    discount = 1.0 / np.log2(np.arange(2, min(k, len(labels)) + 2))
    dcg = np.dot(labels[:k], discount)

    ideal = np.sort(labels)[::-1]
    idcg = np.dot(ideal[:k], discount)

    return dcg / idcg if idcg > 0 else None


def recall_at_k(labels, k):
    """Fraction of a user's positives that land in the top K. Order-blind inside K."""
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = labels.sum()
    if n_pos == 0:
        return None
    return labels[:k].sum() / n_pos


def aggregate(per_user, zero_positive='skip'):
    """Mean over users. `per_user` may contain None for users with no positives.

    zero_positive: 'skip'  -> exclude those users from the mean
                   'zero'  -> count them as 0.0
    """
    if zero_positive == 'skip':
        vals = [v for v in per_user if v is not None]
    elif zero_positive == 'zero':
        vals = [0.0 if v is None else v for v in per_user]
    else:
        raise ValueError("zero_positive must be 'skip' or 'zero', got %r" % zero_positive)

    if not vals:
        return float('nan')
    return float(np.mean(vals))
