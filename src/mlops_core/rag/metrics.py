"""How good a ranking is, judged against a question's labels.

A retrieved chunk is relevant when it contains a label's excerpt - whatever the cut - and
takes that label's grade. Each label is credited once: chunks overlap, so two neighbours
can both hold one excerpt, and counting both would reward returning the same passage
twice. A label whose excerpt no chunk holds any longer (a cut that split it) is reported
by the caller rather than silently counted as a miss.

With one label per question, as the drafted set has, several metrics coincide - average
precision is the reciprocal rank, recall at k is "was it in the top k" - and nDCG is a
discount on the rank. They part ways once pooled judgments add more graded labels, which
is why all four are kept.
"""

import math
from collections.abc import Sequence

from mlops_core.rag.questions import Judgment, contains

# The cut-offs reported. Ten is what an answer can be built from; one is what a single
# citation would be.
CUTOFFS = (1, 3, 5, 10)


def grades(retrieved: Sequence[tuple[str, str]], labels: Sequence[Judgment]) -> list[int]:
    """The grade each retrieved (document, text) earns, best label first, each label
    credited to the first chunk that holds it."""
    credited: set[int] = set()
    earned: list[int] = []
    for document, text in retrieved:
        held = [
            (label.grade, index)
            for index, label in enumerate(labels)
            if index not in credited
            and label.document_id == document
            and contains(text, label.excerpt)
        ]
        grade, index = max(held, default=(0, -1))
        if index >= 0:
            credited.add(index)
        earned.append(grade)
    return earned


def ranking_metrics(earned: Sequence[int], labels: Sequence[Judgment]) -> dict[str, float]:
    """Recall and nDCG at every cut-off, reciprocal rank and average precision, from the
    grades of a ranking and the labels it could have earned. Relevant means graded 1 or
    more; nDCG uses the grades themselves."""
    relevant = sum(label.grade >= 1 for label in labels)
    hits = [grade >= 1 for grade in earned]
    first = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    ideal = sorted((label.grade for label in labels), reverse=True)
    metrics = {
        "reciprocal_rank": 1 / first if first else 0.0,
        "average_precision": (
            sum(sum(hits[:rank]) / rank for rank, hit in enumerate(hits, start=1) if hit) / relevant
            if relevant
            else 0.0
        ),
    }
    for k in CUTOFFS:
        metrics[f"recall_at_{k}"] = sum(hits[:k]) / relevant if relevant else 0.0
        best = _dcg(ideal[:k])
        metrics[f"ndcg_at_{k}"] = _dcg(earned[:k]) / best if best else 0.0
    return metrics


def _dcg(grades: Sequence[int]) -> float:
    return sum((2.0**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
