"""Paired bootstrap: how sure are we that one thing beats another on the same cases?

With a couple of hundred rows or a hundred questions, point metrics decide nothing: a
gap of 0.1 sits inside the noise. So comparisons are **paired** (the same rows, or the
same questions, for both sides) and bootstrapped, which lets a gate ask "how sure are
we?" instead of "which number is bigger?". A paired comparison is far more sensitive
than comparing two independent intervals, because the cases one side finds hard are
hard for both.

When the cases come in families (the sizes of one product), they are not independent
evidence: a model that gets a product wrong gets every size of it wrong. Resampling rows
would count one mistake several times and make every interval too narrow, so with
`groups` the bootstrap resamples whole groups instead (a cluster bootstrap).

Shared, not the model pipeline's: retrieval is gated the same way, and it must not
import the model pipeline to do it. numpy only.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Comparison:
    """How much better a candidate is than a reference, measured on the same cases."""

    # Mean paired difference, candidate minus reference, in the metric's own unit.
    difference: float
    ci_low: float
    ci_high: float
    probability_better: float

    def as_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_difference": self.difference,
            f"{prefix}_ci_low": self.ci_low,
            f"{prefix}_ci_high": self.ci_high,
            f"{prefix}_probability_better": self.probability_better,
        }


def bootstrap_means(
    values: np.ndarray, resamples: int, seed: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Means of `resamples` resamples drawn with replacement, all at once: of rows, or of
    whole groups when `groups` labels each row with its family."""
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = rng.integers(0, len(values), size=(resamples, len(values)))
        return values[draws].mean(axis=1)
    # Per-group sums and sizes, so a resample of groups is two sums, not a loop.
    _, group_of_row = np.unique(groups, return_inverse=True)
    sums = np.bincount(group_of_row, weights=values)
    sizes = np.bincount(group_of_row).astype(float)
    draws = rng.integers(0, len(sums), size=(resamples, len(sums)))
    means: np.ndarray = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return means


def compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    resamples: int = 5000,
    seed: int = 0,
    groups: np.ndarray | None = None,
    higher_is_better: bool = False,
) -> Comparison:
    """Paired bootstrap of `candidate - reference`, case by case or, with `groups`,
    family by family. By default the values are errors, so lower is better; a score
    (recall, nDCG) says `higher_is_better`."""
    difference = candidate - reference
    means = bootstrap_means(difference, resamples, seed, groups)
    wins = means > 0 if higher_is_better else means < 0
    return Comparison(
        difference=float(difference.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        # The candidate wins in this share of resamples.
        probability_better=float(wins.mean()),
    )
