"""The baselines as models the registry can serve.

What anyone could predict without a model - the training mean, a group's mean, a
constant, a rate per unit of exposure, and a range from the training target's quantiles -
fitted on the training split exactly as the gate fits them, and packaged so a model with
no champion is served the best of them instead of nothing. A mediocre model is better
than none; a model worse than the rule anyone would use is not, so when the candidate
loses to a baseline, the baseline is what is served.

Lookups and a rate, numpy and scikit-learn only: the prediction API loads them, and its
image has nothing else.
"""

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


def lookup_key(value: Any) -> str:
    """A group's value as a lookup reads it: the API casts numeric inputs to floats, so
    3 and 3.0 are one group."""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


class Lookup(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """A number for each value of one input column and `default` for any other - a
    group's mean, or an edge of its range - or `default` alone when no column is named:
    the overall mean, or a constant."""

    def __init__(
        self,
        column: str | None = None,
        values: dict[str, float] | None = None,
        default: float = 0.0,
    ) -> None:
        self.column = column
        self.values = values
        self.default = default

    def fit(self, x: Any, y: Any, **fit_params: Any) -> Self:
        return self  # fitted when built, from the training split, as the gate's are

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.column is None:
            return np.full(len(x), float(self.default))
        values = self.values or {}
        keys = x[self.column].tolist()
        return np.array([values.get(lookup_key(key), self.default) for key in keys], dtype=float)


class Rate(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """The training rate per unit of one input column, times the row's own: a count of
    something per resident, say. `default` where the row has no value."""

    def __init__(self, column: str = "", rate: float = 0.0, default: float = 0.0) -> None:
        self.column = column
        self.rate = rate
        self.default = default

    def fit(self, x: Any, y: Any, **fit_params: Any) -> Self:
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        units = pd.to_numeric(x[self.column], errors="coerce").to_numpy(dtype=float)
        scored: np.ndarray = np.where(np.isnan(units), self.default, units * self.rate)
        return scored
