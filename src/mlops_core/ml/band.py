"""A point and a range around it: what a regression with an `interval` serves.

Three pipelines fitted on the same rows: one for the point, two at the quantiles of the
range's edges. `predict` is the point, so whatever scores a model scores this one
unchanged; `band` is the range. Light on purpose - numpy and scikit-learn only - because
the prediction API loads it, and its image has nothing else.
"""

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Band(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """A point model and the two quantile models at the edges of its range."""

    def __init__(self, point: Any, lower: Any, upper: Any) -> None:
        self.point = point
        self.lower = lower
        self.upper = upper

    def fit(self, x: Any, y: Any, **fit_params: Any) -> Self:
        for model in (self.point, self.lower, self.upper):
            model.fit(x, y, **fit_params)
        return self

    def predict(self, x: Any) -> np.ndarray:
        prediction: np.ndarray = self.point.predict(x)
        return prediction

    def band(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """The range's edges. Two quantile models fitted apart can cross on a row; the
        range is then the two in order, never an upside-down one."""
        low, high = self.lower.predict(x), self.upper.predict(x)
        return np.minimum(low, high), np.maximum(low, high)


@dataclass(frozen=True)
class Predicted:
    """What a model says about some rows: a point each, and a range when it has one."""

    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


def predicted(model: Any, x: Any) -> Predicted:
    """Score rows with any served model, its range included when it has one."""
    if isinstance(model, Band):
        lower, upper = model.band(x)
        return Predicted(model.predict(x), lower, upper)
    return Predicted(model.predict(x))
