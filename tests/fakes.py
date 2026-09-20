"""Test doubles shared across test modules.

`httpx` is imported lazily: the serving tests must run without the `data` extra.
"""

from collections.abc import Sized

import httpx
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# A DENUE-shaped answer for the CLI tests: three establishments, and a count to match.
DENUE_ESTABLISHMENTS = [
    {"Id": str(i), "Nombre": f"CAFE {i}", "Latitud": "19.4", "Longitud": "-99.1"} for i in (1, 2, 3)
]


def denue_response(path: str) -> httpx.Response | None:
    """Answer the two DENUE endpoints the extractor calls, or None if it is not DENUE."""
    if "/Cuantificar/" in path:
        return httpx.Response(200, json=[{"AE": "722515", "AG": "09", "Total": "3"}])
    if "/BuscarAreaAct/" in path:
        start = int(path.split("/")[-4])
        return httpx.Response(200, json=DENUE_ESTABLISHMENTS if start == 1 else [])
    return None


class RecordedServer:
    """Replays payloads by URL. Mutate `payloads` to simulate an upstream change."""

    def __init__(self, payloads: dict[str, bytes], redirects: dict[str, str]) -> None:
        self.payloads = payloads
        self.redirects = redirects

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in self.redirects:
            return httpx.Response(302, headers={"Location": self.redirects[url]})
        denue = denue_response(request.url.path)
        if denue is not None:
            return denue
        if url in self.payloads:
            return httpx.Response(
                200,
                content=self.payloads[url],
                headers={"Last-Modified": "Wed, 22 Jul 2026 19:02:11 GMT"},
            )
        return httpx.Response(404)


class ConstantModel(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """A champion that ignores its input, so permutation importance comes out at zero.

    Inherits the sklearn bases because `permutation_importance` inspects estimator tags
    and refuses anything that only looks like an estimator.
    """

    def __init__(self, prediction: float = 82.0) -> None:
        self.prediction = prediction

    def fit(self, x: object, y: object = None) -> "ConstantModel":
        return self

    def predict(self, x: Sized) -> np.ndarray:
        return np.full(len(x), self.prediction)

    def __sklearn_is_fitted__(self) -> bool:
        return True
