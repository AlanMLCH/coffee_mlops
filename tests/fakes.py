"""Test doubles shared across test modules.

`httpx` is imported lazily: the serving tests must run without the `data` extra.
"""

from collections.abc import Sized

import httpx
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# A DENUE-shaped answer for the CLI tests: three establishments, and a count to match.
# Every field the raw contract requires is here, with the shape the real service uses:
# text for everything, and entity+municipality+locality packed into `AreaGeo`. The
# coordinate sits inside the borough that `AreaGeo` declares in the recorded boundary
# fixture, so the spatial join has something to agree with.
DENUE_ESTABLISHMENTS = [
    {
        "Id": str(i),
        "Nombre": f"CAFE {i}",
        "Clase_actividad": "Cafeterías, fuentes de sodas, neverías, refresquerías y similares",
        "CLASE_ACTIVIDAD_ID": "722515",
        "AreaGeo": "090160001",
        "Estrato": "0 a 5 personas",
        "Latitud": "19.45",
        "Longitud": "-99.15",
    }
    for i in (1, 2, 3)
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

    def __init__(
        self,
        payloads: dict[str, bytes],
        redirects: dict[str, str],
        overpass: bytes | None = None,
    ) -> None:
        self.payloads = payloads
        self.redirects = redirects
        self.overpass = overpass

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in self.redirects:
            return httpx.Response(302, headers={"Location": self.redirects[url]})
        denue = denue_response(request.url.path)
        if denue is not None:
            return denue
        # Overpass carries the whole query in the query string, so this matches on the
        # path: a change to the query must not silently turn into a 404.
        if self.overpass is not None and request.url.path.endswith("/api/interpreter"):
            return httpx.Response(200, content=self.overpass)
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
