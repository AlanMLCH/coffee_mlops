"""Test doubles shared across test modules.

`httpx` is imported lazily: the serving tests must run without the `data` extra.
"""

import json
from collections.abc import Sized
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from domains.coffee.config import CoffeeConfig
from mlops_core.config import DomainConfig

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
    for i in (1, 2)
] + [
    # The Starbucks the recorded OSM answer also lists, at its node's coordinates: the
    # one place both registers share, so linking and the name rule's score have a case.
    {
        "Id": "3",
        "Nombre": "STARBUCKS",
        "Clase_actividad": "Cafeterías, fuentes de sodas, neverías, refresquerías y similares",
        "CLASE_ACTIVIDAD_ID": "722515",
        "AreaGeo": "090110001",
        "Estrato": "6 a 10 personas",
        "Latitud": "19.3495012",
        "Longitud": "-99.1969644",
    }
]


def denue_response(path: str) -> httpx.Response | None:
    """Answer the two DENUE endpoints the extractor calls, or None if it is not DENUE."""
    if "/Cuantificar/" in path:
        return httpx.Response(200, json=[{"AE": "722515", "AG": "09", "Total": "3"}])
    if "/BuscarAreaAct/" in path:
        start = int(path.split("/")[-4])
        return httpx.Response(200, json=DENUE_ESTABLISHMENTS if start == 1 else [])
    return None


def with_training[Config: DomainConfig](config: Config, model: str, **update: Any) -> Config:
    """The config with one model's training settings changed: a tiny tuning budget, say."""
    models = [
        m.model_copy(update={"training": m.training.model_copy(update=update)})
        if m.name == model
        else m
        for m in config.models
    ]
    return config.model_copy(update={"models": models})


def without_rate_limits(config: CoffeeConfig) -> CoffeeConfig:
    """The real domain config minus the politeness delays.

    The tests' server is a MockTransport: FAS alone is ~70 requests a pull, and at the
    real second apart every test that extracts would spend a minute being polite to
    nobody. Everything else - endpoints, pages, years, caching - stays as configured.
    """
    quick = {
        name: source.model_copy(update={"rate_limit_seconds": 0.0})
        for name in ("denue", "overpass", "fas", "roasters")
        if (source := getattr(config, name)) is not None
    }
    return config.model_copy(update=quick)


FIXTURES = Path(__file__).parent / "fixtures"


def fas_recording() -> dict[str, Any]:
    """The FAS answers recorded on 2026-09-21, trimmed to the PSD file fixture's keys."""
    recording: dict[str, Any] = json.loads(
        (FIXTURES / "fas_psd_coffee_sample.json").read_text(encoding="utf-8")
    )
    return recording


# FAS lookup endpoint -> the key it has in the recording.
FAS_LOOKUPS = {
    "commodities": "commodities",
    "commodityAttributes": "attributes",
    "unitsOfMeasure": "units",
    "countries": "countries",
}


def fas_response(request: httpx.Request, recording: dict[str, Any]) -> httpx.Response | None:
    """Answer like FAS from a recording, or None if the request is not for FAS.

    Refuses a request without the key header, as the real service does, so a test that
    passes is a test that sent it.
    """
    path = request.url.path
    if "/api/psd/" not in path:
        return None
    if not request.headers.get("X-Api-Key"):
        return httpx.Response(403, json={"error": {"code": "API_KEY_MISSING"}})
    endpoint = path.split("/api/psd/", 1)[1]
    if endpoint in FAS_LOOKUPS:
        return httpx.Response(200, json=recording[FAS_LOOKUPS[endpoint]])
    # commodity/<code>/country/all/year/<year>: a year with no data is an empty list.
    return httpx.Response(200, json=recording["years"].get(endpoint.rsplit("/", 1)[1], []))


SHOP_FIXTURES = FIXTURES / "roasters"
# Shop host -> the name its recordings are saved under.
SHOP_HOSTS = {
    "buna.mx": "buna",
    "almanegra.cafe": "almanegra",
    "cafeconjiribilla.com": "cafeconjiribilla",
    "cucuruchocafe.com": "cucurucho",
}


def shop_response(request: httpx.Request) -> httpx.Response | None:
    """Answer like the roasters' shops from their recordings, or None for another host.

    One catalog page is recorded per shop, so a second page comes back empty, as a
    short catalog's would. A product page that was not recorded is a 404.
    """
    prefix = SHOP_HOSTS.get(request.url.host)
    if prefix is None:
        return None
    path, params = request.url.path, request.url.params
    if path == "/robots.txt":
        return httpx.Response(
            200, text=(SHOP_FIXTURES / f"{request.url.host}.robots.txt").read_text(encoding="utf-8")
        )
    if path == "/products.json":
        recorded = SHOP_FIXTURES / f"{prefix}.products.json"
        body = (
            recorded.read_text(encoding="utf-8")
            if params.get("page") == "1"
            else '{"products": []}'
        )
        return httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "application/json"}
        )
    if path == "/tienda" and params.get("format") == "json":
        body = (SHOP_FIXTURES / f"{prefix}.tienda.json").read_text(encoding="utf-8")
        return httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "application/json"}
        )
    page = SHOP_FIXTURES / f"{prefix}.{path.removeprefix('/products/')}.html"
    if path.startswith("/products/") and page.is_file():
        return httpx.Response(200, text=page.read_text(encoding="utf-8"))
    return httpx.Response(404)


class RecordedServer:
    """Replays payloads by URL. Mutate `payloads` to simulate an upstream change."""

    def __init__(
        self,
        payloads: dict[str, bytes],
        redirects: dict[str, str],
        overpass: bytes | None = None,
        fas: dict[str, Any] | None = None,
    ) -> None:
        self.payloads = payloads
        self.redirects = redirects
        self.overpass = overpass
        self.fas = fas

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
        fas = None if self.fas is None else fas_response(request, self.fas)
        if fas is not None:
            return fas
        shop = shop_response(request)
        if shop is not None:
            return shop
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
