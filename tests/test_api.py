from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

import domains.coffee
from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.roaster_sheets import clean_roasters
from mlops_core.config import DomainConfig, Settings
from mlops_core.ml.registry import ServedModel
from mlops_core.serving import api
from mlops_core.storage import write_table

PREDICT = "/models/review/predict"
LOT = {
    "country": "Mexico",
    "variety": "bourbon",
    "processing_method": "washed",
    "color": "green",
    "altitude_m": 1500,
    "moisture_pct": 11.2,
    "category_one_defects": 0,
    "category_two_defects": 2,
    "graded_on": "2023-05-01",
}


class RecordingModel:
    """Stands in for the champion and remembers what the API asked it to predict."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict(self, x: pd.DataFrame) -> list[float]:
        self.seen = x
        return [83.5]


@pytest.fixture
def market_context(tmp_path: Path) -> pl.DataFrame:
    context = pl.DataFrame(
        {
            "country": ["Mexico", "Mexico"],
            "market_year": [2021, 2022],
            "production": [4000.0, 4100.0],
            "arabica_production": [3600.0, 3700.0],
            "exports": [2000.0, 2050.0],
            "domestic_consumption": [2400.0, 2500.0],
        }
    )
    write_table(context, tmp_path / "coffee" / "clean" / "market_context", inputs={})
    return context


@pytest.fixture
def roaster_origins(tmp_path: Path) -> None:
    """The offer model's context: no coffee has to be listed for a request to be priced."""
    empty = clean_roasters(None, domains.coffee.adapter().config.cleaning)["roaster_origins"]
    write_table(empty, tmp_path / "coffee" / "clean" / "roaster_origins", inputs={})


@pytest.fixture
def model() -> RecordingModel:
    return RecordingModel()


@pytest.fixture
def client(
    coffee_adapter: CoffeeAdapter,
    tmp_path: Path,
    market_context: pl.DataFrame,
    roaster_origins: None,
    model: RecordingModel,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    def fake_load(name: str, tracking_uri: str, cache_dir: Path) -> ServedModel:
        return ServedModel(model, "7", "registry")

    monkeypatch.setattr(api, "load_champion", fake_load)
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path))
    with TestClient(api.create_app(coffee_adapter, Settings())) as test_client:
        yield test_client


def test_health_and_model_report_what_is_loaded(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "models": {"review": "7", "offer": "7"},
    }
    assert client.get("/models/review").json()["model_source"] == "registry"


def test_prediction_uses_the_same_features_as_the_batch_path(
    client: TestClient, coffee_config: DomainConfig, model: RecordingModel
) -> None:
    body = client.post(PREDICT, json=LOT).json()

    # The response names what was predicted instead of assuming it: the API is generic.
    assert (body["target"], body["prediction"]) == ("total_cup_points", 83.5)
    assert body["model_version"] == "7"
    assert model.seen is not None
    # Exactly the declared features, in order, and no leaking sensory column.
    assert list(model.seen.columns) == coffee_config.model_named("review").spec.features
    # Graded in 2023 -> market year 2022: the latest complete one, as in training.
    assert body["context"]["ctx_production"] == 4100.0
    assert body["context"]["ctx_arabica_share"] == pytest.approx(3700 / 4100)


def test_a_lot_without_market_context_is_still_scored(client: TestClient) -> None:
    body = client.post(PREDICT, json={**LOT, "country": "Narnia"}).json()

    assert body["prediction"] == 83.5
    assert body["context"]["ctx_production"] is None


def test_grading_date_defaults_to_today(client: TestClient, model: RecordingModel) -> None:
    client.post(PREDICT, json={k: v for k, v in LOT.items() if k != "graded_on"})

    assert model.seen is not None  # today has no context yet; the row is still scored
    assert model.seen["ctx_production"].isna().all()


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param({"altitude_m": -5}, id="negative-altitude"),
        pytest.param({"moisture_pct": 150}, id="impossible-moisture"),
        pytest.param({"country": None}, id="country-missing"),
        pytest.param({"graded_on": "not-a-date"}, id="bad-date"),
    ],
)
def test_invalid_payloads_are_rejected(client: TestClient, invalid: dict[str, Any]) -> None:
    assert client.post(PREDICT, json={**LOT, **invalid}).status_code == 422


def test_reload_picks_up_a_newly_promoted_champion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api, "load_champion", lambda *_: ServedModel(RecordingModel(), "8", "registry")
    )

    assert client.post("/reload").json()["loaded"]["review"]["model_version"] == "8"
    assert client.get("/health").json()["models"] == {"review": "8", "offer": "8"}


def test_without_a_model_the_service_says_so_instead_of_crashing(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: object) -> ServedModel:
        raise FileNotFoundError("nothing trained yet")

    monkeypatch.setattr(api, "load_champion", no_model)
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path))

    with TestClient(api.create_app(coffee_adapter, Settings())) as client:
        assert client.get("/health").json()["status"] == "no model"
        assert client.post(PREDICT, json=LOT).status_code == 503


def test_dates_are_not_silently_reinterpreted(client: TestClient, model: RecordingModel) -> None:
    client.post(PREDICT, json={**LOT, "graded_on": date(2022, 1, 15).isoformat()})

    assert model.seen is not None
    assert model.seen["ctx_production"].iloc[0] == 4000.0  # 2022 -> market year 2021


def test_a_model_without_its_context_is_not_reported_healthy(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The champion loads, the context table is missing: the service must say it cannot
    predict, not answer /health with "ok" and fail every request."""
    model = RecordingModel()
    monkeypatch.setattr(api, "load_champion", lambda *_: ServedModel(model, "7", "registry"))
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path))  # no clean tables at all

    with TestClient(api.create_app(coffee_adapter, Settings())) as client:
        assert client.get("/health").json()["status"] == "no model"
        assert client.post(PREDICT, json=LOT).status_code == 503


def test_a_model_the_domain_does_not_have_has_no_route(client: TestClient) -> None:
    assert client.post("/models/tasting/predict", json=LOT).status_code == 404


def test_a_reload_that_fails_says_which_model_and_why(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model already loaded keeps serving; the failure is reported, not swallowed."""

    def registry_down(*_: object) -> ServedModel:
        raise ConnectionError("registry down")

    monkeypatch.setattr(api, "load_champion", registry_down)

    body = client.post("/reload").json()

    assert body == {"loaded": {}, "failed": {"review": "registry down", "offer": "registry down"}}
    assert client.get("/health").json()["models"] == {"review": "7", "offer": "7"}


BAG = {
    "shop": "almanegra",
    "bag_grams": 312.5,
    "country": "Mexico",
    "state": "Oaxaca",
    "processing_method": "washed",
    "variety": "typica",
    "altitude_m": 1650,
}


def test_a_bag_is_priced_from_what_the_request_states(
    client: TestClient, model: RecordingModel, coffee_config: DomainConfig
) -> None:
    """A request describes a coffee no catalogue has to list: nothing is looked up."""
    response = client.post("/models/offer/predict", json=BAG)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "price_mxn_per_kg" and body["context"] == {}
    assert model.seen is not None
    assert list(model.seen.columns) == coffee_config.model_named("offer").spec.features
    assert model.seen["bag_grams"].iloc[0] == 312.5


def test_one_model_missing_leaves_the_other_serving(
    coffee_adapter: CoffeeAdapter,
    tmp_path: Path,
    market_context: pl.DataFrame,
    roaster_origins: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A price model the gate never promoted must not take the cup-score model down."""

    def only_review(name: str, *_: object) -> ServedModel:
        if name == "coffee-price-per-kg":
            raise FileNotFoundError("never promoted")
        return ServedModel(RecordingModel(), "7", "registry")

    monkeypatch.setattr(api, "load_champion", only_review)
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path))

    with TestClient(api.create_app(coffee_adapter, Settings())) as client:
        assert client.get("/health").json() == {
            "status": "partial",
            "models": {"review": "7", "offer": None},
        }
        assert client.post(PREDICT, json=LOT).status_code == 200
        assert client.post("/models/offer/predict", json=BAG).status_code == 503
