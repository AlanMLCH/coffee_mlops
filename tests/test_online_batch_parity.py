"""Online and batch inference must build identical features for the same lot.

This is the failure that silently ruins a served model: the API computing a feature
slightly differently from the training and batch path.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.clean import clean_market_context, clean_reviews
from domains.coffee.roaster_sheets import clean_roasters
from domains.coffee.schemas import RAW_SCHEMAS
from domains.coffee.sources.roasters import to_frame
from mlops_core.config import Settings
from mlops_core.contracts import check_contract
from mlops_core.data.extract import latest_ingestion
from mlops_core.data.validate import read_raw
from mlops_core.ml.features import build_features
from mlops_core.ml.registry import ServedModel
from mlops_core.serving import api
from mlops_core.storage import read_table, write_table

LOT_FIELDS = [
    "country",
    "variety",
    "processing_method",
    "color",
    "altitude_m",
    "moisture_pct",
    "category_one_defects",
    "category_two_defects",
    "quakers",
]


class RecordingModel:
    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict(self, x: pd.DataFrame) -> list[float]:
        self.seen = x
        return [83.5] * len(x)


@pytest.fixture
def data_dir(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> Path:
    """Only the tables the model reads: items and market context, cleaned by the domain's
    own functions. Parity is a question about features, not about the map - and this
    test also runs where the API is deployed, which has no spatial engine installed."""
    config = coffee_adapter.config
    frames = {}
    for name in ("cqi_2018", "cqi_2023", "psd_coffee"):
        artifact = latest_ingestion(raw_dir, name)
        assert artifact is not None
        frames[name] = check_contract(RAW_SCHEMAS[name], read_raw(artifact, config.sources[name]))
    clean_dir = raw_dir.parent / "clean"
    write_table(clean_reviews(frames, config.cleaning), clean_dir / "coffee_reviews", {})
    write_table(clean_market_context(frames["psd_coffee"]), clean_dir / "market_context", {})
    build_features(coffee_adapter, "review", raw_dir.parent)
    return raw_dir.parent


@pytest.fixture
def online(
    coffee_adapter: CoffeeAdapter, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, RecordingModel]]:
    model = RecordingModel()
    monkeypatch.setattr(api, "load_champion", lambda *_: ServedModel(model, "7", "registry"))
    monkeypatch.setenv("MLOPS_DATA_DIR", str(data_dir.parent))
    with TestClient(api.create_app(coffee_adapter, Settings())) as client:
        yield client, model


def test_the_api_reproduces_the_batch_feature_row(
    coffee_adapter: CoffeeAdapter, data_dir: Path, online: tuple[TestClient, RecordingModel]
) -> None:
    client, model = online
    reviews = read_table(data_dir / "clean" / "coffee_reviews")
    features = read_table(data_dir / "features" / "review_features")
    # A lot with market context: same country and year as the recorded PSD fixture.
    review = reviews.filter(pl.col("country") == "Mexico").row(0, named=True)

    payload = {field: review[field] for field in LOT_FIELDS} | {
        "graded_on": review["grading_date"].isoformat()
    }
    response = client.post("/models/review/predict", json=payload)

    assert response.status_code == 200, response.text
    assert model.seen is not None
    expected = (
        features.filter(pl.col("review_id") == review["review_id"])
        .select(coffee_adapter.config.model_named("review").spec.features)
        .to_pandas()
    )
    # dtypes included: the API builds its frame differently from the batch path, and a
    # silent dtype difference is exactly how online and batch drift apart.
    pd.testing.assert_frame_equal(
        model.seen.reset_index(drop=True), expected.reset_index(drop=True)
    )


OFFER_FIELDS = ["shop", "bag_grams", "country", "state", "processing_method", "variety"]


@pytest.fixture
def offers_dir(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> Path:
    """The offer model's tables, from the recorded shops, cleaned by the domain."""
    artifact = latest_ingestion(raw_dir, "roaster_catalogs")
    assert artifact is not None
    document = json.loads(artifact.path.read_text(encoding="utf-8"))
    raw = check_contract(RAW_SCHEMAS["roaster_catalogs"], to_frame(document))
    tables = clean_roasters(raw, coffee_adapter.config.cleaning, artifact.manifest.ingested_at)
    clean_dir = raw_dir.parent / "clean"
    for name in ("roaster_offers", "roaster_origins"):
        write_table(tables[name], clean_dir / name, {})
    build_features(coffee_adapter, "offer", raw_dir.parent)
    return raw_dir.parent


def test_the_api_prices_a_bag_as_the_batch_path_does(
    coffee_adapter: CoffeeAdapter, offers_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request stating what the catalogue says about a coffee reaches the model as the
    same row the batch path built from the catalogue."""
    model = RecordingModel()
    monkeypatch.setattr(api, "load_champion", lambda *_: ServedModel(model, "7", "registry"))
    monkeypatch.setenv("MLOPS_DATA_DIR", str(offers_dir.parent))
    spec = coffee_adapter.config.model_named("offer").spec
    features = read_table(offers_dir / "features" / "offer_features")
    offer = features.filter(pl.col("altitude_m").is_not_null()).row(0, named=True)
    payload = {field: offer[field] for field in [*OFFER_FIELDS, "altitude_m"]} | {
        "observed_on": offer["observed_on"].isoformat()
    }

    with TestClient(api.create_app(coffee_adapter, Settings())) as client:
        response = client.post("/models/offer/predict", json=payload)

    assert response.status_code == 200, response.text
    assert model.seen is not None
    expected = features.filter(pl.col("offer_id") == offer["offer_id"]).select(spec.features)
    pd.testing.assert_frame_equal(
        model.seen.reset_index(drop=True), expected.to_pandas().reset_index(drop=True)
    )
