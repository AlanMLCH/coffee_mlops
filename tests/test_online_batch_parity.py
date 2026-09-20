"""Online and batch inference must build identical features for the same lot.

This is the failure that silently ruins a served model: the API computing a feature
slightly differently from the training and batch path.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

from coffee_mlops.config import DomainConfig, Settings
from coffee_mlops.data.clean import build_clean
from coffee_mlops.ml.features import build_features
from coffee_mlops.ml.registry import ServedModel
from coffee_mlops.serving import api
from coffee_mlops.storage import read_table

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
def data_dir(coffee_config: DomainConfig, raw_dir: Path) -> Path:
    build_clean(coffee_config, raw_dir.parent)
    build_features(coffee_config, raw_dir.parent)
    return raw_dir.parent


@pytest.fixture
def online(
    coffee_config: DomainConfig, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, RecordingModel]]:
    model = RecordingModel()
    monkeypatch.setattr(api, "load_champion", lambda *_: ServedModel(model, "7", "registry"))
    monkeypatch.setenv("COFFEE_DATA_DIR", str(data_dir.parent))
    with TestClient(api.create_app(coffee_config, Settings())) as client:
        yield client, model


def test_the_api_reproduces_the_batch_feature_row(
    coffee_config: DomainConfig, data_dir: Path, online: tuple[TestClient, RecordingModel]
) -> None:
    client, model = online
    reviews = read_table(data_dir / "clean" / "coffee_reviews")
    features = read_table(data_dir / "features" / "review_features")
    # A lot with market context: same country and year as the recorded PSD fixture.
    review = reviews.filter(pl.col("country") == "Mexico").row(0, named=True)

    payload = {field: review[field] for field in LOT_FIELDS} | {
        "graded_on": review["grading_date"].isoformat()
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text
    assert model.seen is not None
    expected = (
        features.filter(pl.col("review_id") == review["review_id"])
        .select(coffee_config.model.features)
        .to_pandas()
    )
    pd.testing.assert_frame_equal(
        model.seen.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
