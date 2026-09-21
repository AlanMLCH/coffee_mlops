import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mlops_core.config import DomainConfig
from mlops_core.data.clean import build_clean
from mlops_core.ml import predict as batch
from mlops_core.ml.features import build_features
from mlops_core.ml.predict import PREDICTIONS, batch_predict, score
from mlops_core.ml.registry import ServedModel
from mlops_core.storage import MANIFEST_NAME, read_table

AT = datetime(2026, 9, 19, 12, tzinfo=UTC)


class CountingModel:
    """Returns a different value per row, so misaligned predictions are visible."""

    def predict(self, x: pd.DataFrame) -> list[float]:
        return [80.0 + i for i in range(len(x))]


@pytest.fixture
def data_dir(coffee_config: DomainConfig, raw_dir: Path) -> Path:
    build_clean(coffee_config, raw_dir.parent)
    build_features(coffee_config, raw_dir.parent)
    return raw_dir.parent


@pytest.fixture
def champion(monkeypatch: pytest.MonkeyPatch) -> ServedModel:
    served = ServedModel(CountingModel(), "7", "registry")
    monkeypatch.setattr(batch, "load_champion", lambda *args, **kwargs: served)
    return served


def test_every_row_is_scored_and_keeps_its_key(
    coffee_config: DomainConfig, data_dir: Path, champion: ServedModel
) -> None:
    features = read_table(data_dir / "features" / "review_features")

    scored = score(features, champion, coffee_config.model, AT)

    PREDICTIONS.validate(scored, lazy=True)
    assert scored["review_id"].to_list() == features["review_id"].to_list()
    assert scored["prediction"].to_list() == [80.0 + i for i in range(features.height)]
    assert scored["model_version"].unique().to_list() == ["7"]


def test_predictions_are_written_with_model_and_data_lineage(
    coffee_config: DomainConfig, data_dir: Path, champion: ServedModel
) -> None:
    path = batch_predict(coffee_config, data_dir, "sqlite:///unused", at=AT)

    assert read_table(data_dir / "predictions" / "review_predictions").height == 25
    inputs = json.loads((path.parent / MANIFEST_NAME).read_text())["inputs"]
    assert inputs["review_features"].startswith("built_at=")
    assert inputs["model"] == "coffee-total-cup-points v7"


def test_a_new_run_does_not_touch_the_previous_predictions(
    coffee_config: DomainConfig, data_dir: Path, champion: ServedModel
) -> None:
    first = batch_predict(coffee_config, data_dir, "sqlite:///unused", at=AT)

    second = batch_predict(
        coffee_config, data_dir, "sqlite:///unused", at=datetime(2026, 9, 20, 12, tzinfo=UTC)
    )

    assert first.is_file() and second.parent != first.parent
    # Same scores, different run: only the timestamp of the run differs.
    columns = ["review_id", "prediction", "model_version"]
    assert pl.read_parquet(first).select(columns).equals(pl.read_parquet(second).select(columns))
    assert pl.read_parquet(first)["predicted_at"][0] != pl.read_parquet(second)["predicted_at"][0]
