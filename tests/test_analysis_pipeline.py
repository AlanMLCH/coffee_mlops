import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeConfig
from mlops_core.analysis import pipeline
from mlops_core.analysis.pipeline import build_analysis, champion_importance
from mlops_core.data.clean import build_clean
from mlops_core.ml.features import build_features
from mlops_core.ml.predict import batch_predict
from mlops_core.ml.registry import ServedModel
from mlops_core.storage import MANIFEST_NAME, read_table
from tests.fakes import ConstantModel

AT = datetime(2026, 9, 20, 12, tzinfo=UTC)
# The recorded PSD excerpt stops at market year 2023 and SIAP's file is 2025, so there
# is legitimately nothing to set side by side: the study must come out empty, not fail.
EMPTY_ON_THE_FIXTURES = {"production_crosscheck"}
ALWAYS_WRITTEN = {
    "target_distribution",
    "numeric_profile",
    "categorical_profile",
    "feature_recommendation",
    "market_summary",
    "market_history",
}


def with_market_year(config: CoffeeConfig, year: int) -> CoffeeAdapter:
    market = config.market_analysis.model_copy(update={"market_year": year})
    analysis = config.analysis.model_copy(update={"min_rows": 1})
    return CoffeeAdapter(
        config.model_copy(update={"analysis": analysis, "market_analysis": market})
    )


@pytest.fixture
def analysis_adapter(coffee_config: CoffeeConfig) -> CoffeeAdapter:
    """The recorded PSD excerpt covers 2022-2023, so the market year points there."""
    return with_market_year(coffee_config, 2022)


@pytest.fixture
def data_dir(analysis_adapter: CoffeeAdapter, raw_dir: Path) -> Path:
    build_clean(analysis_adapter, raw_dir.parent)
    build_features(analysis_adapter, raw_dir.parent)
    return raw_dir.parent


@pytest.fixture
def champion(monkeypatch: pytest.MonkeyPatch) -> ServedModel:
    served = ServedModel(ConstantModel(), "7", "registry")
    monkeypatch.setattr(pipeline, "load_champion", lambda *args, **kwargs: served)
    return served


def test_every_study_is_written_as_parquet_and_csv(
    analysis_adapter: CoffeeAdapter, data_dir: Path, champion: ServedModel
) -> None:
    output = build_analysis(analysis_adapter, data_dir, "sqlite:///unused", at=AT)

    assert set(output.tables) >= ALWAYS_WRITTEN
    for name, path in output.tables.items():
        assert path.is_file()
        # The CSV is the copy a person opens; it must sit in the same partition.
        assert (path.parent / f"{name}.csv").is_file()
        if name not in EMPTY_ON_THE_FIXTURES:
            assert read_table(data_dir / "analysis" / name).height > 0


def test_each_study_records_which_partitions_it_read(
    analysis_adapter: CoffeeAdapter, data_dir: Path, champion: ServedModel
) -> None:
    output = build_analysis(analysis_adapter, data_dir, "sqlite:///unused", at=AT)

    manifest = output.tables["numeric_profile"].parent / MANIFEST_NAME
    lineage = json.loads(manifest.read_text())["inputs"]
    assert {"coffee_reviews", "market_context", "review_features"} <= set(lineage)
    assert all(partition.startswith("built_at=") for partition in lineage.values())


def test_residuals_appear_once_there_are_predictions(
    analysis_adapter: CoffeeAdapter,
    data_dir: Path,
    champion: ServedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = build_analysis(analysis_adapter, data_dir, "sqlite:///unused", at=AT)
    assert "residuals" not in before.tables

    monkeypatch.setattr("mlops_core.ml.predict.load_champion", lambda *a, **k: champion)
    batch_predict(analysis_adapter.config, data_dir, "sqlite:///unused")
    after = build_analysis(
        analysis_adapter, data_dir, "sqlite:///unused", at=datetime(2026, 9, 21, 12, tzinfo=UTC)
    )

    assert "residuals" in after.tables
    assert "residual_bias" in after.figures
    assert read_table(data_dir / "analysis" / "residuals").height > 0


def test_importance_is_measured_on_the_test_split(
    analysis_adapter: CoffeeAdapter, data_dir: Path, champion: ServedModel
) -> None:
    importance = champion_importance(analysis_adapter.config, data_dir, "sqlite:///unused")

    assert importance is not None
    assert importance["feature"].to_list() == analysis_adapter.config.model.features
    # A model that ignores its input cannot lose accuracy when a feature is shuffled.
    assert importance["permutation_importance"].to_list() == [0.0] * len(
        analysis_adapter.config.model.features
    )


def test_without_a_champion_the_studies_still_run(
    analysis_adapter: CoffeeAdapter, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_champion(*args: object, **kwargs: object) -> ServedModel:
        raise FileNotFoundError("nothing trained yet")

    monkeypatch.setattr(pipeline, "load_champion", no_champion)

    output = build_analysis(analysis_adapter, data_dir, "sqlite:///unused", at=AT)

    recommendations = read_table(data_dir / "analysis" / "feature_recommendation")
    assert set(output.tables) >= ALWAYS_WRITTEN
    assert "feature_importance" not in output.figures  # nothing measured, nothing drawn
    assert recommendations["permutation_importance"].null_count() == recommendations.height


def test_figures_are_drawn_and_the_selection_is_published(
    analysis_adapter: CoffeeAdapter, data_dir: Path, champion: ServedModel, tmp_path: Path
) -> None:
    docs = tmp_path / "published"

    output = build_analysis(analysis_adapter, data_dir, "sqlite:///unused", at=AT, publish_to=docs)

    assert {"target_distribution", "market_history", "feature_importance"} <= set(output.figures)
    for path in output.figures.values():
        assert path.suffix == ".png" and path.stat().st_size > 0
    # Only the configured selection is copied where the docs can reference it.
    assert {path.name for path in output.published} <= {
        f"{name}.png" for name in analysis_adapter.config.analysis.published_figures
    }
    assert all(path.parent == docs for path in output.published)


def test_nothing_is_published_outside_a_checkout(
    analysis_adapter: CoffeeAdapter, data_dir: Path, champion: ServedModel
) -> None:
    output = build_analysis(analysis_adapter, data_dir, "sqlite:///unused", at=AT)

    assert output.published == []


def test_a_study_that_comes_out_empty_says_so(
    analysis_adapter: CoffeeAdapter,
    data_dir: Path,
    champion: ServedModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence would let someone read an empty table as "nothing to report"."""
    unpublished_year = with_market_year(analysis_adapter.config, 2099)

    build_analysis(unpublished_year, data_dir, "sqlite:///unused", at=AT)

    assert "market_summary came out empty" in caplog.text


def test_an_empty_study_does_not_take_the_figures_down_with_it(
    analysis_adapter: CoffeeAdapter, data_dir: Path, champion: ServedModel
) -> None:
    """The market year is a config value; asking for one the data lacks must degrade,
    not crash, or a stale config breaks the whole analysis run."""
    unpublished_year = with_market_year(analysis_adapter.config, 2099)

    output = build_analysis(unpublished_year, data_dir, "sqlite:///unused", at=AT)

    assert "market_share" not in output.figures
    assert {"target_distribution", "numeric_signal"} <= set(output.figures)
