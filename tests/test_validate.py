import shutil
from collections.abc import Callable
from pathlib import Path

import pandera.errors
import polars as pl
import pytest

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.schemas import RAW_SCHEMAS
from mlops_core.config import DomainConfig
from mlops_core.data.extract import latest_ingestion
from mlops_core.data.validate import check_contract, read_raw, validate_raw


def read(coffee_config: DomainConfig, raw_dir: Path, source: str) -> pl.DataFrame:
    artifact = latest_ingestion(raw_dir, source)
    assert artifact is not None
    return read_raw(artifact, coffee_config.sources[source])


def test_every_configured_source_has_a_contract(coffee_config: DomainConfig) -> None:
    """Including the API sources, which are configured apart from the file downloads."""
    apis = (coffee_config.denue, coffee_config.overpass, coffee_config.fas)
    api = {source.name for source in apis if source}

    assert coffee_config.sources.keys() | api == RAW_SCHEMAS.keys()


def test_recorded_sources_pass_and_come_out_typed(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    frames = {name: s.frame for name, s in validate_raw(coffee_adapter, raw_dir).items()}

    assert {name: df.height for name, df in frames.items()} == {
        "cqi_2018": 14,
        "cqi_2023": 12,
        "psd_coffee": 114,
        "cdmx_boroughs": 16,
        "denue_cafes": 3,
        "osm_cafes": 5,
        "fas_psd_coffee": 114,  # the same rows as psd_coffee, by the other road
    }
    assert frames["cdmx_boroughs"]["area_km2"].dtype == pl.Float64
    assert frames["denue_cafes"]["Latitud"].dtype == pl.Float64  # text upstream
    assert frames["osm_cafes"]["id"].dtype == pl.Int64
    assert frames["cqi_2018"]["Total.Cup.Points"].dtype == pl.Float64
    assert frames["cqi_2023"]["Quakers"].dtype == pl.Int64
    assert frames["psd_coffee"]["Market_Year"].dtype == pl.Int64


def test_r_style_na_is_read_as_null(coffee_config: DomainConfig, raw_dir: Path) -> None:
    df = read(coffee_config, raw_dir, "cqi_2018")

    assert "NA" not in df["altitude_mean_meters"].to_list()
    assert df["altitude_mean_meters"].null_count() > 0


def test_missing_ingestion_fails_with_a_hint(coffee_adapter: CoffeeAdapter, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run extract first"):
        validate_raw(coffee_adapter, tmp_path)


Mutation = Callable[[pl.DataFrame], pl.DataFrame]


@pytest.mark.parametrize(
    ("source", "mutation", "offending_column"),
    [
        pytest.param(
            "cqi_2018",
            lambda df: df.with_columns(pl.lit("11.5").alias("Moisture")),
            "Moisture",
            id="moisture-switched-to-percentage",
        ),
        pytest.param(
            "cqi_2018",
            lambda df: df.drop("Total.Cup.Points"),
            "Total.Cup.Points",
            id="target-column-removed",
        ),
        pytest.param(
            "cqi_2023",
            lambda df: df.with_columns(pl.lit("excellent").alias("Aroma")),
            "Aroma",
            id="score-not-numeric",
        ),
        pytest.param(
            "psd_coffee",
            lambda df: df.with_columns(pl.lit("(METRIC TONS)").alias("Unit_Description")),
            "Unit_Description",
            id="unit-changed",
        ),
        pytest.param(
            "psd_coffee",
            lambda df: df.with_columns(pl.lit("Futures Volume").alias("Attribute_Description")),
            "Attribute_Description",
            id="unknown-attribute",
        ),
        pytest.param(
            "psd_coffee",
            lambda df: pl.concat([df, df.head(1)]),
            "Country_Code",
            id="duplicate-country-year-attribute",
        ),
    ],
)
def test_contract_violations_stop_the_pipeline(
    coffee_config: DomainConfig,
    raw_dir: Path,
    source: str,
    mutation: Mutation,
    offending_column: str,
) -> None:
    broken = mutation(read(coffee_config, raw_dir, source))

    with pytest.raises(pandera.errors.SchemaErrors) as exc:
        check_contract(RAW_SCHEMAS[source], broken)

    assert offending_column in str(exc.value)


def test_all_violations_are_reported_at_once(coffee_config: DomainConfig, raw_dir: Path) -> None:
    broken = read(coffee_config, raw_dir, "cqi_2023").with_columns(
        pl.lit("excellent").alias("Aroma"), pl.lit("-1").alias("Quakers")
    )

    with pytest.raises(pandera.errors.SchemaErrors) as exc:
        check_contract(RAW_SCHEMAS["cqi_2023"], broken)

    assert {"Aroma", "Quakers"} <= set(exc.value.failure_cases["column"].to_list())


def test_an_api_source_that_was_never_ingested_is_skipped_not_raised(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    """A clone with no DENUE token still validates everything else."""
    shutil.rmtree(raw_dir / "denue_cafes")

    validated = validate_raw(coffee_adapter, raw_dir)

    assert "denue_cafes" not in validated
    assert "osm_cafes" in validated
