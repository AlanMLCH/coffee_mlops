from collections.abc import Callable
from pathlib import Path

import httpx
import pandera.errors
import polars as pl
import pytest

from coffee_mlops.config import DomainConfig
from coffee_mlops.extract import extract_all, latest_ingestion
from coffee_mlops.schemas import RAW_SCHEMAS
from coffee_mlops.validate import check_contract, read_raw, validate_raw


@pytest.fixture
def raw_dir(tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client) -> Path:
    extract_all(coffee_config, tmp_path, client)
    return tmp_path


def read(coffee_config: DomainConfig, raw_dir: Path, source: str) -> pl.DataFrame:
    artifact = latest_ingestion(raw_dir, source)
    assert artifact is not None
    return read_raw(artifact, coffee_config.sources[source])


def test_every_configured_source_has_a_contract(coffee_config: DomainConfig) -> None:
    assert coffee_config.sources.keys() == RAW_SCHEMAS.keys()


def test_recorded_sources_pass_and_come_out_typed(
    coffee_config: DomainConfig, raw_dir: Path
) -> None:
    frames = validate_raw(coffee_config, raw_dir)

    assert {name: df.height for name, df in frames.items()} == {
        "cqi_2018": 14,
        "cqi_2023": 12,
        "psd_coffee": 114,
    }
    assert frames["cqi_2018"]["Total.Cup.Points"].dtype == pl.Float64
    assert frames["cqi_2023"]["Quakers"].dtype == pl.Int64
    assert frames["psd_coffee"]["Market_Year"].dtype == pl.Int64


def test_r_style_na_is_read_as_null(coffee_config: DomainConfig, raw_dir: Path) -> None:
    df = read(coffee_config, raw_dir, "cqi_2018")

    assert "NA" not in df["altitude_mean_meters"].to_list()
    assert df["altitude_mean_meters"].null_count() > 0


def test_missing_ingestion_fails_with_a_hint(coffee_config: DomainConfig, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run extract first"):
        validate_raw(coffee_config, tmp_path)


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
