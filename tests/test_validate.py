import shutil
from collections.abc import Callable
from datetime import UTC, datetime
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


def test_every_configured_source_has_a_contract(coffee_adapter: CoffeeAdapter) -> None:
    """Including the API sources, which are configured apart from the file downloads, and
    the shelf survey's closed years, which share the year in course's."""
    config = coffee_adapter.config
    apis = (config.denue, config.overpass, config.fas, config.roasters)
    api = {source.name for source in apis if source}
    api |= {config.denue.workplaces.name} if config.denue and config.denue.workplaces else set()
    contracts = coffee_adapter.raw_contracts()

    assert config.sources.keys() | api == contracts.keys()
    assert RAW_SCHEMAS.keys() < contracts.keys()
    assert contracts["profeco_prices_2025"] is RAW_SCHEMAS["profeco_prices"]


def test_recorded_sources_pass_and_come_out_typed(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    validated = validate_raw(coffee_adapter, raw_dir)
    documents = {d.name for d in coffee_adapter.config.documents}
    frames = {name: s.frame for name, s in validated.items() if name not in documents}

    assert {name: df.height for name, df in frames.items()} == {
        "cqi_2018": 14,
        "cqi_2023": 12,
        "psd_coffee": 114,
        "cdmx_boroughs": 16,
        "denue_cafes": 3,
        "denue_workplaces": 336,  # 16 areas, 7 strata, 3 activity codes each
        "osm_places": 7,
        "fas_psd_coffee": 114,  # the same rows as psd_coffee, by the other road
        "siap_agricola": 57,  # 2025's 13 rows and two for each year from 2003
        "roaster_catalogs": 33,  # offers: a product in one size
        "world_bank_prices": 3,  # months, read from a workbook's sheet
        "ico_prices": 3,  # days, read from a PDF page by the domain
        "profeco_prices": 10,  # coffee, from three fortnights of everything PROFECO prices
        "profeco_prices_2025": 2,  # a closed year's archive, read the same way
        "profeco_prices_2024": 1,
        "fred_usd_mxn": 5,  # days, one of them without a rate
        "census_2020": 5,  # the state, three alcaldias and a small locality
    }
    # Latin-1 on disk, decoded on read: the accents come through as accents.
    assert "Café cereza" in frames["siap_agricola"]["Nomcultivo"].to_list()
    # Every year, each under the headers it was published with, read as one table.
    siap = frames["siap_agricola"]
    assert sorted(set(siap["Anio"].to_list())) == list(range(2003, 2026))
    assert siap.filter(pl.col("Anio") == 2016)["Nomcultivo"].to_list() == [
        "Café cereza", "Maíz grano"
    ]  # fmt: skip
    assert siap.filter(pl.col("Anio") == 2010)["Preciomediorural"].null_count() == 0
    assert frames["cdmx_boroughs"]["area_km2"].dtype == pl.Float64
    assert frames["denue_cafes"]["Latitud"].dtype == pl.Float64  # text upstream
    assert frames["osm_places"]["id"].dtype == pl.Int64
    assert frames["cqi_2018"]["Total.Cup.Points"].dtype == pl.Float64
    assert frames["cqi_2023"]["Quakers"].dtype == pl.Int64
    assert frames["psd_coffee"]["Market_Year"].dtype == pl.Int64
    # The workbook's unnamed first column is named by its position; units are not data.
    assert frames["world_bank_prices"]["column_1"].to_list() == ["2026M07", "2026M08", "2026M09"]
    assert frames["world_bank_prices"]["Coffee, Arabica"].dtype == pl.Float64
    # An accumulated source's rows carry the download they came from.
    assert frames["ico_prices"]["ingested_at"].dtype == pl.Datetime("us", "UTC")
    assert validated["ico_prices"].lineage.startswith("ingested_at=")


def test_r_style_na_is_read_as_null(coffee_config: DomainConfig, raw_dir: Path) -> None:
    df = read(coffee_config, raw_dir, "cqi_2018")

    assert "NA" not in df["altitude_mean_meters"].to_list()
    assert df["altitude_mean_meters"].null_count() > 0


def test_an_old_header_is_renamed_and_a_thousands_comma_taken_out_of_numbers_only(
    tmp_path: Path,
) -> None:
    from mlops_core.config import SourceConfig
    from mlops_core.data.extract import Manifest, RawArtifact

    partition = tmp_path / "ingested_at=20260929T000000Z"
    partition.mkdir()
    (partition / "t.csv").write_text(
        'place,Precio,Sembrada\n"Frontera, Corozal","3,350.00",10271\nOcosingo,12.5,"10,271"\n',
        encoding="utf-8",
    )
    manifest = Manifest(source="t", url="https://s.test/", filename="t.csv", sha256="x",
                        size_bytes=1, ingested_at=datetime(2026, 9, 29, tzinfo=UTC))  # fmt: skip
    source = SourceConfig(url="https://s.test/", filename="t.csv", thousands=",",
                          renamed={"Precio": "price", "absent": "x"})  # fmt: skip

    frame = read_raw(RawArtifact(partition, manifest), source)

    assert frame.columns == ["place", "price", "Sembrada"]
    assert frame["price"].to_list() == ["3350.00", "12.5"]
    assert frame["Sembrada"].to_list() == ["10271", "10271"]
    assert frame["place"].to_list() == ["Frontera, Corozal", "Ocosingo"]  # a name keeps it


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
    assert "osm_places" in validated


def test_a_file_in_another_encoding_must_say_so(coffee_config: DomainConfig, raw_dir: Path) -> None:
    """SIAP is Latin-1 and declares it nowhere: read as UTF-8 it fails, it does not guess."""
    artifact = latest_ingestion(raw_dir, "siap_agricola")
    assert artifact is not None
    as_utf8 = coffee_config.sources["siap_agricola"].model_copy(update={"encoding": "utf-8"})

    with pytest.raises(pl.exceptions.ComputeError):
        read_raw(artifact, as_utf8)


def test_the_documents_are_held_to_the_same_kind_of_contract(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    """A document is a source like any other: read into a frame, checked before use."""
    served = [d.name for d in coffee_adapter.config.documents if d.inbox is None]

    validated = validate_raw(coffee_adapter, raw_dir)

    assert set(served) <= validated.keys()
    parts = validated[served[0]].frame
    assert parts.columns == ["document_id", "part", "part_title", "text"]
    assert parts.height > 0
    # The ones a person hands over are absent here, and that is not a failure.
    assert not any(d.name in validated for d in coffee_adapter.config.documents if d.inbox)


def test_a_workbook_is_read_from_its_header_row_with_its_units_left_out(tmp_path: Path) -> None:
    from mlops_core.config import SourceConfig
    from mlops_core.data.validate import read_sheet
    from tests.files import xlsx

    path = tmp_path / "prices.xlsx"
    rows = [["A title"], [None, "Tea "], [None, "($/kg)"], ["2026M08", 3.1]]
    path.write_bytes(xlsx("Prices", rows))
    source = SourceConfig(
        url="https://bank.test/p.xlsx", filename="p.xlsx", sheet="Prices", header_row=1, skip_rows=1
    )

    frame = read_sheet(path, source)

    assert frame.columns == ["column_1", "Tea"]  # named by position where the sheet has none
    assert frame.rows() == [("2026M08", "3.1")]  # text: the contract does the typing


def test_a_reader_for_a_file_the_config_does_not_download_is_refused(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        CoffeeAdapter, "file_readers", lambda self: {"tea_prices": lambda path: pl.DataFrame()}
    )

    with pytest.raises(ValueError, match=r"Readers for files the config does not download"):
        validate_raw(coffee_adapter, raw_dir)


def test_an_api_source_named_in_accumulate_keeps_every_read(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    """The roasters' catalogue, read twice: both reads are checked and stacked."""
    first = latest_ingestion(raw_dir, "roaster_catalogs")
    assert first is not None
    again = raw_dir / "roaster_catalogs" / "ingested_at=20991231T000000000000Z"
    shutil.copytree(first.partition, again)
    later = first.manifest.model_copy(update={"ingested_at": datetime(2099, 12, 31, tzinfo=UTC)})
    (again / "manifest.json").write_text(later.model_dump_json(), encoding="utf-8")

    validated = validate_raw(coffee_adapter, raw_dir)["roaster_catalogs"]

    assert validated.reads == 2 and validated.frame["ingested_at"].n_unique() == 2
    assert validated.lineage == f"{again.name} and 1 earlier"


def test_accumulate_must_name_a_source_that_exists(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    config = coffee_adapter.config.model_copy(update={"accumulate": ["ico_prices", "tea_leaves"]})

    with pytest.raises(
        ValueError, match=r"`accumulate` names sources that do not exist: \['tea_leaves'\]"
    ):
        validate_raw(CoffeeAdapter(config), raw_dir)


def test_one_read_is_checked_on_its_own(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> None:
    """A file read and an API read alike, each against its source's contract."""
    from mlops_core.data.extract import latest_ingestion
    from mlops_core.data.validate import validate_read

    for name, rows in (("ico_prices", 3), ("roaster_catalogs", 33)):
        artifact = latest_ingestion(raw_dir, name)
        assert artifact is not None

        checked = validate_read(coffee_adapter, name, artifact)

        assert checked.frame.height == rows
        assert checked.lineage == artifact.partition.name
