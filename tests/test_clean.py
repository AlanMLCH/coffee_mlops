import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from coffee_mlops.clean import (
    altitude_from_text,
    build_clean,
    clean_market_context,
    clean_reviews,
    parse_grading_date,
)
from coffee_mlops.config import DomainConfig
from coffee_mlops.schemas import MARKET_CONTEXT, PSD_ATTRIBUTES, coffee_reviews_schema
from coffee_mlops.storage import MANIFEST_NAME, read_table
from coffee_mlops.validate import validate_raw

Frames = dict[str, pl.DataFrame]


@pytest.fixture
def frames(coffee_config: DomainConfig, raw_dir: Path) -> Frames:
    return {name: source.frame for name, source in validate_raw(coffee_config, raw_dir).items()}


def set_first(df: pl.DataFrame, column: str, value: object) -> pl.DataFrame:
    """Overwrite `column` in the first row only."""
    first = pl.int_range(pl.len()) == 0
    return df.with_columns(
        pl.when(first).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )


@pytest.mark.parametrize(
    ("text", "meters"),
    [
        ("1200", 1200.0),
        ("1700-1930", 1815.0),
        ("1200 - 1300", 1250.0),
        ("1200~1600", 1400.0),
        ("4895 A 5650", 5272.5),  # parsed; the plausible-range rule nulls it later
        (None, None),
    ],
)
def test_altitude_from_text(text: str | None, meters: float | None) -> None:
    df = pl.DataFrame({"a": [text]}, schema={"a": pl.String})

    assert df.select(altitude_from_text(pl.col("a")))["a"].item() == meters


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("April 4th, 2015", date(2015, 4, 4)),
        ("September 21st, 2022", date(2022, 9, 21)),
        ("May 2nd, 2023", date(2023, 5, 2)),
        ("March 3rd, 2011", date(2011, 3, 3)),
        ("November 15th, 2017\n", date(2017, 11, 15)),  # stray newline in the 2018 scrape
    ],
)
def test_parse_grading_date(text: str, expected: date) -> None:
    df = pl.DataFrame({"d": [text]})

    assert df.select(parse_grading_date(pl.col("d")))["d"].item() == expected


def test_both_snapshots_become_one_table_that_meets_the_contract(
    coffee_config: DomainConfig, frames: Frames
) -> None:
    reviews = clean_reviews(frames, coffee_config.cleaning)

    coffee_reviews_schema(coffee_config.cleaning).validate(reviews, lazy=True)
    assert reviews.group_by("snapshot").len().sort("snapshot").rows() == [
        ("cqi_2018", 13),
        ("cqi_2023", 12),
    ]


def test_never_cupped_lot_is_dropped_but_low_real_scores_stay(
    coffee_config: DomainConfig, frames: Frames
) -> None:
    # A real 2018 cupping scored 59.83: Clean Cup and Sweetness are per cup and can be 1.33.
    frames["cqi_2023"] = set_first(frames["cqi_2023"], "Total Cup Points", 59.83)

    reviews = clean_reviews(frames, coffee_config.cleaning)

    assert "cqi_2018-1312" not in reviews["review_id"].to_list()  # every score is 0
    assert 59.83 in reviews["total_cup_points"].to_list()


def test_moisture_is_a_percentage_in_both_snapshots(
    coffee_config: DomainConfig, frames: Frames
) -> None:
    moisture = clean_reviews(frames, coffee_config.cleaning)["moisture_pct"].drop_nulls()

    assert moisture.min() > 1


def test_physically_impossible_values_become_null(
    coffee_config: DomainConfig, frames: Frames
) -> None:
    frames["cqi_2023"] = set_first(
        set_first(frames["cqi_2023"], "Altitude", "4895 A 5650"), "Moisture Percentage", 0.0
    )

    first = clean_reviews(frames, coffee_config.cleaning).filter(
        pl.col("review_id") == "cqi_2023-0"
    )

    assert first["altitude_m"].item() is None
    assert first["moisture_pct"].item() is None


def test_labels_are_mapped_to_closed_vocabularies(
    coffee_config: DomainConfig, frames: Frames
) -> None:
    frames["cqi_2023"] = set_first(
        set_first(frames["cqi_2023"], "Country of Origin", "United States (Hawaii)"),
        "Processing Method",
        "SEMI-LAVADO",
    )

    first = clean_reviews(frames, coffee_config.cleaning).filter(
        pl.col("review_id") == "cqi_2023-0"
    )

    assert first["country"].item() == "United States"
    assert first["processing_method"].item() == "semi_washed"


def test_blank_text_becomes_null(coffee_config: DomainConfig, frames: Frames) -> None:
    frames["cqi_2023"] = set_first(frames["cqi_2023"], "Variety", "   ")

    first = clean_reviews(frames, coffee_config.cleaning).filter(
        pl.col("review_id") == "cqi_2023-0"
    )

    assert first["variety"].item() is None


def test_unseen_label_stops_the_pipeline(coffee_config: DomainConfig, frames: Frames) -> None:
    frames["cqi_2023"] = set_first(frames["cqi_2023"], "Processing Method", "Koji Fermented")

    with pytest.raises(ValueError, match=r"koji fermented.*processing_methods"):
        clean_reviews(frames, coffee_config.cleaning)


def test_market_context_is_one_row_per_country_and_year(frames: Frames) -> None:
    context = clean_market_context(frames["psd_coffee"])

    MARKET_CONTEXT.validate(context, lazy=True)
    assert context.select("country", "market_year").rows() == [
        (country, year) for country in ("Brazil", "Colombia", "Mexico") for year in (2022, 2023)
    ]
    assert context.columns[2:] == list(PSD_ATTRIBUTES.values())


def test_attribute_missing_from_download_still_gets_a_null_column(frames: Frames) -> None:
    psd = frames["psd_coffee"].filter(pl.col("Attribute_Description") != "Soluble Exports")

    context = clean_market_context(psd)

    assert context["soluble_exports"].null_count() == context.height


def test_build_clean_writes_both_tables_with_lineage(
    coffee_config: DomainConfig, raw_dir: Path
) -> None:
    data_dir = raw_dir.parent
    paths = build_clean(coffee_config, data_dir, at=datetime(2026, 9, 19, tzinfo=UTC))

    assert set(paths) == {"coffee_reviews", "market_context"}
    assert read_table(data_dir / "clean" / "coffee_reviews").height == 25
    manifest = json.loads((paths["coffee_reviews"].parent / MANIFEST_NAME).read_text())
    assert set(manifest["inputs"]) == {"cqi_2018", "cqi_2023"}
    assert all(p.startswith("ingested_at=") for p in manifest["inputs"].values())
