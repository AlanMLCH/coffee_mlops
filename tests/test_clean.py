import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

import domains.coffee
from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.clean import (
    altitude_from_text,
    clean_boroughs,
    clean_coffee_shops,
    clean_market_context,
    clean_mexico_production,
    clean_reviews,
    parse_grading_date,
    reconcile_market_sources,
    shop_kind,
)
from domains.coffee.schemas import (
    BOROUGHS,
    MARKET_CONTEXT,
    PSD_ATTRIBUTES,
    coffee_reviews_schema,
    coffee_shops_schema,
)
from mlops_core.config import DomainConfig
from mlops_core.contracts import check_contract
from mlops_core.data.clean import build_clean
from mlops_core.data.validate import validate_raw
from mlops_core.storage import MANIFEST_NAME, read_table

Frames = dict[str, pl.DataFrame]
RULES = domains.coffee.adapter().config.cleaning


@pytest.fixture
def frames(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> Frames:
    return {name: source.frame for name, source in validate_raw(coffee_adapter, raw_dir).items()}


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


def test_variety_is_case_normalized(coffee_config: DomainConfig, frames: Frames) -> None:
    frames["cqi_2023"] = set_first(frames["cqi_2023"], "Variety", "Gesha")

    first = clean_reviews(frames, coffee_config.cleaning).filter(
        pl.col("review_id") == "cqi_2023-0"
    )

    assert first["variety"].item() == "gesha"


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
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    data_dir = raw_dir.parent
    paths = build_clean(coffee_adapter, data_dir, at=datetime(2026, 9, 19, tzinfo=UTC))

    assert set(paths) == {
        "coffee_reviews",
        "market_context",
        "boroughs",
        "coffee_shops",
        "mexico_production",
    }
    assert read_table(data_dir / "clean" / "coffee_reviews").height == 25
    manifest = json.loads((paths["coffee_reviews"].parent / MANIFEST_NAME).read_text())
    assert set(manifest["inputs"]) == {"cqi_2018", "cqi_2023"}
    assert all(p.startswith("ingested_at=") for p in manifest["inputs"].values())


def shops(frames: Frames) -> pl.DataFrame:
    return clean_coffee_shops(frames, frames["cdmx_boroughs"], RULES)


def test_both_registers_become_one_table_of_places(frames: Frames) -> None:
    table = check_contract(coffee_shops_schema(RULES), shops(frames))

    assert dict(table["source"].value_counts().iter_rows()) == {"denue": 3, "osm": 7}
    assert table["shop_id"].to_list()[:1] == ["denue-1"]
    # Two registers, two vocabularies: each keeps what only it records.
    assert table.filter(pl.col("source") == "denue")["employees_band"].null_count() == 0
    assert table.filter(pl.col("source") == "osm")["employees_band"].null_count() == 7


def test_every_place_is_put_in_a_borough(frames: Frames) -> None:
    table = shops(frames)

    assert table["borough_id"].null_count() == 0
    assert set(table.filter(pl.col("source") == "denue")["borough"]) == {
        "Miguel Hidalgo",
        "Tláhuac",
    }


def test_the_join_is_audited_against_the_borough_the_source_declares(
    frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    """The check that proves the projection and the axis order: DENUE states its own
    borough, so the join can be scored instead of trusted."""
    caplog.set_level(logging.INFO)

    shops(frames)

    assert "agrees with the source's own borough on 3 of 3 places (100.00%)" in caplog.text


def test_a_disagreement_is_reported_rather_than_absorbed(
    frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    moved = dict(frames)
    moved["denue_cafes"] = frames["denue_cafes"].with_columns(pl.lit("090150001").alias("AreaGeo"))

    clean_coffee_shops(moved, frames["cdmx_boroughs"], RULES)

    assert "agrees with the source's own borough on 0 of 3" in caplog.text


def test_a_place_outside_every_borough_is_kept_and_counted(
    frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    adrift = dict(frames)
    adrift["denue_cafes"] = set_first(frames["denue_cafes"], "Latitud", 0.0)

    table = clean_coffee_shops(adrift, frames["cdmx_boroughs"], RULES)

    assert "1 places fell outside every borough" in caplog.text
    assert table["borough_id"].null_count() == 1  # the row stays, unplaced


def test_an_element_without_a_coordinate_is_dropped(
    frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    """OSM is crowd-sourced: a cafe can be tagged without ever being placed."""
    caplog.set_level(logging.WARNING)
    unplaced = dict(frames)
    unplaced["osm_places"] = set_first(frames["osm_places"], "latitude", None)

    table = clean_coffee_shops(unplaced, frames["cdmx_boroughs"], RULES)

    assert "Dropped 1 places with no coordinate" in caplog.text
    assert table.height == 9  # 3 from DENUE, 7 from OSM, less the one with no point


def test_one_register_is_enough_to_build_the_table(frames: Frames) -> None:
    """A clone with no DENUE token still gets the OpenStreetMap half."""
    table = clean_coffee_shops({"osm_places": frames["osm_places"]}, frames["cdmx_boroughs"], RULES)

    assert table["source"].unique().to_list() == ["osm"]
    assert table["declared_borough_id"].null_count() == table.height


def test_no_register_at_all_says_what_to_run(frames: Frames) -> None:
    with pytest.raises(ValueError, match="run extract first"):
        clean_coffee_shops({}, frames["cdmx_boroughs"], RULES)


def test_boroughs_keep_their_polygon_and_their_official_key(frames: Frames) -> None:
    table = check_contract(BOROUGHS, clean_boroughs(frames["cdmx_boroughs"]))

    assert table.height == 16
    assert "Cuauhtémoc" in table["borough"].to_list()
    # The geometry travels as WKB, so reading the table needs no spatial extension.
    assert table["boundary"].dtype == pl.Binary


def test_the_api_and_the_file_agree_on_every_row(frames: Frames) -> None:
    result = reconcile_market_sources(frames["psd_coffee"], frames["fas_psd_coffee"])

    assert result.agree
    assert result.rows == 114


def test_a_revised_value_is_reported_not_absorbed(
    frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    """The day a circular lands in one road before the other, the build says so."""
    caplog.set_level(logging.WARNING)
    revised = set_first(frames["fas_psd_coffee"], "Value", 1.0)

    result = reconcile_market_sources(frames["psd_coffee"], revised)

    assert (result.only_file, result.only_api, result.different) == (0, 0, 1)
    assert "1 with different values" in caplog.text


def test_a_row_on_only_one_side_is_counted_on_that_side(frames: Frames) -> None:
    file, api = frames["psd_coffee"], frames["fas_psd_coffee"]

    assert reconcile_market_sources(file, api.slice(1)).only_file == 1
    assert reconcile_market_sources(file.slice(1), api).only_api == 1


def test_market_context_is_built_from_the_file_whatever_the_api_says(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The keyless source feeds the table, so every clone builds the same one; the API
    is audited against it."""
    caplog.set_level(logging.INFO)

    paths = build_clean(coffee_adapter, raw_dir.parent)

    manifest = json.loads((paths["market_context"].parent / MANIFEST_NAME).read_text())
    assert set(manifest["inputs"]) == {"psd_coffee"}
    assert "The FAS API and the PSD file agree on all 114 rows" in caplog.text


def test_a_clean_table_without_a_contract_is_refused(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table nobody promised anything about would reach readers unchecked."""
    contracts = dict(coffee_adapter.clean_contracts())
    del contracts["coffee_shops"]
    monkeypatch.setattr(coffee_adapter, "clean_contracts", lambda: contracts)

    with pytest.raises(ValueError, match="do not match their contracts"):
        build_clean(coffee_adapter, raw_dir.parent)

    assert not (raw_dir.parent / "clean").exists()  # nothing half-written


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("CAFETERIA LA ESQUINA", "coffee"),
        ("CAFECITO TUNTUN", "coffee"),  # a diminutive is still coffee
        ("Café Fuerte", "coffee"),  # accents and case do not matter
        ("PICKUP CAFFEE", "coffee"),
        ("STARBUCKS COFFEE", "coffee"),
        ("NEVERIA Y CAFETERIA LA FLOR", "coffee"),  # a name that says coffee sells coffee
        ("CAFETERIA ESCOLAR PRIMARIA BENITO JUAREZ", "school"),  # a tuck shop is not a cafe
        ("PALETERIA LA MICHOACANA", "ice_cream"),
        ("JUGOS Y LICUADOS DOÑA MARY", "juice"),
        ("FUENTE DE SODAS EL OASIS", "soda_fountain"),
        ("LA ESQUINA DEL TE", "tea"),
        ("TIERRA GARAT", "unclassified"),  # a roaster whose name says nothing a rule reads
        ("SIN NOMBRE", "unnamed"),
        ("", "unnamed"),
        (None, "unnamed"),
    ],
)
def test_a_place_is_what_its_name_says(name: str | None, kind: str) -> None:
    names = pl.DataFrame({"name": [name]}, schema={"name": pl.String})

    labelled = names.select(shop_kind(pl.col("name"), RULES.shop_kinds).alias("kind"))

    assert labelled["kind"].item() == kind


def test_an_osm_tag_nobody_mapped_stops_the_run(frames: Frames) -> None:
    unmapped = dict(frames)
    unmapped["osm_places"] = set_first(frames["osm_places"], "amenity", "fast_food")

    with pytest.raises(ValueError, match="fast_food"):
        clean_coffee_shops(unmapped, frames["cdmx_boroughs"], RULES)


def test_a_place_both_registers_list_is_linked_both_ways(frames: Frames) -> None:
    """The recorded Starbucks is in both: linked, so a count across them counts it once."""
    table = shops(frames).filter(pl.col("matched_shop_id").is_not_null())

    assert dict(zip(table["shop_id"], table["matched_shop_id"], strict=True)) == {
        "denue-3": "osm-node-319644388",
        "osm-node-319644388": "denue-3",
    }
    kinds = dict(zip(table["kind_basis"], table["kind"], strict=True))
    assert kinds == {"name": "coffee", "tag": "coffee"}


CROP = domains.coffee.adapter().config.production


def test_a_municipality_split_across_districts_becomes_one_row(frames: Frames) -> None:
    """Ocosingo sits in three CADERs of two districts: one municipality, summed."""
    production = clean_mexico_production(frames["siap_agricola"], CROP)

    ocosingo = production.filter(pl.col("municipality_id") == "07059").row(0, named=True)
    assert production["municipality_id"].is_unique().all()
    assert ocosingo["planted_ha"] == 2870 + 1008 + 1860
    # Yield from the totals, not an average of the rows' own yields.
    assert ocosingo["yield_t_per_ha"] == pytest.approx(
        ocosingo["production_t"] / ocosingo["harvested_ha"]
    )
    assert "Avena forrajera en verde" not in production["municipality"].to_list()


def test_the_municipal_key_is_inegis(frames: Frames) -> None:
    """Zero-padded to CVEGEO, so the table joins INEGI's boundary layers."""
    production = clean_mexico_production(frames["siap_agricola"], CROP)

    comala = production.filter(pl.col("municipality") == "Comala").row(0, named=True)
    assert (comala["state_id"], comala["municipality_id"]) == ("06", "06003")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        pytest.param("Nomunidad", "Kilogramo", "Nomunidad", id="unit-changed"),
        pytest.param("Nomcultivo", "Café pergamino", "Nomcultivo", id="crop-renamed"),
    ],
)
def test_the_same_crop_id_meaning_something_else_stops_the_run(
    frames: Frames, column: str, value: str, message: str
) -> None:
    changed = frames["siap_agricola"].with_columns(pl.lit(value).alias(column))

    with pytest.raises(ValueError, match=message):
        clean_mexico_production(changed, CROP)


def test_a_crop_siap_does_not_carry_says_where_to_look(frames: Frames) -> None:
    other = CROP.model_copy(update={"crop_id": "0000000"})

    with pytest.raises(ValueError, match=r"production.crop_id"):
        clean_mexico_production(frames["siap_agricola"], other)


def test_nothing_harvested_means_no_yield_rather_than_zero(frames: Frames) -> None:
    unharvested = set_first(
        frames["siap_agricola"].filter(pl.col("Idcultivo") == CROP.crop_id), "Cosechada", 0.0
    )
    unharvested = set_first(unharvested, "Volumenproduccion", 0.0)
    only_first = unharvested.head(1)

    row = clean_mexico_production(only_first, CROP).row(0, named=True)

    assert (row["yield_t_per_ha"], row["rural_price_mxn_per_t"]) == (None, None)
