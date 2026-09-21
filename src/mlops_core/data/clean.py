"""Clean layer.

- Both CQI snapshots -> one canonical `coffee_reviews` table (one row per graded lot).
- USDA PSD (long format) -> `market_context` (one row per country and market year).
- INEGI's borough polygons -> `boroughs` (one row per alcaldia, geometry as WKB).
- DENUE + OpenStreetMap -> `coffee_shops` (one row per place, placed in a borough).

Transforms are pure functions over validated frames; `build_clean` does the I/O.
Every rule that encodes coffee knowledge (aliases, vocabularies, plausible ranges)
comes from the domain config, not from this module.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from mlops_core.config import CleaningConfig, DomainConfig
from mlops_core.contracts import check_contract
from mlops_core.data.geo import attribute_points
from mlops_core.data.schemas import (
    BOROUGHS,
    COFFEE_SHOPS,
    MARKET_CONTEXT,
    PSD_ATTRIBUTES,
    SENSORY_COLUMNS,
    SENSORY_SCORES,
    SENSORY_SCORES_2018,
    coffee_reviews_schema,
)
from mlops_core.data.validate import validate_raw
from mlops_core.storage import write_table

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["country", "region", "variety", "processing_method", "color", "grading_date"]

# DENUE packs entity + municipality + locality into `AreaGeo`; the first five characters
# are the borough's official CVEGEO, the same key INEGI's polygons carry.
BOROUGH_ID_LENGTH = 5
SHOP_SOURCES = ("denue_cafes", "osm_cafes")


def altitude_from_text(text: pl.Expr) -> pl.Expr:
    """'1200', '1200 - 1300', '1200~1600' -> mean of the numbers in the text."""
    return (
        text.str.extract_all(r"\d+(?:\.\d+)?").list.eval(pl.element().cast(pl.Float64)).list.mean()
    )


def parse_grading_date(text: pl.Expr) -> pl.Expr:
    """'April 4th, 2015' -> 2015-04-04."""
    return (
        text.str.strip_chars()
        .str.replace(r"(\d+)(st|nd|rd|th),", "$1,")
        .str.strptime(pl.Date, "%B %d, %Y")
    )


def _harmonize_2018(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        (pl.lit("cqi_2018-") + pl.col("")).alias("review_id"),
        pl.lit("cqi_2018").alias("snapshot"),
        pl.col("Country.of.Origin").alias("country"),
        pl.col("Region").alias("region"),
        pl.col("Variety").alias("variety"),
        pl.col("Processing.Method").alias("processing_method"),
        pl.col("Color").alias("color"),
        pl.col("Grading.Date").alias("grading_date"),
        pl.col("altitude_mean_meters").alias("altitude_m"),
        (pl.col("Moisture") * 100).round(2).alias("moisture_pct"),  # stored as a fraction
        pl.col("Category.One.Defects").alias("category_one_defects"),
        pl.col("Category.Two.Defects").alias("category_two_defects"),
        pl.col("Quakers").alias("quakers"),
        *[
            pl.col(raw).alias(canonical)
            for raw, canonical in zip(SENSORY_SCORES_2018, SENSORY_COLUMNS, strict=True)
        ],
        pl.col("Total.Cup.Points").alias("total_cup_points"),
    )


def _harmonize_2023(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        (pl.lit("cqi_2023-") + pl.col("ID")).alias("review_id"),
        pl.lit("cqi_2023").alias("snapshot"),
        pl.col("Country of Origin").alias("country"),
        pl.col("Region").alias("region"),
        pl.col("Variety").alias("variety"),
        pl.col("Processing Method").alias("processing_method"),
        pl.col("Color").alias("color"),
        pl.col("Grading Date").alias("grading_date"),
        altitude_from_text(pl.col("Altitude")).alias("altitude_m"),
        pl.col("Moisture Percentage").alias("moisture_pct"),
        pl.col("Category One Defects").alias("category_one_defects"),
        pl.col("Category Two Defects").alias("category_two_defects"),
        pl.col("Quakers").alias("quakers"),
        *[
            pl.col(raw).alias(canonical)
            for raw, canonical in zip(SENSORY_SCORES, SENSORY_COLUMNS, strict=True)
        ],
        pl.col("Total Cup Points").alias("total_cup_points"),
    )


def _map_vocabulary(
    df: pl.DataFrame, column: str, mapping: Mapping[str, str | None], config_key: str
) -> pl.DataFrame:
    """Map free-text labels onto a closed vocabulary; an unseen label stops the pipeline."""
    labels = pl.col(column).str.to_lowercase()
    unknown = sorted(set(df.select(labels.drop_nulls())[column]) - mapping.keys())
    if unknown:
        raise ValueError(
            f"Unmapped {column} labels {unknown}: add them to `cleaning.{config_key}` "
            "in the domain config"
        )
    return df.with_columns(
        labels.replace_strict(dict(mapping), default=None, return_dtype=pl.String)
    )


def clean_reviews(frames: Mapping[str, pl.DataFrame], rules: CleaningConfig) -> pl.DataFrame:
    df = pl.concat([_harmonize_2018(frames["cqi_2018"]), _harmonize_2023(frames["cqi_2023"])])

    # The 2018 scrape writes missing text as "" rather than NA.
    df = df.with_columns(
        pl.when(pl.col(c).str.strip_chars() != "").then(pl.col(c).str.strip_chars())
        for c in TEXT_COLUMNS
    )
    df = _map_vocabulary(df, "processing_method", rules.processing_methods, "processing_methods")
    df = _map_vocabulary(df, "color", rules.colors, "colors")

    altitude_low, altitude_high = rules.altitude_m
    df = df.with_columns(
        pl.col("country").replace(rules.country_aliases),
        pl.col("variety").str.to_lowercase(),
        parse_grading_date(pl.col("grading_date")),
        pl.when(pl.col("altitude_m").is_between(altitude_low, altitude_high)).then("altitude_m"),
        # 0% moisture is physically impossible for green coffee: it means "not measured".
        pl.when(pl.col("moisture_pct") > 0).then("moisture_pct"),
    )

    # No country: cannot be placed or joined. Total of 0: the lot was never cupped (every
    # score is 0). Low but real totals stay: per-cup scores like Clean Cup can be 1.33.
    keep = pl.col("country").is_not_null() & (pl.col("total_cup_points") > 0)
    dropped = df.filter(~keep)["review_id"].to_list()
    if dropped:
        logger.warning("Dropped %d reviews (no country or never cupped): %s", len(dropped), dropped)
    return df.filter(keep).sort("review_id")


def clean_market_context(psd: pl.DataFrame) -> pl.DataFrame:
    wide = psd.select(
        pl.col("Country_Name").alias("country"),
        pl.col("Market_Year").alias("market_year"),
        pl.col("Attribute_Description").replace_strict(PSD_ATTRIBUTES).alias("attribute"),
        pl.col("Value"),
    ).pivot(on="attribute", index=["country", "market_year"], values="Value")
    # An attribute nobody reported in this download still gets its (null) column.
    missing = [c for c in PSD_ATTRIBUTES.values() if c not in wide.columns]
    return (
        wide.with_columns(pl.lit(None, pl.Float64).alias(c) for c in missing)
        .select("country", "market_year", *PSD_ATTRIBUTES.values())
        .sort("country", "market_year")
    )


@dataclass(frozen=True)
class Reconciliation:
    """How two sources of the same table compare, key by key."""

    rows: int
    only_file: int
    only_api: int
    different: int

    @property
    def agree(self) -> bool:
        return self.only_file == self.only_api == self.different == 0


def reconcile_market_sources(file: pl.DataFrame, api: pl.DataFrame) -> Reconciliation:
    """Compare the PSD file with the FAS API, the way the join is compared with DENUE.

    `market_context` is built from the file: it needs no key and is one request, so any
    clone can rebuild it. The API is the same data by another road, and checking one
    against the other on every build is what turns "the API is equivalent" from
    something verified once into something that stays true - or says when it stops.
    A difference is reported, not raised: the file is still a consistent source.
    """
    key = ["Country_Code", "Market_Year", "Attribute_ID"]
    joined = file.select(*key, pl.col("Value").alias("file")).join(
        api.select(*key, pl.col("Value").alias("api")), on=key, how="full", coalesce=True
    )
    result = Reconciliation(
        rows=joined.height,
        only_file=joined["api"].null_count(),
        only_api=joined["file"].null_count(),
        different=joined.filter(pl.col("file") != pl.col("api")).height,
    )
    if result.agree:
        logger.info("The FAS API and the PSD file agree on all %d rows", result.rows)
    else:
        logger.warning(
            "The FAS API and the PSD file disagree: %d rows only in the file, %d only in "
            "the API, %d with different values (market_context is built from the file)",
            result.only_file,
            result.only_api,
            result.different,
        )
    return result


def clean_boroughs(areas: pl.DataFrame) -> pl.DataFrame:
    """The boundary layer as the domain's own table: alcaldias, with their polygons."""
    return areas.rename({"area_id": "borough_id", "area_name": "borough"}).select(
        "borough_id", "borough", "area_km2", "boundary"
    )


def _blank_to_null(column: pl.Expr) -> pl.Expr:
    """DENUE writes an unknown value as an empty string, like the 2018 CQI scrape."""
    return pl.when(column.str.len_chars() > 0).then(column)


def _denue_shops(denue: pl.DataFrame) -> pl.DataFrame:
    """DENUE's register, narrowed to what a place is in this project.

    Every row of the activity class is kept, ice-cream parlours and soda fountains
    included: the class is wider than coffee, and no name-based filter would be honest
    before it has been measured. `source` says where a row came from, so a later rule
    can be applied -- and argued with -- on top of this table instead of inside it.
    """
    return denue.select(
        (pl.lit("denue-") + pl.col("Id")).alias("shop_id"),
        pl.lit("denue").alias("source"),
        _blank_to_null(pl.col("Nombre")).alias("name"),
        pl.lit(None, pl.String).alias("brand"),  # DENUE records no brand
        _blank_to_null(pl.col("Estrato")).alias("employees_band"),
        pl.col("Latitud").alias("latitude"),
        pl.col("Longitud").alias("longitude"),
        pl.col("AreaGeo").str.slice(0, BOROUGH_ID_LENGTH).alias("declared_borough_id"),
    )


def _osm_shops(osm: pl.DataFrame) -> pl.DataFrame:
    """OSM's elements, keyed by type and id because a node and a way can share a number."""
    return osm.select(
        (pl.lit("osm-") + pl.col("type") + pl.lit("-") + pl.col("id").cast(pl.String)).alias(
            "shop_id"
        ),
        pl.lit("osm").alias("source"),
        pl.col("name"),
        pl.col("brand"),
        pl.lit(None, pl.String).alias("employees_band"),  # OSM records no size
        pl.col("latitude"),
        pl.col("longitude"),
        pl.lit(None, pl.String).alias("declared_borough_id"),  # nor which borough it is in
    )


def clean_coffee_shops(frames: Mapping[str, pl.DataFrame], areas: pl.DataFrame) -> pl.DataFrame:
    """Both registers as one table of places, each one placed inside a borough.

    The sources sit side by side rather than merged: DENUE is the official register, OSM
    is what people mapped, they disagree about what exists, and deciding which is right
    is analysis, not cleaning.
    """
    readers = {"denue_cafes": _denue_shops, "osm_cafes": _osm_shops}
    parts = [reader(frames[name]) for name, reader in readers.items() if name in frames]
    if not parts:
        raise ValueError("No register of places has been ingested: run extract first")
    shops = pl.concat(parts)

    located = shops.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
    if located.height != shops.height:
        # OSM is crowd-sourced: an element can be tagged without ever being placed.
        logger.warning("Dropped %d places with no coordinate", shops.height - located.height)

    placed = attribute_points(located, areas, "latitude", "longitude").rename(
        {"area_id": "borough_id", "area_name": "borough"}
    )
    _report_placement(placed)
    return placed.select(*COFFEE_SHOPS.columns).sort("shop_id")


def _report_placement(placed: pl.DataFrame) -> None:
    """Say how the join went, in the two ways it can go wrong.

    A point outside every polygon is unplaceable and easy to notice. A point inside the
    wrong polygon is worse, because nothing about it looks wrong -- so where the source
    states its own borough (DENUE does, OSM does not) the join is scored against it.
    That is what proves the projection, the axis order and the encoding are right: on
    the real data it agrees 9,860 times out of 9,860.
    """
    outside = placed.filter(pl.col("borough_id").is_null()).height
    if outside:
        logger.warning("%d places fell outside every borough", outside)
    audited = placed.filter(pl.col("declared_borough_id").is_not_null())
    if audited.is_empty():
        return
    agreed = int((audited["declared_borough_id"] == audited["borough_id"]).sum())
    report = logger.info if agreed == audited.height else logger.warning
    report(
        "The spatial join agrees with the source's own borough on %d of %d places (%.2f%%)",
        agreed,
        audited.height,
        100 * agreed / audited.height,
    )


def build_clean(
    config: DomainConfig, data_dir: Path, at: datetime | None = None
) -> dict[str, Path]:
    """Validate the latest raw data, clean it, check the clean contracts, write Parquet."""
    sources = validate_raw(config, data_dir / "raw")
    frames = {name: source.frame for name, source in sources.items()}
    lineage = {name: source.artifact.partition.name for name, source in sources.items()}

    reviews = check_contract(
        coffee_reviews_schema(config.cleaning), clean_reviews(frames, config.cleaning)
    )
    context = check_contract(MARKET_CONTEXT, clean_market_context(frames["psd_coffee"]))
    if "fas_psd_coffee" in frames:  # absent without a key, and nothing depends on it
        reconcile_market_sources(frames["psd_coffee"], frames["fas_psd_coffee"])
    areas = frames["cdmx_boroughs"]
    boroughs = check_contract(BOROUGHS, clean_boroughs(areas))
    shops = check_contract(COFFEE_SHOPS, clean_coffee_shops(frames, areas))

    built_at = at or datetime.now(UTC)
    clean_dir = data_dir / "clean"
    return {
        "coffee_reviews": write_table(
            reviews,
            clean_dir / "coffee_reviews",
            {k: lineage[k] for k in ("cqi_2018", "cqi_2023")},
            built_at,
        ),
        "market_context": write_table(
            context, clean_dir / "market_context", {"psd_coffee": lineage["psd_coffee"]}, built_at
        ),
        "boroughs": write_table(
            boroughs, clean_dir / "boroughs", {"cdmx_boroughs": lineage["cdmx_boroughs"]}, built_at
        ),
        "coffee_shops": write_table(
            shops,
            clean_dir / "coffee_shops",
            # Which registers this build actually saw: DENUE is absent without a token.
            {k: v for k, v in lineage.items() if k in (*SHOP_SOURCES, "cdmx_boroughs")},
            built_at,
        ),
    }
