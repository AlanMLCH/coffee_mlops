"""Clean layer.

- Both CQI snapshots -> one canonical `coffee_reviews` table (one row per graded lot).
- USDA PSD (long format) -> `market_context` (one row per country and market year).
- INEGI's borough polygons -> `boroughs` (one row per alcaldia, geometry as WKB).
- DENUE + OpenStreetMap -> `coffee_shops` (one row per place, placed in a borough).

Transforms are pure functions over validated frames. Reading the raw layer, holding
each table to its contract and writing it with lineage is the core's job
(`mlops_core.data.clean`); `clean_tables` is all the core asks of this module. Every
rule that encodes coffee knowledge (aliases, vocabularies, plausible ranges) comes from
the domain config.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
from polars.expr.whenthen import ChainedThen, Then

from domains.coffee.config import UNCLASSIFIED, CleaningConfig, ShopKindRule
from domains.coffee.schemas import (
    PSD_ATTRIBUTES,
    SENSORY_COLUMNS,
    SENSORY_SCORES,
    SENSORY_SCORES_2018,
    coffee_shops_schema,
)
from mlops_core.adapter import CleanTable
from mlops_core.data.geo import attribute_points, match_places

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["country", "region", "variety", "processing_method", "color", "grading_date"]

# DENUE packs entity + municipality + locality into `AreaGeo`; the first five characters
# are the borough's official CVEGEO, the same key INEGI's polygons carry.
BOROUGH_ID_LENGTH = 5
SHOP_SOURCES = ("denue_cafes", "osm_places")


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


def normalised_name(name: pl.Expr) -> pl.Expr:
    """Upper case, accents stripped, trimmed: `Café  ` and `CAFE` are the same word."""
    return (
        name.str.normalize("NFKD")
        .str.replace_all(r"\p{Mn}", "")
        .str.to_uppercase()
        .str.strip_chars()
    )


def shop_kind(name: pl.Expr, rules: list[ShopKindRule]) -> pl.Expr:
    """What a place is, read from its name: the first rule whose pattern matches.

    A classification, not a filter. Every row stays in the table with its kind, so a
    reader who disagrees with a rule sees exactly which rows it moved - and the rule can
    be scored, which is what `analysis.kind_agreement` does against OSM's own tags.
    """
    normalised = normalised_name(name.fill_null(""))
    first, *rest = rules
    # polars types the chain link by link (Then, then ChainedThen), so the variable
    # is declared as either.
    kind: Then | ChainedThen = pl.when(normalised.str.contains(first.pattern)).then(
        pl.lit(first.kind)
    )
    for rule in rest:
        kind = kind.when(normalised.str.contains(rule.pattern)).then(pl.lit(rule.kind))
    return kind.otherwise(pl.lit(UNCLASSIFIED))


def _denue_shops(denue: pl.DataFrame, rules: CleaningConfig) -> pl.DataFrame:
    """DENUE's register, each establishment labelled with what its name says it is.

    Every row of the activity class is kept - juice stands, ice-cream parlours and
    school tuck shops included - and `kind` says which is which, so a coffee-only view
    is a filter a reader applies, and can argue with, rather than rows that vanished.
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
        shop_kind(pl.col("Nombre"), rules.shop_kinds).alias("kind"),
        pl.lit("name").alias("kind_basis"),
    )


def _osm_shops(osm: pl.DataFrame, rules: CleaningConfig) -> pl.DataFrame:
    """OSM's elements, keyed by type and id because a node and a way can share a number.

    Their kind comes from the mappers' own `amenity` tag, through a closed vocabulary: a
    tag nobody mapped stops the run instead of becoming a guess.
    """
    unknown = sorted(set(osm["amenity"]) - rules.osm_kinds.keys())
    if unknown:
        raise ValueError(f"Unmapped OSM amenities {unknown}: add them to `cleaning.osm_kinds`")
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
        pl.col("amenity").replace_strict(rules.osm_kinds, return_dtype=pl.String).alias("kind"),
        pl.lit("tag").alias("kind_basis"),
    )


def clean_coffee_shops(
    frames: Mapping[str, pl.DataFrame], areas: pl.DataFrame, rules: CleaningConfig
) -> pl.DataFrame:
    """Both registers as one table of places, each one placed inside a borough, given a
    kind, and linked to its twin in the other register when there is one.

    The sources sit side by side rather than merged: DENUE is the official register, OSM
    is what people mapped, and they disagree about what exists. `matched_shop_id` says
    where they agree, so a count across both can avoid counting one place twice.
    """
    readers = {"denue_cafes": _denue_shops, "osm_places": _osm_shops}
    parts = [reader(frames[name], rules) for name, reader in readers.items() if name in frames]
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
    linked = _link_registers(placed, rules)
    return linked.select(*coffee_shops_schema(rules).columns).sort("shop_id")


def _link_registers(shops: pl.DataFrame, rules: CleaningConfig) -> pl.DataFrame:
    """Point each place both registers list at its twin, in both directions."""
    by_source = {
        source: shops.filter(pl.col("source") == source).select(
            pl.col("shop_id").alias("id"), "name", "latitude", "longitude"
        )
        for source in ("denue", "osm")
    }
    match = rules.register_match
    pairs = match_places(
        by_source["denue"],
        by_source["osm"],
        radius_m=match.radius_m,
        min_similarity=match.min_name_similarity,
    )
    links = pl.concat(
        [
            pairs.select(
                pl.col("left_id").alias("shop_id"), pl.col("right_id").alias("matched_shop_id")
            ),
            pairs.select(
                pl.col("right_id").alias("shop_id"), pl.col("left_id").alias("matched_shop_id")
            ),
        ]
    )
    logger.info(
        "%d places are listed by both registers (within %.0f m, names at least %.2f alike)",
        pairs.height,
        match.radius_m,
        match.min_name_similarity,
    )
    return shops.join(links, on="shop_id", how="left")


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


def clean_tables(
    frames: Mapping[str, pl.DataFrame], rules: CleaningConfig
) -> dict[str, CleanTable]:
    """Validated raw frames -> the domain's four clean tables, each with its sources."""
    if "fas_psd_coffee" in frames:  # absent without a key, and nothing depends on it
        reconcile_market_sources(frames["psd_coffee"], frames["fas_psd_coffee"])
    areas = frames["cdmx_boroughs"]
    # Which registers this build actually saw: DENUE is absent without a token.
    shop_inputs = tuple(name for name in (*SHOP_SOURCES, "cdmx_boroughs") if name in frames)
    return {
        "coffee_reviews": CleanTable(clean_reviews(frames, rules), ("cqi_2018", "cqi_2023")),
        "market_context": CleanTable(clean_market_context(frames["psd_coffee"]), ("psd_coffee",)),
        "boroughs": CleanTable(clean_boroughs(areas), ("cdmx_boroughs",)),
        "coffee_shops": CleanTable(clean_coffee_shops(frames, areas, rules), shop_inputs),
    }
