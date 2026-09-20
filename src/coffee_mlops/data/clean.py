"""Clean layer.

- Both CQI snapshots -> one canonical `coffee_reviews` table (one row per graded lot).
- USDA PSD (long format) -> `market_context` (one row per country and market year).

Transforms are pure functions over validated frames; `build_clean` does the I/O.
Every rule that encodes coffee knowledge (aliases, vocabularies, plausible ranges)
comes from the domain config, not from this module.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from coffee_mlops.config import CleaningConfig, DomainConfig
from coffee_mlops.contracts import check_contract
from coffee_mlops.data.schemas import (
    MARKET_CONTEXT,
    PSD_ATTRIBUTES,
    SENSORY_COLUMNS,
    SENSORY_SCORES,
    SENSORY_SCORES_2018,
    coffee_reviews_schema,
)
from coffee_mlops.data.validate import validate_raw
from coffee_mlops.storage import write_table

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["country", "region", "variety", "processing_method", "color", "grading_date"]


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
    }
