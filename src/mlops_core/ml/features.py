"""Features layer: clean reviews + point-in-time market context -> model-ready table.

Only stateless, row-wise transforms live here. Anything fitted on data (encoders,
rare-category grouping, imputation) lives in the model pipeline, so it is learned
on the training split only and ships with the model to serving.
"""

from datetime import UTC, datetime
from pathlib import Path

import pandera.polars as pa
import polars as pl

from mlops_core.config import DomainConfig, ModelSpec
from mlops_core.contracts import check_contract
from mlops_core.storage import latest_partition, read_table, write_table

KEY_COLUMNS = ["review_id", "snapshot", "grading_date"]


def market_features(context: pl.DataFrame) -> pl.DataFrame:
    """One row per (country, market_year) with the context features."""
    production = pl.col("production")
    return context.select(
        "country",
        "market_year",
        production.alias("ctx_production"),
        pl.when(production > 0)
        .then(pl.col("arabica_production") / production)
        .alias("ctx_arabica_share"),
        # Can exceed 1: re-exports and stock drawdowns.
        pl.when(production > 0).then(pl.col("exports") / production).alias("ctx_export_share"),
        pl.col("domestic_consumption").alias("ctx_domestic_consumption"),
    )


def add_market_context(items: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Point-in-time join: an item graded in year Y sees market year Y-1, the latest one
    that was complete at grading time. Shared by the batch build and online serving."""
    market_year = (pl.col("grading_date").dt.year() - 1).alias("market_year")
    return items.with_columns(market_year).join(
        market_features(context), on=["country", "market_year"], how="left"
    )


def review_features_schema(spec: ModelSpec) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        name="review_features",
        strict=True,
        unique=["review_id"],
        columns={
            "review_id": pa.Column(pl.String),
            "snapshot": pa.Column(pl.String),
            "grading_date": pa.Column(pl.Date),
            **{c: pa.Column(pl.String, nullable=True) for c in spec.categorical},
            **{c: pa.Column(pl.Float64, nullable=True) for c in spec.numeric},
            spec.target: pa.Column(pl.Float64),
        },
    )


def build_review_features(
    reviews: pl.DataFrame, context: pl.DataFrame, spec: ModelSpec
) -> pl.DataFrame:
    # Leakage columns are dropped here, so no downstream consumer can pick them up.
    return add_market_context(reviews, context).select(
        *KEY_COLUMNS,
        *spec.categorical,
        *[pl.col(c).cast(pl.Float64) for c in spec.numeric],
        spec.target,
    )


def build_features(config: DomainConfig, data_dir: Path, at: datetime | None = None) -> Path:
    """Read the latest clean tables, build the feature table, check it, write Parquet."""
    clean_dir = data_dir / "clean"
    lineage = {}
    for table in ("coffee_reviews", "market_context"):
        partition = latest_partition(clean_dir / table)
        lineage[table] = partition.name if partition else ""
    features = build_review_features(
        read_table(clean_dir / "coffee_reviews"),
        read_table(clean_dir / "market_context"),
        config.model,
    )
    features = check_contract(review_features_schema(config.model), features)
    return write_table(
        features, data_dir / "features" / "review_features", lineage, at or datetime.now(UTC)
    )
