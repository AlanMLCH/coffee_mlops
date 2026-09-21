"""Shared Pandera helper. Both pipelines validate frames the same way."""

import pandera.polars as pa
import polars as pl


def check_contract(schema: pa.DataFrameSchema, df: pl.DataFrame) -> pl.DataFrame:
    """Validate and type `df`, raising `SchemaErrors` with every failure at once."""
    # pandera 0.33 (polars backend) crashes with a polars ColumnNotFoundError when it
    # coerces a missing column. Check presence first so it is reported as a SchemaErrors.
    presence = pa.DataFrameSchema({c: pa.Column(nullable=True) for c in schema.columns})
    presence.validate(df, lazy=True)
    return schema.validate(df, lazy=True)
