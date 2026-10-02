"""The `shelf_price` model's `enrich`: what a jar or bag of coffee should cost on a shelf.

One item is one product's price in one store on one day, as PROFECO's survey read it.
The target is that price in pesos of the latest month INEGI's INPC has published:
2024's prices and 2026's on one scale, so the model learns what a product costs where,
not how fast the peso lost value - and a fair price is said in today's pesos. What an
item knows is what the shelf says: the brand, the size, instant or ground, sweetened,
decaffeinated; the chain, its kind of store, and the state.

Set against the price the shelf asked (in the same pesos), the prediction says whether a
reading was a deal or a markup: the batch scores every reading, and the months the model
was not trained on are where a deal is an honest one.

A request describes a product and a store. A brand is matched as the survey writes it
with accents and capitals aside ("nescafe clasico" is "Nescafé. Clásico").
"""

import re
import unicodedata
from collections.abc import Mapping

import polars as pl

SHELF_TABLE = "consumer_prices"
INDEX_TABLE = "consumer_price_index"
SHELF_CONTEXT = (INDEX_TABLE,)


def folded(items: pl.DataFrame, column: str) -> pl.Expr:
    """Lower case, without accents or punctuation: "Nescafé. Clásico" -> "nescafe clasico".
    Folded once per distinct value: a brand is read a hundred thousand times."""
    values = items[column].cast(pl.String).drop_nulls().unique().to_list()
    return (
        pl.col(column)
        .cast(pl.String)  # a request that names no chain sends a column of nothing
        .replace_strict({value: fold(value) for value in values}, default=None)
    )


def fold(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text)
    letters = "".join(c for c in plain if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", letters).strip()


def add_shelf_context(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each reading - or a request - with its product named, and its price in pesos of
    the index's latest month. Readings of one product in one store on one day are one
    item, at their mean price: the survey sometimes records a jar twice."""
    index = context[INDEX_TABLE]
    latest = index.sort("month")["index"].tail(1)
    today = float(latest[0]) if latest.len() else None
    by_month = index.select(pl.col("month"), pl.col("index").alias("_index"))
    flags = [pl.col("sweetened").cast(pl.Float64), pl.col("decaf").cast(pl.Float64)]
    keys = ["store", "brand", "grams", "product", "sweetened", "decaf", "date"]
    readings = (
        items.with_columns(
            folded(items, "brand").alias("brand"), folded(items, "chain").alias("chain")
        )
        .group_by(keys, maintain_order=True)
        .agg(
            pl.col("chain", "store_type", "state").first(),
            pl.col("price_mxn").mean(),
        )
    )
    product_name = pl.format(
        "{} {} g{}{}",
        pl.col("brand"),
        pl.col("grams").cast(pl.Int64),
        pl.when(pl.col("sweetened")).then(pl.lit(" sweetened")).otherwise(pl.lit("")),
        pl.when(pl.col("decaf")).then(pl.lit(" decaf")).otherwise(pl.lit("")),
    )
    return (
        readings.with_columns(pl.col("date").dt.month_start().alias("_month"))
        .join(by_month, left_on="_month", right_on="month", how="left")
        .with_columns(
            pl.concat_str(
                "store", product_name, pl.col("date").dt.strftime("%Y-%m-%d"), separator="|"
            ).alias("reading_id"),
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
            product_name.alias("product_name"),
            *flags,
            _in_todays_pesos(today),
        )
        .drop("_month", "_index")
    )


def _in_todays_pesos(today: float | None) -> pl.Expr:
    """A price in pesos of the index's latest month. A month the index has not published
    yet is priced at the latest month's; without the index (no token), prices stay in
    their own pesos, and the model learns inflation along with everything else."""
    price = pl.col("price_mxn")
    if today is None:
        return price.alias("price_today_mxn")
    return (price * today / pl.col("_index").fill_null(today)).alias("price_today_mxn")
