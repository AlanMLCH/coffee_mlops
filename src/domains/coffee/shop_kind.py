"""The `shop_kind` model's `enrich`: whether a place DENUE lists with coffee shops, ice
cream parlours, juice bars and soda fountains is a coffee shop, when its name does not say.

One item is one DENUE place. Its name decided its kind when it carried a word a rule
knows (CAFÉ, NEVERÍA, JUGOS); 2,802 of 9,860 carry none and stay unclassified - and OSM,
where it knows them, calls 35 of 37 coffee shops. Those are what this model is for. It
learns from the places the rules did classify - a coffee shop, or another kind - what
else a place says: the other words of its name, how many people work there, when it was
registered, and the zone around it. The words the rules read are never features: the
model would only learn the rules back, and an unclassified name has none of them.

A classified place has a target (1, a coffee shop; 0, any other kind); an unclassified
one has none, and is scored. The sum of its probabilities is how many coffee shops the
unclassified places hold.
"""

import re
import unicodedata
from collections.abc import Mapping

import polars as pl

from domains.coffee.zone_profile import ZONE_CONTEXT, zone_profile

COFFEE = "coffee"
# Kinds no rule decided: an unnamed place, or a name no rule reads.
UNDECIDED = ("unclassified", "unnamed")
SHOP_CONTEXT = ZONE_CONTEXT
# The thirty most frequent words of three letters or more in DENUE's names (2026-10-02)
# that no kind rule reads, function words aside: chosen by how often they appear, never
# by what they predict. KRISPY and KREME come together, and no rule ever classified them.
NAME_WORDS = (
    "puesto", "venta", "desayunos", "crepas", "snacks", "tortas", "restaurante", "kreme",
    "krispy", "turno", "postres", "ensaladas", "casa", "barra", "antojitos", "pasteleria",
    "cocteles", "cocina", "dulce", "matutino", "creperia", "gourmet", "pan", "comida",
    "tierra", "plaza", "neverias", "refresquerias", "similares", "oasis",
)  # fmt: skip
ZONE_TRAITS = ("people_per_km2", "schooling_years", "internet_pct", "aged_65_plus_pct")


def name_words(name: str | None) -> list[str]:
    """A name's words, lower case and without accents."""
    plain = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in plain if not unicodedata.combining(c)).casefold()
    return re.findall(r"[a-z0-9]+", folded)


def add_place_traits(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """DENUE's places - or a request - with the words of their names, their size and
    their zone; and, where a rule classified them, whether they are a coffee shop."""
    profile = zone_profile(*(context[table] for table in ZONE_CONTEXT)).select(
        "zone_id", pl.col("borough_id").alias("zone_borough_id"), *ZONE_TRAITS,
        "metro_stations", "metrobus_stations",
    )  # fmt: skip
    words = pl.col("name").map_elements(name_words, return_dtype=pl.List(pl.String))
    kind = pl.col("kind")
    return (
        items.filter(pl.col("source") == "denue")
        # A request may leave these out: typed as text all the same, or nothing joins.
        .with_columns(pl.col("zone_id", "borough_id", "employees_band", "name").cast(pl.String))
        .with_columns(words.alias("_words"))
        .join(profile, on="zone_id", how="left")
        .with_columns(
            pl.coalesce("borough_id", "zone_borough_id").alias("borough_id"),
            # A place outside every zone is a group of its own: it trains or tests alone.
            pl.coalesce("zone_id", pl.concat_str(pl.lit("none-"), "shop_id")).alias("block"),
            pl.col("listed_since").dt.year().cast(pl.String).alias("listed_year"),
            pl.col("listed_since").dt.year().cast(pl.Float64).alias("listed_since_year"),
            pl.col("_words").list.len().cast(pl.Float64).alias("name_word_count"),
            *[
                pl.col("_words").list.contains(word).cast(pl.Float64).alias(f"name_{word}")
                for word in NAME_WORDS
            ],
            pl.when(kind.is_in(UNDECIDED))
            .then(None)
            .otherwise((kind == COFFEE).cast(pl.Float64))
            .alias("is_coffee"),
        )
        .drop("_words", "zone_borough_id")
    )
