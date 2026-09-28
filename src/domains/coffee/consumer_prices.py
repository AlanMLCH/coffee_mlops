"""PROFECO's shelf prices -> `consumer_prices`: one row per price recorded.

What a jar of instant or a bag of ground coffee costs in a supermarket, a convenience
store, a market or a pharmacy, anywhere in Mexico, fortnight by fortnight. The table
puts it per kilogram, so a 120 g jar and a 400 g bag can be compared with each other
and with a specialty roaster's bag; says what else the presentation declares (a blend
with sugar or caramel is priced per kilogram of both; decaf); and gives the city's
prices the borough their store is in.

Every read of the survey is kept (the source accumulates): PROFECO publishes the year so
far, so the archive of January 2027 will no longer hold 2026. Each fortnight's rows come
from the latest read that carries that fortnight's file - a correction can only come
later - and a fortnight no newer read carries is kept from the read that had it.

One thing in the files is repaired, from what the files themselves say: June's wrote
some accented letters as a question mark ("Nescafé. Cl?sico", "Naucalpan de Ju?rez").
A value is restored only when the same column spells it whole somewhere else, and
only one spelling fits. Otherwise brands, chains and presentations stay as PROFECO
writes them.

The borough is the one the store declares, matched to INEGI's names, and its
coordinates are checked against it rather than trusted over it. DENUE's coordinates
agree with its declared boroughs 9,860 times out of 9,860; PROFECO's disagree for 7 of
the city's 120 stores (2026-09-27), two within 200 m of the line and the rest 1.5 to
12 km away, in the wrong borough: a market known to be in Azcapotzalco lands in
Iztacalco. There the coordinates are the error, so the declaration is kept, and the
log says how often the two agree.
"""

import logging
import re

import polars as pl

from domains.coffee.config import ConsumerPricesConfig
from domains.coffee.roaster_sheets import bag_grams
from domains.coffee.schemas import CONSUMER_PRICES
from domains.coffee.sources.profeco import FILE, record_date
from mlops_core.data.geo import attribute_points

logger = logging.getLogger(__name__)

# The columns some of whose values lost letters.
LOST_LETTERS = ("marca", "presentacion", "nombre_comercial", "municipio")
_LOST = "?"
_NON_ASCII = "[^\\x00-\\x7f]"  # the one letter a "?" stands for
_KEY = "__borough_key"  # a name folded, to meet INEGI's spelling of it
READ_AT = "ingested_at"  # the read each row came from, when the reads are stacked


def clean_consumer_prices(
    raw: pl.DataFrame, areas: pl.DataFrame, rules: ConsumerPricesConfig
) -> pl.DataFrame:
    """The survey's coffee rows, per kilogram, each placed in a borough if it is in one."""
    raw = latest_fortnights(raw)
    restored = raw.with_columns(restore_lost_letters(raw[column]) for column in LOST_LETTERS)
    date = record_date(pl.col("fecha_registro"))
    month = date.dt.truncate("1mo")
    presentation = folded(pl.col("presentacion"))
    prices = restored.select(
        date.alias("date"),
        pl.when(date.dt.day() <= 15)
        .then(month)
        .otherwise(month.dt.offset_by("15d"))
        .alias("fortnight"),
        pl.col("producto").replace_strict(rules.products, return_dtype=pl.String).alias("product"),
        pl.col("marca").alias("brand"),
        pl.col("presentacion").alias("presentation"),
        presentation.str.contains(rules.sweetened).alias("sweetened"),
        presentation.str.contains(rules.decaf).alias("decaf"),
        pl.col("precio").alias("price_mxn"),
        pl.col("cadena_comercial").alias("chain"),
        pl.col("giro").alias("store_type"),
        pl.col("nombre_comercial").alias("store"),
        pl.col("estado").alias("state"),
        pl.col("municipio").alias("municipality"),
        pl.col("latitud").alias("latitude"),
        pl.col("longitud").alias("longitude"),
    )
    sizes = {text: bag_grams(None, text) for text in prices["presentation"].unique().to_list()}
    prices = prices.with_columns(
        pl.col("presentation").replace_strict(sizes, return_dtype=pl.Float64).alias("grams")
    ).with_columns(
        (pl.col("price_mxn") / pl.col("grams") * 1000).round(2).alias("price_mxn_per_kg")
    )
    in_boroughs = _in_boroughs(_once(prices), areas, rules)
    _audit_coordinates(in_boroughs, areas, rules.city)
    return in_boroughs.select(*CONSUMER_PRICES.columns).sort(
        "date", "state", "store", "brand", "presentation", "price_mxn"
    )


def latest_fortnights(raw: pl.DataFrame) -> pl.DataFrame:
    """Each fortnight's file from the latest read that carries it. Without reads stacked
    (no `ingested_at`), the one read there is."""
    if READ_AT not in raw.columns:
        return raw
    latest = pl.col(READ_AT).max().over(FILE)
    kept = raw.filter(pl.col(READ_AT) == latest)
    logger.info(
        "consumer_prices: %d fortnights from %d reads of the survey",
        kept[FILE].n_unique(), raw[READ_AT].n_unique(),
    )  # fmt: skip
    return kept.drop(READ_AT)


def restore_lost_letters(values: pl.Series) -> pl.Series:
    """Put back letters a file wrote as "?", where the column spells the value whole
    elsewhere: "Nescafé. Cl?sico" is "Nescafé. Clásico" because another row says so. A
    value with no such twin, or with more than one that fits, stays as it is."""
    broken = values.filter(values.str.contains(_LOST, literal=True)).unique().to_list()
    if not broken:
        return values
    whole = [value for value in values.unique().drop_nulls().to_list() if _LOST not in value]
    restored, left = {}, []
    for value in broken:
        pattern = re.compile(
            "^" + _NON_ASCII.join(re.escape(part) for part in value.split(_LOST)) + "$"
        )
        fits = [candidate for candidate in whole if pattern.match(candidate)]
        if len(fits) == 1:
            restored[value] = fits[0]
        else:
            left.append(value)
    logger.info(
        "%s: restored %d values with lost letters; %d have no single whole spelling %s",
        values.name, len(restored), len(left), sorted(left)[:5],
    )  # fmt: skip
    return values.replace(restored)


def folded(text: pl.Expr) -> pl.Expr:
    """Lower-case and accent-free, as `roaster_sheets.fold` is, for a whole column."""
    return text.str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()


def _once(prices: pl.DataFrame) -> pl.DataFrame:
    """Each price once: restoring letters can make two rows one. A shelf with two prices
    for one product on one day is left with both, and counted."""
    unique = prices.unique(maintain_order=True)
    if unique.height != prices.height:
        logger.info("consumer_prices: %d rows repeated another", prices.height - unique.height)
    key = ["date", "store", "latitude", "longitude", "brand", "presentation"]
    twice = unique.filter(pl.len().over(key) > 1).select(key).unique().height
    if twice:
        logger.info(
            "consumer_prices: %d times a shelf had two prices for one product on one day; "
            "PROFECO does not say why, so both are kept",
            twice,
        )
    return unique


def _in_boroughs(
    prices: pl.DataFrame, areas: pl.DataFrame, rules: ConsumerPricesConfig
) -> pl.DataFrame:
    """The city's prices in the borough their store declares, with INEGI's key and
    spelling ("Gustavo a. Madero" is "Gustavo A. Madero"); everywhere else, none."""
    city = rules.city
    official = areas.select(
        folded(pl.col("area_name")).alias(_KEY),
        pl.col("area_id").alias("borough_id"),
        pl.col("area_name").alias("borough"),
    )
    name = pl.col("municipality").replace(rules.borough_aliases)
    keyed = prices.with_columns(pl.when(pl.col("state") == city).then(folded(name)).alias(_KEY))
    declared = keyed.join(official, on=_KEY, how="left", maintain_order="left")
    unknown = declared.filter(pl.col(_KEY).is_not_null() & pl.col("borough_id").is_null())
    if not unknown.is_empty():
        logger.warning(
            "consumer_prices: %d of %s's prices name no borough INEGI has: %s",
            unknown.height, city, sorted(unknown["municipality"].unique().to_list()),
        )  # fmt: skip
    return declared.drop(_KEY)


def _audit_coordinates(prices: pl.DataFrame, areas: pl.DataFrame, city: str) -> None:
    """Where each store's coordinates fall, against the borough it declares: said, not
    enforced. Counted by store, since a store's prices share its coordinates."""
    stores = prices.group_by("store", "latitude", "longitude").agg(
        pl.col("borough_id").first(), pl.len().alias("prices")
    )
    placed = attribute_points(stores, areas, "latitude", "longitude")
    declared = placed.filter(pl.col("borough_id").is_not_null())
    if not declared.is_empty():
        agree = declared.filter(pl.col("area_id") == pl.col("borough_id"))
        report = logger.info if agree.height == declared.height else logger.warning
        report(
            "consumer_prices: coordinates fall in the declared borough for %d of %d stores "
            "(%d of %d prices); the declaration is kept",
            agree.height, declared.height, agree["prices"].sum(), declared["prices"].sum(),
        )  # fmt: skip
    strays = placed.filter(pl.col("borough_id").is_null() & pl.col("area_id").is_not_null())
    if not strays.is_empty():
        logger.warning(
            "consumer_prices: %d stores outside %s have coordinates inside it (%d prices)",
            strays.height, city, strays["prices"].sum(),
        )  # fmt: skip
