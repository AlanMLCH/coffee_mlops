"""Every urban AGEB with what is known about it, and the `zones` model's `enrich`.

One row per zone of the 2020 Census: who lives there and how densely, their schooling,
who works, who is over 65, which homes have internet, a car or a computer; the Metro and
Metrobús stations inside it; the food and drink places of other kinds DENUE lists in it
(juice bars, ice cream, soda fountains - a street that sells to people walking by); and
its borough's jobs per resident, the daytime population the census does not count. And
the coffee shops DENUE lists in it by name, which is what the model predicts.

The study of zones reads the same table, so the model and the study never disagree on
what a zone is. polars only: the prediction API looks zones up too.
"""

from collections.abc import Mapping
from datetime import date

import polars as pl

from domains.coffee.schemas import METRO, METROBUS

ZONES_TABLE = "census_zones"
SHOPS_TABLE = "coffee_shops"
STATIONS_TABLE = "transit_stations"
BOROUGHS_TABLE = "boroughs"
ZONE_CONTEXT = (ZONES_TABLE, SHOPS_TABLE, STATIONS_TABLE, BOROUGHS_TABLE)

COFFEE = "coffee"
# DENUE's places of the same activity that are not coffee shops, by name. Unclassified
# names are not counted: a third of the ones OSM knows are coffee shops, which would put
# the target into a feature.
OTHER_KINDS = ("juice", "ice_cream", "soda_fountain", "tea")
CENSUS = "2020"
CENSUS_DAY = date(2020, 3, 15)  # the census' reference date: what a zone is "as of"

PER_DWELLING = {
    "internet_pct": "dwellings_with_internet",
    "car_pct": "dwellings_with_car",
    "computer_pct": "dwellings_with_computer",
}


def zone_profile(
    zones: pl.DataFrame, shops: pl.DataFrame, stations: pl.DataFrame, boroughs: pl.DataFrame
) -> pl.DataFrame:
    """A row per zone. Shares are of the zone's lived-in dwellings or people; a figure
    INEGI withheld, or a zone without dwellings, stays null."""
    denue = shops.filter(pl.col("source") == "denue")
    places = denue.group_by("zone_id").agg(
        (pl.col("kind") == COFFEE).sum().cast(pl.Int64).alias("coffee_shops"),
        pl.col("kind").is_in(OTHER_KINDS).sum().cast(pl.Int64).alias("other_places"),
    )
    boarding = stations.group_by("zone_id").agg(
        (pl.col("system") == METRO).sum().cast(pl.Int64).alias("metro_stations"),
        (pl.col("system") == METROBUS).sum().cast(pl.Int64).alias("metrobus_stations"),
    )
    daytime = boroughs.select(
        "borough_id",
        (pl.col("jobs_estimate") / pl.col("population")).alias("borough_jobs_per_resident"),
    )
    people, dwellings = pl.col("population"), pl.col("dwellings")
    return (
        zones.select(
            "zone_id",
            "borough_id",
            "borough",
            "population",
            "area_km2",
            (people / pl.col("area_km2")).alias("people_per_km2"),
            "schooling_years",
            *[
                pl.when(dwellings > 0).then(100 * pl.col(count) / dwellings).alias(name)
                for name, count in PER_DWELLING.items()
            ],
            pl.when(people > 0)
            .then(100 * pl.col("people_65_plus") / people)
            .alias("aged_65_plus_pct"),
            pl.when(people > 0)
            .then(100 * pl.col("economically_active") / people)
            .alias("active_pct"),
        )
        .join(places, on="zone_id", how="left")
        .join(boarding, on="zone_id", how="left")
        .join(daytime, on="borough_id", how="left")
        .with_columns(
            pl.col("coffee_shops", "other_places", "metro_stations", "metrobus_stations").fill_null(
                0
            )
        )
    )


def add_zone_profile(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each zone asked about, as the census and the registers describe it. A zone is
    looked up by its key alone, in batch and online alike."""
    profile = zone_profile(
        context[ZONES_TABLE], context[SHOPS_TABLE], context[STATIONS_TABLE], context[BOROUGHS_TABLE]
    )
    return (
        items.select("zone_id")
        .join(profile, on="zone_id", how="left")
        .with_columns(
            pl.lit(CENSUS).alias("census"),
            pl.lit(CENSUS_DAY).alias("census_day"),
            pl.col("coffee_shops").cast(pl.Float64),
        )
    )
