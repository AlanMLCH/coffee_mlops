"""INEGI's intercensal survey -> `borough_profile`: who lives in each borough now.

The 2020 Census counted everyone, once; the 2025 Intercensal Survey asked a sample of 7.3
million dwellings the same questions and more - age, work, the commute, whether a home
is owned or rented - and estimated every municipality from it. Its principal results
come as one file for the whole country, a row per area and estimator: the value, its
standard error, the limits of its interval and its coefficient of variation.

The profile keeps what the config names, for the boroughs the boundary layer has, as a
row per borough and indicator with the value inside its own interval: an estimate read
without its precision reads as a count. Checked, not trusted: the sixteen boroughs'
counts must come to the city's own total, as the census' do.
"""

import logging

import polars as pl

from domains.coffee.config import BoroughProfileConfig
from domains.coffee.schemas import SURVEY_ESTIMATORS

logger = logging.getLogger(__name__)

STATE, TOTAL = "000", "0000"  # the survey's municipality code for the state; a total row


def clean_borough_profile(
    survey: pl.DataFrame, areas: pl.DataFrame, profile: BoroughProfileConfig
) -> pl.DataFrame:
    """The survey's estimates for every borough the boundary layer has, a row per
    borough and indicator. A borough the survey does not name has no rows."""
    ids = areas.select(pl.col("area_id").alias("borough_id"), pl.col("area_name").alias("borough"))
    totals = survey.filter(pl.col("CVE_LOC") == TOTAL).with_columns(
        pl.concat_str("CVE_ENT", "CVE_MUN").alias("borough_id")
    )
    entities = ids["borough_id"].str.slice(0, 2).unique().to_list()
    _check_counts(totals.filter(pl.col("CVE_ENT").is_in(entities)), profile)
    columns = list(profile.indicators)
    long = (
        totals.join(ids, on="borough_id")
        .filter(pl.col("ESTIMADOR").is_in(list(SURVEY_ESTIMATORS)))
        .select("borough_id", "borough", "ESTIMADOR", *columns)
        .unpivot(index=["borough_id", "borough", "ESTIMADOR"], variable_name="column")
    )
    named = pl.DataFrame(
        {
            "column": columns,
            "indicator": [i.name for i in profile.indicators.values()],
            "unit": [i.unit for i in profile.indicators.values()],
        }
    )
    wide = long.with_columns(pl.col("ESTIMADOR").replace_strict(SURVEY_ESTIMATORS)).pivot(
        on="ESTIMADOR", index=["borough_id", "borough", "column"], values="value"
    )
    # Nothing to pivot leaves no estimator columns: an empty table keeps its shape.
    missing = [name for name in SURVEY_ESTIMATORS.values() if name not in wide.columns]
    table = (
        wide.with_columns(pl.lit(None, pl.Float64).alias(name) for name in missing)
        .join(named, on="column")
        .select(
            "borough_id",
            "borough",
            pl.lit(profile.year, pl.Int64).alias("year"),
            "indicator",
            "unit",
            *[pl.col(name).cast(pl.Float64) for name in SURVEY_ESTIMATORS.values()],
        )
        .sort("borough_id", "indicator")
    )
    _log_precision(table, len(ids))
    return table


def _check_counts(totals: pl.DataFrame, profile: BoroughProfileConfig) -> None:
    """Every count the profile reads, summed over the boroughs, comes to the state's own
    estimate: a borough missing or misread would not."""
    values = totals.filter(pl.col("ESTIMADOR") == "Valor")
    state = values.filter(pl.col("CVE_MUN") == STATE)
    boroughs = values.filter(pl.col("CVE_MUN") != STATE)
    for column, indicator in profile.indicators.items():
        if not indicator.count:
            continue
        stated = float(state.select(pl.col(column).sum()).item())
        summed = float(boroughs.select(pl.col(column).sum()).item())
        if abs(stated - summed) > 0.5:  # the file prints whole numbers
            raise ValueError(
                f"{profile.source}: the boroughs' {indicator.name} add up to {summed:,.0f}, "
                f"the state's estimate says {stated:,.0f}: a row was misread or is missing"
            )


def _log_precision(table: pl.DataFrame, boroughs: int) -> None:
    """What the survey can and cannot say here: INEGI calls an estimate precise below a
    coefficient of variation of 15%, acceptable to 30%, and to be used with care above."""
    named = table["borough_id"].n_unique()
    if named < boroughs:
        logger.warning("borough_profile: the survey names %d of %d boroughs", named, boroughs)
    if table.is_empty():
        return
    worst = table.sort("cv", descending=True, nulls_last=True).head(1)
    loose = table.filter(pl.col("cv") > 15)
    logger.info(
        "borough_profile: %d boroughs x %d indicators; %d estimates above a 15%% coefficient "
        "of variation, the loosest %s in %s (%.1f%%)",
        named,
        table["indicator"].n_unique(),
        loose.height,
        worst["indicator"].item(),
        worst["borough"].item(),
        worst["cv"].item(),
    )
