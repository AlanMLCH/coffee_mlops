"""What households spend on coffee to drink at home: INEGI's household income and
expenditure survey (ENIGH), read for its coffee.

The clean table keeps a row per household of the sample - who it stands for (its weight),
where it is, the survey's design (stratum and primary sampling unit), its size and income
- with what it paid for each of the domain's coffees in the quarter and what it took from
its own harvest. Every figure built from it is the survey's: weighted by the households
each one stands for, and with an interval from resampling the sampling units within their
strata (the design's clusters: households of one unit are neighbours, not independent).

Two cautions travel with every figure. The survey records food over one week and states
it per quarter, so "bought coffee" is "bought coffee in the week of the survey": a jar of
instant lasts longer, and the share of households that drink it is larger. And coffee drunk
in a coffee shop is a meal out, which the survey does not split by what was eaten: these
are the cups made at home.
"""

import numpy as np
import polars as pl

from domains.coffee.config import HouseholdSpendingConfig

KEYS = ["folioviv", "foliohog"]
PAID = "G1"  # paid for the household (G2 is paid for another one)
OWN_HARVEST = "G3"  # taken from what the household itself grows
NATIONAL = "Mexico"
RESAMPLES = 1000
SEED = 7
LEVEL = 0.95


def clean_household_coffee(
    spending: pl.DataFrame, households: pl.DataFrame, survey: HouseholdSpendingConfig
) -> pl.DataFrame:
    """A row per household of the survey: who it is, and what it spent on coffee in the
    quarter - paid for each of the domain's coffees, and taken from its own harvest.

    Raises if a purchase belongs to no household, or to one in another state, or if a paid
    purchase has no value: either file would then be another edition than the other."""
    orphans = spending.join(households, on=KEYS, how="anti").height
    if orphans:
        raise ValueError(f"{orphans} coffee purchases belong to no household of the survey")
    placed = spending.join(households.select(*KEYS, "ubica_geo"), on=KEYS)
    moved = placed.filter(pl.col("entidad") != pl.col("ubica_geo").str.slice(0, 2)).height
    if moved:
        raise ValueError(f"{moved} coffee purchases name another state than their household")
    paid = spending.filter(pl.col("tipo_gasto") == PAID)
    blank = paid.filter(pl.col("gasto_tri").is_null()).height
    if blank:
        raise ValueError(f"{blank} paid coffee purchases have no value")
    unknown = set(households["ubica_geo"].str.slice(0, 2)) - set(survey.states)
    if unknown:
        raise ValueError(f"Households in states {sorted(unknown)}, not in the config's states")
    bought = paid.group_by(KEYS).agg(
        pl.col("gasto_tri").filter(pl.col("clave") == code).sum().alias(f"{name}_quarter_mxn")
        for code, name in survey.products.items()
    )
    grown = (
        spending.filter(pl.col("tipo_gasto") == OWN_HARVEST)
        .group_by(KEYS)
        .agg(pl.col("gas_nm_tri").fill_null(0).sum().alias("own_harvest_quarter_mxn"))
    )
    money = [*(f"{name}_quarter_mxn" for name in survey.products.values()),
             "own_harvest_quarter_mxn"]  # fmt: skip
    table = (
        households.join(bought, on=KEYS, how="left")
        .join(grown, on=KEYS, how="left")
        .select(
            pl.lit(survey.year, pl.Int64).alias("year"),
            pl.concat_str("folioviv", pl.lit("-"), "foliohog").alias("household_id"),
            pl.col("ubica_geo").str.slice(0, 2).alias("state_id"),
            pl.col("ubica_geo").str.slice(0, 2).replace_strict(survey.states).alias("state"),
            pl.col("ubica_geo").alias("municipality_id"),
            pl.col("est_dis").alias("stratum"),
            pl.col("upm").alias("psu"),
            pl.col("factor").alias("weight"),
            pl.col("tot_integ").alias("members"),
            pl.col("ing_cor").alias("income_quarter_mxn"),
            *(pl.col(column).fill_null(0.0) for column in money),
        )
    )
    return with_deciles(table).sort("household_id")


def with_deciles(table: pl.DataFrame) -> pl.DataFrame:
    """Each household's income decile: households ordered by income, then cut where each
    tenth of the households they stand for (by weight, as INEGI cuts them) ends."""
    ordered = table.sort("income_quarter_mxn", "household_id")
    share = pl.col("weight").cum_sum() / pl.col("weight").sum()
    decile = (share * 10).ceil().clip(1, 10).cast(pl.Int64)
    return ordered.with_columns(decile.alias("income_decile"))


def coffee_columns(survey: HouseholdSpendingConfig) -> list[str]:
    """The columns of what a household paid for each coffee."""
    return [f"{name}_quarter_mxn" for name in survey.products.values()]


def household_coffee_by_state(table: pl.DataFrame, survey: HouseholdSpendingConfig) -> pl.DataFrame:
    """Each state's households, and the country's: how many bought coffee to drink at home
    in the survey's week, each coffee's share, what a household spends a month, what one
    that bought spends, the share of its income, and how many drank their own harvest."""
    rows = [_summary(table, survey) | {"state_id": None, "state": NATIONAL}]
    for (state_id, state), group in table.group_by("state_id", "state", maintain_order=True):
        rows.append(_summary(group, survey) | {"state_id": state_id, "state": state})
    return pl.DataFrame(rows).select("state_id", "state", pl.exclude("state_id", "state"))


def household_coffee_by_decile(
    table: pl.DataFrame, survey: HouseholdSpendingConfig
) -> pl.DataFrame:
    """The same figures by income decile, for the country and the city: whether richer
    households buy coffee more often, spend more on it, and give it more of their income."""
    city = survey.states[survey.city]
    rows = []
    for area, households in ((NATIONAL, table), (city, table.filter(state_id=survey.city))):
        for (decile,), group in households.group_by("income_decile"):
            rows.append(_summary(group, survey) | {"area": area, "income_decile": decile})
    frame = pl.DataFrame(rows).select("area", "income_decile", pl.exclude("area", "income_decile"))
    return frame.sort("area", "income_decile", descending=[True, False])


def _summary(group: pl.DataFrame, survey: HouseholdSpendingConfig) -> dict[str, object]:
    columns = coffee_columns(survey)
    spent = pl.sum_horizontal(columns)
    group = group.with_columns(spent.alias("coffee"), (spent > 0).cast(pl.Float64).alias("bought"))
    w = pl.col("weight")
    totals = group.select(
        households=w.sum(),
        bought=(w * pl.col("bought")).sum(),
        coffee=(w * pl.col("coffee")).sum(),
        income=(w * pl.col("income_quarter_mxn")).sum(),
        own=(w * (pl.col("own_harvest_quarter_mxn") > 0)).sum(),
        **{column: (w * (pl.col(column) > 0)).sum() for column in columns},
    ).row(0, named=True)
    share_low, share_high = survey_interval(group, "bought")
    month_low, month_high = survey_interval(group, "coffee")
    households, bought, income = totals["households"], totals["bought"], totals["income"]
    return {
        "households_sampled": group.height,
        "households": households,
        "bought_share": bought / households,
        "bought_share_low": share_low,
        "bought_share_high": share_high,
        **{
            f"{name}_share": totals[column] / households
            for name, column in zip(survey.products.values(), columns, strict=True)
        },
        "monthly_mxn": totals["coffee"] / households / 3,
        "monthly_mxn_low": month_low / 3,
        "monthly_mxn_high": month_high / 3,
        "monthly_per_buyer_mxn": totals["coffee"] / bought / 3 if bought else None,
        "income_per_mille": 1000 * totals["coffee"] / income if income else None,
        "own_harvest_share": totals["own"] / households,
    }


def survey_interval(group: pl.DataFrame, value: str) -> tuple[float, float]:
    """The interval of a weighted mean per household, resampling the survey's primary
    sampling units with replacement within each stratum (each stratum keeps its number of
    units): neighbours' answers move together, and households are not independent draws."""
    units = (
        group.group_by("stratum", "psu")
        .agg((pl.col("weight") * pl.col(value)).sum().alias("y"), pl.col("weight").sum().alias("w"))
        .sort("stratum", "psu")
    )
    sizes = units.group_by("stratum", maintain_order=True).len()["len"].to_numpy().astype(np.int64)
    starts = np.repeat(np.cumsum(sizes) - sizes, sizes)
    counts = np.repeat(sizes, sizes)
    draws = np.random.default_rng(SEED).random((RESAMPLES, units.height))
    picked = starts + (draws * counts).astype(np.int64)
    y, w = units["y"].to_numpy()[picked].sum(axis=1), units["w"].to_numpy()[picked].sum(axis=1)
    means = y / w
    tail = 100 * (1 - LEVEL) / 2
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
