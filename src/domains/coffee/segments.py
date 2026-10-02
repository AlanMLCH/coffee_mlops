"""Three studies of who passes prices on and who sets them (v1.1.1): studies, not served
models - they describe what is there, and no gate has a rule to hold them against.

- **Price transmission**: when green coffee moves 1%, how much of it reaches the shelf,
  and the farm gate, and after how many months.
- **Municipality types**: Mexico's coffee municipalities grouped by what twenty years of
  SIAP say about them - yield, the price paid for their cherry, how big they are, whether
  they are growing or leaving coffee, how much they lose.
- **Chain strategies**: PROFECO's chains placed by what they charge against everyone
  else for the same jar the same month, and how often they cut it.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 7
RESAMPLES = 1000
LEVEL = 0.95

# The shelf's steps and the green coffee each is made of: a jar of instant is mostly
# robusta, a bag of ground coffee in a supermarket mostly arabica.
SHELF_STEPS = {"instant": "robustas", "ground": "other_milds"}
MONTHLY_LAGS = 6
ANNUAL_LAGS = 1
BLOCK = {"monthly": 6, "annual": 3}  # a bootstrap block: months and years run together

TRANSMISSION = {
    "step": pl.String,
    "frequency": pl.String,
    "lag": pl.Int64,
    "pass_through": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "changes": pl.Int64,
}


def price_transmission(
    green: pl.DataFrame, shelf: pl.DataFrame, production: pl.DataFrame
) -> pl.DataFrame:
    """How much of a 1% move in green coffee each step's price has moved after each lag:
    the cumulative elasticity of a distributed lag of monthly (shelf) or yearly (farm gate)
    log changes, with an interval from a moving-block bootstrap.

    The shelf: PROFECO's median price per kilogram of plain instant and ground coffee, in
    pesos, against the World Bank's robustas and other milds in pesos (the exchange rate's
    own moves are part of what reaches a Mexican shelf). The farm gate: SIAP's national
    rural price of the cherry, against the year's mean other milds in pesos."""
    rows: list[dict[str, Any]] = []
    for product, indicator in SHELF_STEPS.items():
        plain = shelf.filter(pl.col("product") == product, ~pl.col("sweetened"), ~pl.col("decaf"))
        monthly = plain.group_by(pl.col("date").dt.truncate("1mo").alias("period")).agg(
            pl.col("price_mxn_per_kg").median().alias("price")
        )
        market = green.filter(pl.col("indicator") == indicator).select(
            "period", pl.col("mxn_per_kg").alias("green")
        )
        series = monthly.join(market, on="period", how="inner").sort("period")
        rows += _passed_on(f"{product} coffee on a shelf", "monthly", series, MONTHLY_LAGS)
    yearly_green = (
        green.filter(pl.col("indicator") == SHELF_STEPS["ground"])
        .group_by(pl.col("period").dt.year().cast(pl.Int64).alias("year"))
        .agg(pl.col("mxn_per_kg").mean().alias("green"), pl.len().alias("months"))
        .filter(pl.col("months") == 12)
    )
    cherry = (
        production.group_by("year")
        .agg((pl.col("value_mxn").sum() / pl.col("production_t").sum()).alias("price"))
        .join(yearly_green, on="year", how="inner")
        .sort("year")
    )
    rows += _passed_on("cherry at the farm gate", "annual", cherry, ANNUAL_LAGS)
    return pl.DataFrame(rows, schema=TRANSMISSION)


def _passed_on(step: str, frequency: str, series: pl.DataFrame, lags: int) -> list[dict[str, Any]]:
    changes = series.select(
        pl.col("price").log().diff().alias("y"), pl.col("green").log().diff().alias("x")
    )
    x, y = changes["x"].to_numpy(), changes["y"].to_numpy()
    design = np.column_stack([np.ones(len(x))] + [np.roll(x, lag) for lag in range(lags + 1)])
    usable = slice(lags + 1, None)  # the first change, and the lags before it, are unknown
    design, y = design[usable], y[usable]
    if len(y) <= design.shape[1] + 2:
        return []
    estimate = _cumulative(design, y)
    rng = np.random.default_rng(SEED)
    block = BLOCK[frequency]
    starts = np.arange(len(y) - block + 1)
    draws = []
    for _ in range(RESAMPLES):
        picked = rng.choice(starts, size=int(np.ceil(len(y) / block)))
        rows = np.concatenate([np.arange(s, s + block) for s in picked])[: len(y)]
        draws.append(_cumulative(design[rows], y[rows]))
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.nanpercentile(draws, tail, axis=0), np.nanpercentile(draws, 100 - tail, axis=0)
    return [
        {"step": step, "frequency": frequency, "lag": lag, "pass_through": float(estimate[lag]),
         "ci_low": float(low[lag]), "ci_high": float(high[lag]), "changes": len(y)}
        for lag in range(lags + 1)
    ]  # fmt: skip


def _cumulative(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The distributed lag's coefficients, summed lag by lag: after k periods, how much of
    a 1% move has arrived."""
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    cumulative: np.ndarray = np.cumsum(coefficients[1:])
    return cumulative


# What a municipality is, over the years it grew coffee.
TRAITS = {
    "yield_t_per_ha": "tonnes of cherry per harvested hectare, median of its years",
    "price_index": "its cherry's price against its state's that year, median (1 = the state's)",
    "harvested_ha": "harvested hectares, median of its years",
    "area_trend_pct": "change of its harvested area per year, % (log-linear)",
    "lost_pct": "share of its planted area lost, mean of its years, % (losses are rare years)",
}
PROFILES = {
    "cluster": pl.Int64,
    "type": pl.String,
    "municipalities": pl.Int64,
    **dict.fromkeys(TRAITS, pl.Float64),
    "k": pl.Int64,
    "silhouette": pl.Float64,
}
LABELS = {"yield_t_per_ha": "yield", "price_index": "price", "harvested_ha": "size",
          "area_trend_pct": "growth", "lost_pct": "losses"}  # fmt: skip
MIN_YEARS = 10  # a type needs a history, not a year
CLUSTER_KS = range(2, 7)


def municipality_types(production: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Each coffee municipality with at least `MIN_YEARS` years of harvest, its traits, and
    its type: k-means on the standardised traits (size on a log scale), k chosen by the
    mean silhouette between 2 and 6. A type is named by the traits that set it apart -
    above or below the municipalities' median by more than half a standard deviation.
    Returns (municipalities, types)."""
    harvested = production.filter(
        pl.col("harvested_ha") > 0, pl.col("production_t") > 0, pl.col("planted_ha") > 0
    )
    state_price = harvested.group_by("state_id", "year").agg(
        (pl.col("value_mxn").sum() / pl.col("production_t").sum()).alias("state_price")
    )
    yearly = harvested.join(state_price, on=["state_id", "year"], how="left").with_columns(
        (pl.col("rural_price_mxn_per_t") / pl.col("state_price")).alias("price_index"),
        (100 * pl.col("lost_ha").fill_null(0) / pl.col("planted_ha")).alias("lost_pct"),
    )
    traits = (
        yearly.group_by("municipality_id")
        .agg(
            pl.col("state", "municipality").last(),
            pl.len().alias("years"),
            pl.col("yield_t_per_ha").median(),
            pl.col("price_index").median(),
            pl.col("harvested_ha").median(),
            pl.col("lost_pct").mean(),
            _trend("harvested_ha").alias("area_trend_pct"),
        )
        .filter(pl.col("years") >= MIN_YEARS)
        .with_columns(pl.col(list(TRAITS)).fill_nan(None))
        .drop_nulls(list(TRAITS))
        .sort("municipality_id")
    )
    # Percentile ranks, not standard scores: a few municipalities with yields or losses
    # far from the rest would otherwise be clusters of their own, and every other one
    # "typical" (the first try: two clusters, of 484 and 2).
    if traits.height <= max(CLUSTER_KS):  # too few to group: said by an empty table
        typed = traits.with_columns(
            pl.lit(None, pl.Int64).alias("cluster"), pl.lit(None, pl.String).alias("type")
        )
        return typed.clear(), pl.DataFrame(schema=PROFILES)
    scaled = traits.select((pl.col(t).rank("average") / pl.len() - 0.5) for t in TRAITS).to_numpy()
    runs = []
    for k in CLUSTER_KS:
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(scaled)
        runs.append((float(silhouette_score(scaled, labels)), k, labels))
    silhouette, k, labels = max(runs, key=lambda run: run[0])
    typed = traits.with_columns(pl.Series("cluster", labels, dtype=pl.Int64))
    profiles = _profiles(typed, scaled, labels, silhouette, k)
    return typed.join(profiles.select("cluster", "type"), on="cluster"), profiles


def _trend(column: str) -> pl.Expr:
    """The yearly change of a quantity, % - the slope of its log on the year."""
    log, year = pl.col(column).log(), pl.col("year").cast(pl.Float64)
    slope = pl.cov(year, log) / year.var()
    return (slope.exp() - 1) * 100


def _profiles(
    typed: pl.DataFrame, scaled: np.ndarray, labels: np.ndarray, silhouette: float, k: int
) -> pl.DataFrame:
    rows = []
    for cluster in sorted(set(labels.tolist())):
        centre = scaled[labels == cluster].mean(axis=0)
        words = [
            f"{'high' if value > 0 else 'low'} {LABELS[name]}"
            for name, value in zip(TRAITS, centre, strict=True)
            if abs(value) > 0.15  # the centre sits beyond the 35th or 65th percentile
        ]
        members = typed.filter(pl.col("cluster") == cluster)
        rows.append(
            {"cluster": cluster, "type": ", ".join(words) or "typical",
             "municipalities": members.height,
             **{t: float(members[t].median()) for t in TRAITS},  # type: ignore[arg-type]
             "k": k, "silhouette": silhouette}
        )  # fmt: skip
    return pl.DataFrame(rows, schema=PROFILES).sort("municipalities", descending=True)


MIN_READINGS = 1000  # a chain judged on fewer readings is judged on a few stores
MIN_STORES = 5  # and an interval from resampling two stores is no interval
CUT = 0.9  # a reading this far under the store's own price that quarter is a cut
CHAINS = {
    "chain": pl.String,
    "store_type": pl.String,
    "readings": pl.Int64,
    "stores": pl.Int64,
    "price_index": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "cut_pct": pl.Float64,
}


def chain_strategies(shelf: pl.DataFrame) -> pl.DataFrame:
    """Each chain with enough readings: what it charges against every chain for the same
    product the same month - the mean ratio to the month's national median, 1.05 is 5%
    dearer, with an interval from resampling its stores - and how often it cuts: the share
    of its readings under 90% of what the same store asked for that product that quarter.

    A chain is *premium* or *discount* only when its interval leaves 1 out, *at the
    market* otherwise; *promotional* or *steady* by its cuts against the median chain's."""
    product = ["brand", "presentation"]
    month, quarter = pl.col("date").dt.truncate("1mo"), pl.col("date").dt.truncate("1q")
    priced = shelf.with_columns(
        (pl.col("price_mxn") / pl.col("price_mxn").median().over(*product, month)).alias("ratio"),
        (
            pl.col("price_mxn")
            < CUT * pl.col("price_mxn").median().over("store", *product, quarter)
        ).alias("cut"),
    )
    rows = []
    rng = np.random.default_rng(SEED)
    tail = (1 - LEVEL) / 2 * 100
    for (chain,), readings in priced.group_by("chain"):
        by_store = readings.group_by("store").agg(pl.col("ratio").sum(), pl.len().alias("n"))
        if readings.height < MIN_READINGS or by_store.height < MIN_STORES:
            continue
        sums, sizes = by_store["ratio"].to_numpy(), by_store["n"].to_numpy().astype(float)
        draws = rng.integers(0, len(sums), size=(RESAMPLES, len(sums)))
        means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        rows.append(
            {"chain": chain, "store_type": readings["store_type"].mode()[0],
             "readings": readings.height, "stores": by_store.height,
             "price_index": float(readings["ratio"].mean()),  # type: ignore[arg-type]
             "ci_low": float(np.percentile(means, tail)),
             "ci_high": float(np.percentile(means, 100 - tail)),
             "cut_pct": 100 * float(readings["cut"].cast(pl.Float64).sum()) / readings.height}
        )  # fmt: skip
    chains = pl.DataFrame(rows, schema=CHAINS)
    usual_cuts = chains["cut_pct"].median()
    level = (
        pl.when(pl.col("ci_low") > 1)
        .then(pl.lit("premium"))
        .when(pl.col("ci_high") < 1)
        .then(pl.lit("discount"))
        .otherwise(pl.lit("at the market"))
    )
    cuts = (
        pl.when(pl.col("cut_pct") > usual_cuts)
        .then(pl.lit("promotional"))
        .otherwise(pl.lit("steady"))
    )
    return chains.with_columns(pl.format("{}, {}", level, cuts).alias("strategy")).sort(
        "price_index", descending=True
    )
