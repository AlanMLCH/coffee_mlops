"""The analyses themselves: pure functions from layers to tables.

Each one answers a question someone actually asks about this domain, and the last one
answers the question that changes the model: which features are worth keeping. Nothing
here does I/O, so every table can be tested on a handful of rows.
"""

from typing import cast

import numpy as np
import polars as pl

from coffee_mlops.config import ModelSpec

# Quality bands used when reporting error: a buyer cares about these ranges, not deciles.
QUALITY_BANDS = [
    (0.0, 82.0, "low (<82)"),
    (82.0, 85.0, "mid (82-85)"),
    (85.0, 101.0, "high (>=85)"),
]


# Periods are ordered by when they happened, never by name: "new" sorts before "old".
TIME_COLUMN = "grading_date"


def periods_in_order(frame: pl.DataFrame, period: str) -> list[str]:
    ordered = frame.group_by(period).agg(pl.col(TIME_COLUMN).min().alias("start")).sort("start")
    return [str(name) for name in ordered[period].to_list()]


def target_distribution(reviews: pl.DataFrame, target: str, period: str) -> pl.DataFrame:
    """How the target is distributed in each period. The shape matters as much as the
    mean: a truncated period (no bad coffee) evaluates a model on a different problem."""
    order = {name: position for position, name in enumerate(periods_in_order(reviews, period))}
    return (
        reviews.group_by(period)
        .agg(
            pl.len().alias("n"),
            pl.col(target).mean().alias("mean"),
            pl.col(target).std().alias("sd"),
            pl.col(target).min().alias("min"),
            pl.col(target).quantile(0.25).alias("q25"),
            pl.col(target).median().alias("median"),
            pl.col(target).quantile(0.75).alias("q75"),
            pl.col(target).max().alias("max"),
        )
        .sort(pl.col(period).replace_strict(order, return_dtype=pl.Int32))
    )


def numeric_profile(features: pl.DataFrame, spec: ModelSpec, period: str) -> pl.DataFrame:
    """Per numeric feature: how often it is missing, how it moves with the target, and
    how far its distribution drifted between periods (in standard deviations)."""
    periods = periods_in_order(features, period)
    rows = []
    for column in spec.numeric:
        values = features[column]
        by_period = {
            name: features.filter(pl.col(period) == name)[column].drop_nulls() for name in periods
        }
        first, last = by_period[periods[0]], by_period[periods[-1]]
        spread = _scalar(first.std())
        rows.append(
            {
                "feature": column,
                "missing_pct": 100.0 * values.null_count() / features.height,
                "mean": values.mean(),
                "sd": values.std(),
                "correlation_with_target": _correlation(features, column, spec.target),
                # Standardised difference of means: 0.5 already means a different population.
                "drift_sd": (
                    abs(_scalar(last.mean()) - _scalar(first.mean())) / spread if spread else None
                ),
            }
        )
    return pl.DataFrame(rows).sort("correlation_with_target", descending=True, nulls_last=True)


def categorical_profile(
    features: pl.DataFrame, spec: ModelSpec, period: str, min_rows: int
) -> pl.DataFrame:
    """Per level of each categorical feature: its weight in each period and its mean
    target. A level that grows from 6% to 30% is a composition change, not drift."""
    periods = periods_in_order(features, period)
    first, last = periods[0], periods[-1]
    frames = []
    for column in spec.categorical:
        counts = (
            features.group_by(column, period)
            .agg(pl.len().alias("n"), pl.col(spec.target).mean().alias("mean_target"))
            .pivot(on=period, index=column, values=["n", "mean_target"])
        )
        n_first, n_last = f"n_{first}", f"n_{last}"
        frames.append(
            counts.rename({column: "level"})
            .with_columns(pl.lit(column).alias("feature"))
            .with_columns(
                pl.col(n_first).fill_null(0),
                pl.col(n_last).fill_null(0),
            )
            .with_columns(
                (pl.col(n_first) / pl.col(n_first).sum()).alias("share_first"),
                (pl.col(n_last) / pl.col(n_last).sum()).alias("share_last"),
            )
            .select(
                "feature",
                "level",
                pl.col(n_first).alias("n_first"),
                pl.col(n_last).alias("n_last"),
                "share_first",
                "share_last",
                pl.col(f"mean_target_{first}").alias("mean_target_first"),
                pl.col(f"mean_target_{last}").alias("mean_target_last"),
            )
            .filter((pl.col("n_first") + pl.col("n_last")) >= min_rows)
        )
    return (
        pl.concat(frames)
        .with_columns((pl.col("share_last") - pl.col("share_first")).alias("share_change"))
        .sort("share_change", descending=True)
    )


def residuals_by_group(
    predictions: pl.DataFrame,
    features: pl.DataFrame,
    spec: ModelSpec,
    group: str,
    min_rows: int,
    period: str,
) -> pl.DataFrame:
    """Where the model is wrong: by a categorical feature and by quality band, **per
    period**.

    Split by period on purpose. The batch job scores every row it has, training rows
    included, and a model is always flattering on the data it learned from; mixing the
    two would report an error nobody will ever see in production.
    """
    scored = predictions.join(features.drop(period), on="review_id", how="inner").with_columns(
        (pl.col("prediction") - pl.col(spec.target)).alias("error")
    )
    by_group = _error_summary(scored, group, "group", period).rename({group: "level"})
    banded = scored.with_columns(_band(spec.target).alias("quality_band"))
    by_band = _error_summary(banded, "quality_band", "quality_band", period).rename(
        {"quality_band": "level"}
    )
    return (
        pl.concat([by_group, by_band])
        .filter(pl.col("n") >= min_rows)
        .sort(period, "kind", "mae", descending=[False, False, True])
    )


def feature_recommendation(
    numeric: pl.DataFrame, categorical: pl.DataFrame, importance: pl.DataFrame | None
) -> pl.DataFrame:
    """One row per feature with the evidence needed to keep, review or drop it.

    The action is deliberately a suggestion, not an automatic change: the last time the
    numbers looked damning (market context: no correlation, half the splits) removing the
    features made the model worse. Evidence informs the decision, it does not make it.
    """
    numeric_rows = numeric.select(
        "feature",
        pl.lit("numeric").alias("kind"),
        "missing_pct",
        pl.col("correlation_with_target").abs().alias("signal"),
        "drift_sd",
    )
    categorical_rows = (
        categorical.group_by("feature")
        .agg(
            pl.lit("categorical").first().alias("kind"),
            pl.lit(None, pl.Float64).first().alias("missing_pct"),
            # Spread of the level means: how much knowing the level moves the target.
            pl.col("mean_target_first").std().alias("signal"),
            pl.col("share_change").abs().max().alias("drift_sd"),
        )
        .select("feature", "kind", "missing_pct", "signal", "drift_sd")
    )
    table = pl.concat([numeric_rows, categorical_rows])
    if importance is not None:
        table = table.join(importance, on="feature", how="left")
    else:
        table = table.with_columns(pl.lit(None, pl.Float64).alias("permutation_importance"))
    return table.with_columns(_suggested_action()).sort(
        "permutation_importance", descending=True, nulls_last=True
    )


def market_summary(context: pl.DataFrame, year: int, top: int) -> pl.DataFrame:
    """Who produces the world's coffee in one market year, and what they do with it."""
    producing = context.filter((pl.col("market_year") == year) & (pl.col("production") > 0))
    return (
        producing.select(
            "country",
            "production",
            (100 * pl.col("production") / pl.col("production").sum()).alias("world_share_pct"),
            (pl.col("exports") / pl.col("production")).alias("export_ratio"),
            "domestic_consumption",
            (pl.col("imports") / pl.col("domestic_consumption")).alias("imported_share_of_use"),
        )
        .sort("production", descending=True)
        .head(top)
    )


def market_history(context: pl.DataFrame, country: str, since: int) -> pl.DataFrame:
    """One country through time: production, what it exports, and what it imports to
    drink. For Mexico those three lines are the whole story of the domestic market."""
    return (
        context.filter((pl.col("country") == country) & (pl.col("market_year") >= since))
        .select(
            "market_year",
            "production",
            "exports",
            "domestic_consumption",
            "imports",
            (100 * pl.col("imports") / pl.col("domestic_consumption")).alias(
                "imported_share_of_use_pct"
            ),
        )
        .sort("market_year")
    )


def _correlation(frame: pl.DataFrame, column: str, target: str) -> float | None:
    pairs = frame.select(column, target).drop_nulls()
    if pairs.height < 3 or (pairs[column].std() or 0) == 0:
        return None  # a constant column has no correlation to report
    return float(np.corrcoef(pairs[column].to_numpy(), pairs[target].to_numpy())[0, 1])


def _band(target: str) -> pl.Expr:
    """Label each row with its quality band. Written out rather than folded in a loop,
    because polars' when/then chain changes type at every link."""
    score = pl.col(target)
    (_, low_edge, low_label), (_, mid_edge, mid_label), (*_, high_label) = QUALITY_BANDS
    return (
        pl.when(score < low_edge)
        .then(pl.lit(low_label))
        .when(score < mid_edge)
        .then(pl.lit(mid_label))
        .otherwise(pl.lit(high_label))
    )


def _scalar(value: object) -> float:
    """polars types an aggregate as any scalar (dates included); these are always
    numeric, and a missing aggregate means an empty group."""
    return 0.0 if value is None else cast(float, value)


def _error_summary(scored: pl.DataFrame, column: str, kind: str, period: str) -> pl.DataFrame:
    return (
        scored.group_by(period, column)
        .agg(
            pl.lit(kind).first().alias("kind"),
            pl.len().alias("n"),
            pl.col("error").abs().mean().alias("mae"),
            pl.col("error").mean().alias("bias"),
        )
        .select(period, column, "kind", "n", "mae", "bias")
    )


def _suggested_action() -> pl.Expr:
    """A label, not a decision: it says what to look at, never what to remove."""
    return (
        pl.when(pl.col("missing_pct") > 50)
        .then(pl.lit("review: mostly missing"))
        .when(pl.col("drift_sd") > 0.5)
        .then(pl.lit("review: distribution moved"))
        .when(pl.col("signal") < 0.05)
        .then(pl.lit("review: little signal on its own"))
        .otherwise(pl.lit("keep"))
        .alias("suggested_action")
    )
