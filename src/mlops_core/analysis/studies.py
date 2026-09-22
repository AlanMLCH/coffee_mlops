"""The analyses themselves: pure functions from layers to tables.

Each one answers a question every domain gets asked, and the last one answers the
question that changes the model: which features are worth keeping. A domain's own
studies live with the domain. Nothing here does I/O, so every table can be tested on a
handful of rows.
"""

from typing import cast

import numpy as np
import polars as pl

from mlops_core.config import ModelSpec, TargetBands


def periods_in_order(frame: pl.DataFrame, period: str, time: str) -> list[str]:
    """Periods ordered by when they happened, never by name: "new" sorts before "old"."""
    ordered = frame.group_by(period).agg(pl.col(time).min().alias("start")).sort("start")
    return [str(name) for name in ordered[period].to_list()]


def target_distribution(items: pl.DataFrame, target: str, period: str, time: str) -> pl.DataFrame:
    """How the target is distributed in each period. The shape matters as much as the
    mean: a truncated period (nothing below some score) evaluates a model on a different problem."""
    order = {name: position for position, name in enumerate(periods_in_order(items, period, time))}
    return (
        items.group_by(period)
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


def numeric_profile(
    features: pl.DataFrame, spec: ModelSpec, period: str, time: str
) -> pl.DataFrame:
    """Per numeric feature: how often it is missing, how it moves with the target, and
    how far its distribution drifted between periods (in standard deviations). With a
    single period there is nothing to drift from: the drift is null, not zero."""
    periods = periods_in_order(features, period, time)
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
                    abs(_scalar(last.mean()) - _scalar(first.mean())) / spread
                    if spread and len(periods) > 1
                    else None
                ),
            }
        )
    # Typed up front: a column that is null for every feature (drift, with one period)
    # would otherwise be inferred as Null and refuse to join anything typed.
    schema = {
        "feature": pl.String,
        "missing_pct": pl.Float64,
        "mean": pl.Float64,
        "sd": pl.Float64,
        "correlation_with_target": pl.Float64,
        "drift_sd": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort(
        "correlation_with_target", descending=True, nulls_last=True
    )


def categorical_profile(
    features: pl.DataFrame, spec: ModelSpec, period: str, time: str, min_rows: int
) -> pl.DataFrame:
    """Per level of each categorical feature: its weight in each period and its mean
    target. A level that grows from 6% to 30% is a composition change, not drift. With a
    single period, first and last are the same one, and its rows are counted once."""
    periods = periods_in_order(features, period, time)
    first, last = periods[0], periods[-1]
    rows = pl.col("n_first") if first == last else pl.col("n_first") + pl.col("n_last")
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
            .with_columns(pl.col(*dict.fromkeys([n_first, n_last])).fill_null(0))
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
            .filter(rows >= min_rows)
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
    item_id: str,
    bands: TargetBands,
) -> pl.DataFrame:
    """Where the model is wrong: by a categorical feature and by target band, **per
    period**.

    Split by period on purpose. The batch job scores every row it has, training rows
    included, and a model is always flattering on the data it learned from; mixing the
    two would report an error nobody will ever see in production.
    """
    scored = predictions.join(features.drop(period), on=item_id, how="inner").with_columns(
        (pl.col("prediction") - pl.col(spec.target)).alias("error")
    )
    by_group = _error_summary(scored, group, "group", period).rename({group: "level"})
    banded = scored.with_columns(_band(spec.target, bands).alias("quality_band"))
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
    numbers looked damning (context features: no correlation, half the splits) removing the
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


def _correlation(frame: pl.DataFrame, column: str, target: str) -> float | None:
    pairs = frame.select(column, target).drop_nulls()
    if pairs.height < 3 or (pairs[column].std() or 0) == 0:
        return None  # a constant column has no correlation to report
    return float(np.corrcoef(pairs[column].to_numpy(), pairs[target].to_numpy())[0, 1])


def _band(target: str, bands: TargetBands) -> pl.Expr:
    """Label each row with its band of the target. Left-closed, so a value on an edge
    belongs to the band above it, as the config promises."""
    return pl.col(target).cut(bands.edges, labels=bands.labels, left_closed=True).cast(pl.String)


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
