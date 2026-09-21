"""Analysis pipeline: turn the layers into tables somebody can read and act on.

A fourth pipeline, independent of the other three: it consumes what they leave on disk
and never feeds them back automatically. Its output is evidence — including a per-feature
recommendation used to decide what the model should look at next — and evidence is
reviewed by a person before it changes a config.

Every table is written as Parquet (what the catalog and the dashboard read) and as CSV
(what a human opens in a spreadsheet).
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from sklearn.inspection import permutation_importance

from mlops_core.analysis.figures import render_all
from mlops_core.analysis.studies import (
    categorical_profile,
    feature_recommendation,
    market_history,
    market_summary,
    numeric_profile,
    residuals_by_group,
    target_distribution,
)
from mlops_core.config import DomainConfig
from mlops_core.ml.registry import load_champion
from mlops_core.ml.train import temporal_split, xy
from mlops_core.storage import latest_partition, new_partition, read_table, write_table

logger = logging.getLogger(__name__)

REVIEWS = "coffee_reviews"
CONTEXT = "market_context"
FEATURES = "review_features"
PREDICTIONS = "review_predictions"
FIGURES = "figures"


@dataclass(frozen=True)
class AnalysisOutput:
    """Where this run left its evidence."""

    tables: dict[str, Path] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    published: list[Path] = field(default_factory=list)


def champion_importance(
    config: DomainConfig, data_dir: Path, tracking_uri: str
) -> pl.DataFrame | None:
    """How much the champion's error grows when each feature is shuffled.

    Permutation importance on the test split, not the split counts LightGBM reports: a
    tree can spend half its splits on a feature that carries no signal, which is exactly
    what the market-context features looked like until this was measured.
    """
    try:
        served = load_champion(
            config.training.registered_model, tracking_uri, data_dir / "model_cache"
        )
    except Exception as unavailable:  # nothing trained yet, or the registry is down
        logger.warning("Skipping permutation importance: %s", unavailable)
        return None
    _, test = temporal_split(read_table(data_dir / "features" / FEATURES), config.training)
    x_test, y_test = xy(test, config.model)
    result = permutation_importance(
        served.model,
        x_test,
        y_test,
        n_repeats=config.analysis.permutation_repeats,
        random_state=config.training.seed,
        scoring="neg_mean_absolute_error",
    )
    return pl.DataFrame(
        {
            "feature": config.model.features,
            # Positive = shuffling it made the model worse, so the model relies on it.
            "permutation_importance": [float(v) for v in result.importances_mean],
            "permutation_importance_sd": [float(v) for v in result.importances_std],
        }
    )


def build_analysis(
    config: DomainConfig,
    data_dir: Path,
    tracking_uri: str,
    at: datetime | None = None,
    publish_to: Path | None = None,
) -> AnalysisOutput:
    """Compute every study from the latest layers, write it as Parquet and CSV, draw
    the figures, and copy the published selection where the docs can reference it."""
    analysis, spec = config.analysis, config.model
    reviews = read_table(data_dir / "clean" / REVIEWS)
    context = read_table(data_dir / "clean" / CONTEXT)
    features = read_table(data_dir / "features" / FEATURES)

    numeric = numeric_profile(features, spec, analysis.period_column)
    categorical = categorical_profile(features, spec, analysis.period_column, analysis.min_rows)
    importance = champion_importance(config, data_dir, tracking_uri)

    tables = {
        "target_distribution": target_distribution(reviews, spec.target, analysis.period_column),
        "numeric_profile": numeric,
        "categorical_profile": categorical,
        "feature_recommendation": feature_recommendation(numeric, categorical, importance),
        "market_summary": market_summary(context, analysis.market_year, analysis.top_countries),
        "market_history": market_history(
            context, analysis.spotlight_country, analysis.history_since
        ),
    }
    predictions = _latest_predictions(data_dir)
    if predictions is not None:
        tables["residuals"] = residuals_by_group(
            predictions,
            features,
            spec,
            config.training.stratify_by,
            analysis.min_rows,
            analysis.period_column,
        )
    else:
        logger.warning("No batch predictions yet: skipping the residual study")

    built_at = at or datetime.now(UTC)
    lineage = _lineage(data_dir)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        written[name] = write_table(
            table, data_dir / "analysis" / name, lineage, built_at, also_csv=True
        )
        if table.is_empty():
            # Usually a config asking for something the data does not have, such as a
            # market year that has not been published yet. Silence would hide it.
            logger.warning("%s came out empty: check the analysis config against the data", name)
        else:
            logger.info("%s: %d rows", name, table.height)

    figures = _write_figures(tables, config, data_dir, built_at)
    published = _publish(figures, config.analysis.published_figures, publish_to)
    return AnalysisOutput(written, figures, published)


def _write_figures(
    tables: dict[str, pl.DataFrame], config: DomainConfig, data_dir: Path, built_at: datetime
) -> dict[str, Path]:
    """One PNG per figure, in a partition beside the tables they were drawn from."""
    analysis = config.analysis
    drawn = render_all(
        tables,
        period=analysis.period_column,
        target=config.model.target,
        country=analysis.spotlight_country,
        year=analysis.market_year,
    )
    partition = new_partition(data_dir / "analysis" / FIGURES, "built_at", built_at)
    paths = {}
    for name, figure in drawn.items():
        paths[name] = partition / f"{name}.png"
        figure.savefig(paths[name], facecolor=figure.get_facecolor())
        plt.close(figure)  # figures hold memory until closed
    return paths


def _publish(figures: dict[str, Path], selection: list[str], publish_to: Path | None) -> list[Path]:
    """Copy the selected figures where they are committed and rendered in the docs.

    Only a curated few: the rest stay in `data/`, rebuildable and untracked.
    """
    if publish_to is None:
        return []
    publish_to.mkdir(parents=True, exist_ok=True)
    published = []
    for name in selection:
        if name not in figures:
            logger.warning("Figure '%s' was not drawn, so it was not published", name)
            continue
        destination = publish_to / f"{name}.png"
        shutil.copyfile(figures[name], destination)
        published.append(destination)
    return published


def _latest_predictions(data_dir: Path) -> pl.DataFrame | None:
    table_dir = data_dir / "predictions" / PREDICTIONS
    return read_table(table_dir) if latest_partition(table_dir) else None


def _lineage(data_dir: Path) -> dict[str, str]:
    """Which partition of each input this analysis was computed from."""
    sources = {
        REVIEWS: data_dir / "clean" / REVIEWS,
        CONTEXT: data_dir / "clean" / CONTEXT,
        FEATURES: data_dir / "features" / FEATURES,
        PREDICTIONS: data_dir / "predictions" / PREDICTIONS,
    }
    found = {name: latest_partition(path) for name, path in sources.items()}
    return {name: partition.name for name, partition in found.items() if partition}
