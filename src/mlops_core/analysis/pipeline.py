"""Analysis pipeline: turn the layers into tables somebody can read and act on.

A fourth pipeline, independent of the other three: it consumes what they leave on disk
and never feeds them back automatically. Its output is evidence — including a per-feature
recommendation used to decide what the model should look at next — and evidence is
reviewed by a person before it changes a config.

Every table is written as Parquet (what the catalog and the dashboard read) and as CSV
(what a human opens in a spreadsheet). The core's studies run once per model and are
named after it (`review_residuals`); the domain's own studies run once.
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance

from mlops_core.adapter import DomainAdapter
from mlops_core.analysis.figures import render_all
from mlops_core.analysis.studies import (
    categorical_profile,
    feature_recommendation,
    numeric_profile,
    residuals_by_group,
    target_distribution,
)
from mlops_core.config import DomainConfig, ModelConfig
from mlops_core.ml.registry import load_champion
from mlops_core.ml.train import split_items, xy
from mlops_core.storage import latest_partition, new_partition, read_table, write_table

logger = logging.getLogger(__name__)

FIGURES = "figures"


@dataclass(frozen=True)
class AnalysisOutput:
    """Where this run left its evidence."""

    tables: dict[str, Path] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    published: list[Path] = field(default_factory=list)


def champion_importance(
    config: DomainConfig, model: ModelConfig, data_dir: Path, tracking_uri: str
) -> pl.DataFrame | None:
    """How much the champion's error grows when each feature is shuffled.

    Permutation importance on the test split, not the split counts LightGBM reports: a
    tree can spend half its splits on a feature that carries no signal, which is exactly
    what an item's context features can look like until it is measured.
    """
    try:
        served = load_champion(
            model.training.registered_model, tracking_uri, data_dir / "model_cache"
        )
    except Exception as unavailable:  # nothing trained yet, or the registry is down
        logger.warning("Skipping permutation importance for %s: %s", model.name, unavailable)
        return None
    features = read_table(data_dir / "features" / model.features_table)
    _, test = split_items(features, model)
    x_test, y_test = xy(test, model.spec)
    result = permutation_importance(
        served.model,
        x_test,
        y_test,
        n_repeats=config.analysis.permutation_repeats,
        random_state=model.training.seed,
        scoring="neg_mean_absolute_error",
    )
    return pl.DataFrame(
        {
            "feature": model.spec.features,
            # Positive = shuffling it made the model worse, so the model relies on it.
            "permutation_importance": [float(v) for v in result.importances_mean],
            "permutation_importance_sd": [float(v) for v in result.importances_std],
        }
    )


def build_analysis(
    adapter: DomainAdapter,
    data_dir: Path,
    tracking_uri: str,
    at: datetime | None = None,
    publish_to: Path | None = None,
) -> AnalysisOutput:
    """Compute every study from the latest layers - each model's and the domain's - write
    each as Parquet and CSV, draw the figures, and copy the published selection where
    the docs can reference it."""
    config = adapter.config
    clean = {
        name: read_table(data_dir / "clean" / name)
        for name in adapter.clean_contracts()
        if latest_partition(data_dir / "clean" / name)
    }
    tables: dict[str, pl.DataFrame] = {}
    drawn: dict[str, Figure] = {}
    for model in config.models:
        if latest_partition(data_dir / "features" / model.features_table) is None:
            logger.warning("%s has no feature table yet: skipping its studies", model.name)
            continue
        studied = model_studies(config, model, clean, data_dir, tracking_uri)
        figures = render_all(
            studied,
            period=model.items.period,
            target=model.spec.target,
            band_labels=model.target_bands.labels,
        )
        tables |= {f"{model.name}_{name}": table for name, table in studied.items()}
        drawn |= {f"{model.name}_{name}": figure for name, figure in figures.items()}
    domain = dict(adapter.studies(clean))
    tables |= domain
    drawn |= dict(adapter.figures(domain))

    built_at = at or datetime.now(UTC)
    lineage = _lineage(data_dir, adapter)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        written[name] = write_table(
            table, data_dir / "analysis" / name, lineage, built_at, also_csv=True
        )
        if table.is_empty():
            # Usually a config asking for something the data does not have, such as a
            # year that has not been published yet. Silence would hide it.
            logger.warning("%s came out empty: check the analysis config against the data", name)
        else:
            logger.info("%s: %d rows", name, table.height)

    saved = _save_figures(drawn, data_dir, built_at)
    published = _publish(saved, config.analysis.published_figures, publish_to)
    return AnalysisOutput(written, saved, published)


def model_studies(
    config: DomainConfig,
    model: ModelConfig,
    clean: dict[str, pl.DataFrame],
    data_dir: Path,
    tracking_uri: str,
) -> dict[str, pl.DataFrame]:
    """The studies every model gets: its target, its features, and its champion's errors."""
    analysis, spec, items = config.analysis, model.spec, model.items
    features = read_table(data_dir / "features" / model.features_table)
    numeric = numeric_profile(features, spec, items.period, items.time)
    categorical = categorical_profile(features, spec, items.period, items.time, analysis.min_rows)
    importance = champion_importance(config, model, data_dir, tracking_uri)
    tables = {
        "target_distribution": target_distribution(
            clean[items.table], spec.target, items.period, items.time
        ),
        "numeric_profile": numeric,
        "categorical_profile": categorical,
        "feature_recommendation": feature_recommendation(numeric, categorical, importance),
    }
    predictions = _latest_predictions(data_dir, model)
    if predictions is None:
        logger.warning("No %s predictions yet: skipping its residual study", model.name)
        return tables
    tables["residuals"] = residuals_by_group(
        predictions,
        features,
        spec,
        model.training.stratify_by,
        analysis.min_rows,
        items.period,
        items.id,
        model.target_bands,
    )
    return tables


def _save_figures(drawn: dict[str, Figure], data_dir: Path, built_at: datetime) -> dict[str, Path]:
    """One PNG per figure, in a partition beside the tables they were drawn from."""
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


def _latest_predictions(data_dir: Path, model: ModelConfig) -> pl.DataFrame | None:
    table_dir = data_dir / "predictions" / model.predictions_table
    return read_table(table_dir) if latest_partition(table_dir) else None


def _lineage(data_dir: Path, adapter: DomainAdapter) -> dict[str, str]:
    """Which partition of each input this analysis was computed from."""
    models = adapter.config.models
    sources = {
        **{name: data_dir / "clean" / name for name in adapter.clean_contracts()},
        **{m.features_table: data_dir / "features" / m.features_table for m in models},
        **{m.predictions_table: data_dir / "predictions" / m.predictions_table for m in models},
    }
    found = {name: latest_partition(path) for name, path in sources.items()}
    return {name: partition.name for name, partition in found.items() if partition}
