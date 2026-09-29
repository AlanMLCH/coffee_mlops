"""Do the tasting notes a roaster writes help price its coffee?

`analysis.flavor_prices` found no flavour category that raises the price within a shop
and a size once the family-wise interval is honest (spice +12% [-0.3, +26], a proxy for
Yemen, Rwanda and Kenya). That is a question about each category on its own; this asks
the model's question: with the notes as features next to the sheet's origin, process,
variety and altitude, does the price model get better?

Measured as the gate measures a group model: every coffee predicted by a model fitted
without it (out-of-fold, all coffees), paired against the model without the notes, and
the bootstrap resamples coffees, not bags. Each feature set gets its own search with the
same budget: tuning one and reusing its parameters for the other would measure the
search as much as the features. The bar is the gate's, 95%.

The notes are one indicator per SCA category (floral, fruity, ...) plus how many notes a
coffee names, null for a coffee whose description names none: no note is "not said", not
"not floral". They are today's notes joined to every read of the offers; the reads are
days apart, so a description that changed in between is the leak this leaves open, and
the reason a promoted version would need the notes per read first.

Run with the services up:  uv run python experiments/flavor_features.py
"""

import logging
import re

import mlflow
import numpy as np
import polars as pl

from mlops_core.adapter import load_adapter
from mlops_core.config import GroupSplit, Settings
from mlops_core.ml.evaluation import absolute_errors
from mlops_core.ml.train import out_of_fold, split_items, tune
from mlops_core.provenance import experiment_name
from mlops_core.stats import compare
from mlops_core.storage import read_table

MODEL = "offer"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flavor-features")


def note_columns(flavors: pl.DataFrame, categories: list[str]) -> pl.DataFrame:
    """One row per coffee with notes: an indicator per category and the notes' count."""
    named = {c: "note_" + re.sub(r"\W+", "_", c) for c in categories}
    return (
        flavors.group_by("coffee_id")
        .agg(
            pl.col("note_en").n_unique().alias("notes_n"),
            *[(pl.col("category") == c).any().cast(pl.Float64).alias(n) for c, n in named.items()],
        )
        .sort("coffee_id")
    )


def main() -> None:
    adapter = load_adapter("coffee")
    config = adapter.config
    settings = Settings()
    model = config.model_named(MODEL)
    split = model.training.split
    assert isinstance(split, GroupSplit)
    data_dir = settings.data_dir / config.name
    features = read_table(data_dir / "features" / model.features_table)
    flavors = read_table(data_dir / "clean" / "roaster_flavors")
    categories = sorted(flavors["category"].unique())
    notes = note_columns(flavors, categories)
    widened = features.join(notes, on="coffee_id", how="left")
    added = [c for c in notes.columns if c != "coffee_id"]
    logger.info(
        "%d offers of %d coffees; %d coffees have notes (%d offers)",
        widened.height,
        widened["coffee_id"].n_unique(),
        notes.height,
        widened.filter(pl.col("notes_n").is_not_null()).height,
    )

    candidates = {
        "without notes": model,
        "with notes": model.model_copy(
            update={
                "spec": model.spec.model_copy(update={"numeric": [*model.spec.numeric, *added]})
            }
        ),
    }
    cfg = model.training
    families = widened.sort(model.items.id)[split.column].to_numpy()
    errors: dict[str, np.ndarray] = {}
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name(config, model))
    with mlflow.start_run(run_name="experiment-flavor-features"):
        mlflow.set_tags({"experiment": "flavor_features", "pipeline": "none"})
        for name, candidate in candidates.items():
            train, _ = split_items(widened, candidate)  # tuning never sees the test coffees
            params, cv_mae = tune(train, candidate)
            observed, predicted, _ = out_of_fold(widened, candidate, params)
            errors[name] = absolute_errors(observed, predicted)
            slug = name.replace(" ", "_")
            mlflow.log_metrics({f"cv_mae_{slug}": cv_mae, f"oof_mae_{slug}": errors[name].mean()})
            logger.info(
                "%-14s CV MAE %6.1f  out-of-fold MAE %6.1f  (%s)",
                name,
                cv_mae,
                errors[name].mean(),
                ", ".join(f"{k} {v}" for k, v in sorted(params.items())),
            )
        verdict = compare(
            errors["with notes"],
            errors["without notes"],
            cfg.bootstrap_resamples,
            cfg.seed,
            families,
        )
        mlflow.log_metrics(verdict.as_metrics("with_notes_versus_without"))
        has_notes = widened.sort(model.items.id)["notes_n"].is_not_null().to_numpy()
        for label, rows in (("with notes", has_notes), ("without notes", ~has_notes)):
            logger.info(
                "  on the %d offers of coffees %s: %.1f with the notes, %.1f without",
                rows.sum(),
                label,
                errors["with notes"][rows].mean(),
                errors["without notes"][rows].mean(),
            )
        logger.info(
            "\nwith notes vs without, out of fold: %+.1f MAE (95%% CI %+.1f to %+.1f), "
            "%.0f%% sure it is better; the gate asks for %.0f%%",
            verdict.difference,
            verdict.ci_low,
            verdict.ci_high,
            100 * verdict.probability_better,
            100 * cfg.min_probability_better,
        )


if __name__ == "__main__":
    main()
