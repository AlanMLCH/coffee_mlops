"""What the analysis last wrote, as the explorer's Findings and Models tabs show it.

The tabs read the saved partitions instead of recomputing: what they show is the evidence
on disk, stamped with the partitions it came from, so it can never disagree with the
tables. The core computes the same studies for every model, named after it
(`review_residuals`); anything else in the analysis layer is the domain's own.
"""

import json
from pathlib import Path

from mlops_core.storage import MANIFEST_NAME, latest_partition

ANALYSIS, FIGURES = "analysis", "figures"
CORE_STUDIES = (
    "target_distribution",
    "categorical_profile",
    "feature_recommendation",
    "numeric_profile",
    "residuals",
)
CORE_FIGURES = ("target_distribution", "feature_importance", "numeric_signal", "residual_bias")


def figure(data_dir: Path, name: str) -> Path | None:
    """The newest drawing of a figure, or None if none was drawn."""
    partition = latest_partition(data_dir / ANALYSIS / FIGURES)
    if partition is None:
        return None
    path = partition / f"{name}.png"
    return path if path.is_file() else None


def lineage(data_dir: Path, model: str) -> dict[str, str]:
    """When a model's studies were built, and which partition of each input they read."""
    partition = latest_partition(data_dir / ANALYSIS / f"{model}_target_distribution")
    if partition is None:
        return {}
    manifest = json.loads((partition / MANIFEST_NAME).read_text(encoding="utf-8"))
    return {"built_at": str(manifest["built_at"]), **manifest["inputs"]}


def domain_studies(data_dir: Path, models: list[str]) -> list[str]:
    """The studies the domain added, whatever they are called: every built one that is
    neither a model's copy of the core's nor the figures."""
    root = data_dir / ANALYSIS
    if not root.is_dir():
        return []
    core = {f"{model}_{name}" for model in models for name in CORE_STUDIES} | {FIGURES}
    built = (path.name for path in root.iterdir() if latest_partition(path) is not None)
    return sorted(name for name in built if name not in core)


def domain_figures(data_dir: Path, models: list[str]) -> list[Path]:
    """The domain's figures in the newest drawing, the models' own left out."""
    partition = latest_partition(data_dir / ANALYSIS / FIGURES)
    if partition is None:
        return []
    core = {f"{model}_{name}" for model in models for name in CORE_FIGURES}
    return sorted(path for path in partition.glob("*.png") if path.stem not in core)
