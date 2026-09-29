"""Which code, and which rows, produced a result.

MLflow records the entry point but not the revision when the pipeline runs from a CLI,
so without this a run cannot be traced back to the code that made it. `dirty` matters as
much as the hash: a run made from uncommitted changes is not reproducible, and saying so
is better than implying otherwise.

A training run is also tagged with the rows it learned from (`DATA_VERSION`, the content
hash `storage.rows_version`), so anyone - the trainer, the monitor, the orchestrator -
can ask whether a model was already trained on the rows in front of it.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mlops_core.config import DomainConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_VERSION = "data_version"  # the run tag: which rows the model learned from


@dataclass(frozen=True)
class CodeVersion:
    commit: str
    dirty: bool

    def as_tags(self) -> dict[str, str]:
        return {"git_commit": self.commit, "git_dirty": str(self.dirty)}


def _git(repo: Path, *args: str) -> str | None:
    """Run a git command, or return None where git cannot answer (no git, no checkout,
    a container built from a tarball). Provenance is nice to have, never a hard failure."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version(repo: Path = REPO_ROOT) -> CodeVersion | None:
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None
    return CodeVersion(commit=commit, dirty=bool(_git(repo, "status", "--porcelain")))


def experiment_name(config: DomainConfig, model: ModelConfig) -> str:
    """One MLflow experiment per model: runs of different targets are not comparable."""
    return f"{config.name}-{model.name}"


def trained_on(config: DomainConfig, model_name: str, version: str) -> str | None:
    """The id of a training run of this model on these rows, if there is one. Needs the
    tracking URI set: the runs are the record."""
    import mlflow  # only a caller that asks needs it; the base imports no client
    from mlflow import MlflowClient

    experiment = mlflow.get_experiment_by_name(
        experiment_name(config, config.model_named(model_name))
    )
    if experiment is None:
        return None
    runs = MlflowClient().search_runs(
        [experiment.experiment_id], f"tags.{DATA_VERSION} = '{version}'", max_results=1
    )
    return runs[0].info.run_id if runs else None
