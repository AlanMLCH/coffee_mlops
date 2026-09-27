"""When the orchestrator should act on its own: pure functions its sensors call.

Two questions, each answered from what is on disk and in the registry:

- **Is there new data for a model?** Its clean tables (items and context) now come from
  raw data other than the data its feature table was last built from. A clean build
  writes new partitions whatever happened; the data version (`storage.data_version`)
  changes only when a download brought something new.
- **Is a retraining due, and not done?** The monitor's newest verdict for the model calls
  for one, on data no training run of the model has used yet. A source that stopped
  changing - a frozen snapshot - is then retrained once, not on every run.
"""

from pathlib import Path

from mlops_core.adapter import DomainAdapter
from mlops_core.config import DomainConfig
from mlops_core.ml.train import trained_on
from mlops_core.monitoring.drift import latest_verdict
from mlops_core.storage import built_from, latest_data_version


def new_data(adapter: DomainAdapter, data_dir: Path) -> dict[str, str]:
    """Each model whose clean inputs changed since its features were built, with the data
    version it should be scored and monitored on."""
    due = {}
    for model in adapter.config.models:
        tables = [model.items.table, *adapter.context_tables(model.name)]
        current = latest_data_version(data_dir, tables)
        if current is None:  # the clean layer is not built yet
            continue
        if current != built_from(data_dir, data_dir / "features" / model.features_table):
            due[model.name] = current
    return due


def retraining_due(config: DomainConfig, data_dir: Path) -> dict[str, str]:
    """Each model whose newest verdict calls for retraining on data not trained on yet,
    with that data version. Needs the tracking URI set: the runs are the record."""
    due = {}
    for model in config.models:
        verdict = latest_verdict(data_dir, model.name)
        if verdict is None or not verdict.retrain or not verdict.data_version:
            continue
        if trained_on(config, model.name, verdict.data_version) is None:
            due[model.name] = verdict.data_version
    return due
