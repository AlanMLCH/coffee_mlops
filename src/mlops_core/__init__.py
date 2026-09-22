"""The domain-agnostic core: extraction, contracts, layers, training, serving and
analysis. A domain plugs in through `mlops_core.adapter` and nothing here names it."""

import os

# MLflow prints a hint for AI coding assistants on every import; keep CLI output clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Artifact progress bars are written for humans at a terminal; they garble service logs.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
