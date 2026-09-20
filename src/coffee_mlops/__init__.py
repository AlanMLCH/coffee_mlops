"""Coffee MLOps: end-to-end ML/AI engineering platform, stage 1 (static sources)."""

import os

# MLflow prints a hint for AI coding assistants on every import; keep CLI output clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Artifact progress bars are written for humans at a terminal; they garble service logs.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
