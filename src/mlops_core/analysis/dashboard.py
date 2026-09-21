"""A small Streamlit dashboard over whatever the analysis pipeline last wrote.

It reads the saved partitions instead of recomputing: what you see is exactly the
evidence on disk, stamped with the partition it came from, and every table can be
downloaded as the same CSV the pipeline wrote. Nothing here computes a statistic, so
the dashboard can never disagree with the tables.

Run it with `make dashboard` (or `streamlit run src/mlops_core/analysis/dashboard.py`).
"""

import json
from pathlib import Path

import polars as pl
import streamlit as st

from mlops_core.config import Settings, load_domain_config
from mlops_core.storage import MANIFEST_NAME, latest_partition

ANALYSIS = "analysis"
FIGURES = "figures"


def analysis_dir(settings: Settings, domain: str) -> Path:
    return settings.data_dir / domain / ANALYSIS


def load(analysis: Path, name: str) -> pl.DataFrame | None:
    """The latest complete partition of one study, or None if it was never built."""
    partition = latest_partition(analysis / name)
    return pl.read_parquet(partition / f"{name}.parquet") if partition else None


def figure(analysis: Path, name: str) -> Path | None:
    partition = latest_partition(analysis / FIGURES)
    if partition is None:
        return None
    path = partition / f"{name}.png"
    return path if path.is_file() else None


def show(analysis: Path, name: str, caption: str, figure_name: str | None = None) -> None:
    """One study: its figure, its table, and the CSV behind both."""
    table = load(analysis, name)
    if table is None:
        st.info(f"`{name}` has not been built yet. Run `make analysis`.")
        return
    st.subheader(caption)
    image = figure(analysis, figure_name) if figure_name else None
    if image:
        st.image(str(image))
    st.dataframe(table, width="stretch")
    st.download_button(
        "Download CSV", table.write_csv(), file_name=f"{name}.csv", key=f"download-{name}"
    )


def lineage(analysis: Path) -> dict[str, str]:
    """Which partition of every input the studies on screen were computed from."""
    partition = latest_partition(analysis / "target_distribution")
    if partition is None:
        return {}
    manifest = json.loads((partition / MANIFEST_NAME).read_text())
    return {"built_at": str(manifest["built_at"]), **manifest["inputs"]}


def main() -> None:
    settings = Settings()
    config = load_domain_config(settings.domain)
    analysis = analysis_dir(settings, config.name)

    st.set_page_config(page_title=f"{config.name} analysis", layout="wide")
    st.title(f"{config.name}: analysis")
    stamp = lineage(analysis)
    if stamp:
        st.caption(" · ".join(f"{key}: {value}" for key, value in stamp.items()))
    else:
        st.warning("No analysis has been built yet. Run `make analysis` first.")
        return

    data_tab, features_tab, model_tab, market_tab = st.tabs(["Data", "Features", "Model", "Market"])
    with data_tab:
        show(analysis, "target_distribution", "The target, period by period", "target_distribution")
        show(analysis, "categorical_profile", "Which categories each period is made of")
    with features_tab:
        show(
            analysis,
            "feature_recommendation",
            "What to keep, and what to look at again",
            "feature_importance",
        )
        st.caption(
            "`suggested_action` is a prompt to look, never an instruction: features with "
            "no correlation of their own have already proved useful here."
        )
        show(analysis, "numeric_profile", "Numeric features in detail", "numeric_signal")
    with model_tab:
        residuals = load(analysis, "residuals")
        if residuals is None:
            st.info("No batch predictions yet. Run `make predict`.")
        else:
            image = figure(analysis, "residual_bias")
            if image:
                st.image(str(image))
            period = st.selectbox("Period", sorted(residuals[config.analysis.period_column]))
            st.dataframe(
                residuals.filter(pl.col(config.analysis.period_column) == period),
                width="stretch",
            )
            st.caption(
                "Error on the period the model trained on is not a forecast of anything; "
                "the newest period is the one to read."
            )
    with market_tab:
        show(analysis, "market_summary", "Who grows the world's coffee", "market_share")
        show(
            analysis,
            "market_history",
            f"{config.analysis.spotlight_country} through time",
            "market_history",
        )


if __name__ == "__main__":
    main()
