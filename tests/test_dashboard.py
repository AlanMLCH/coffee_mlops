"""The dashboard is the one place a person looks at the evidence, so it gets a test.

Streamlit's AppTest runs the script the same way the server does, which catches the
failures that matter here: a renamed table, a missing figure, an API that moved.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeConfig
from mlops_core.analysis import pipeline
from mlops_core.analysis.pipeline import build_analysis
from mlops_core.data.clean import build_clean
from mlops_core.ml.features import build_features
from mlops_core.ml.registry import ServedModel
from tests.fakes import ConstantModel

DASHBOARD = Path(pipeline.__file__).with_name("dashboard.py")


@pytest.fixture
def built_analysis(
    coffee_config: CoffeeConfig, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A data dir with the layers and the studies already built."""
    market = coffee_config.market_analysis.model_copy(update={"market_year": 2022})
    analysis = coffee_config.analysis.model_copy(update={"min_rows": 1})
    config = CoffeeAdapter(
        coffee_config.model_copy(update={"analysis": analysis, "market_analysis": market})
    )
    monkeypatch.setattr(
        pipeline, "load_champion", lambda *a, **k: ServedModel(ConstantModel(), "1", "cache")
    )
    build_clean(config, raw_dir.parent)
    build_features(config, "review", raw_dir.parent)
    build_analysis(config, raw_dir.parent, "sqlite:///unused")
    return raw_dir.parent.parent


def run_dashboard(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("MLOPS_DATA_DIR", str(data_dir))
    app = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    app.run()
    return app


def test_the_dashboard_shows_the_studies_and_where_they_came_from(
    built_analysis: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run_dashboard(built_analysis, monkeypatch)

    assert not app.exception
    assert app.title[0].value == "coffee: analysis"
    # The stamp says which partitions are on screen: a dashboard without it invites
    # someone to read last week's numbers as today's.
    assert "review_features: built_at=" in app.caption[0].value
    # The last tab is the domain's: the core does not know its studies by name.
    assert [tab.label for tab in app.tabs] == ["Data", "Features", "Model", "Coffee"]
    assert {"Market summary", "Market history"} <= {header.value for header in app.subheader}
    assert len(app.dataframe) >= 5


def test_an_empty_data_dir_says_what_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run_dashboard(tmp_path, monkeypatch)

    assert not app.exception
    assert "make analysis" in app.warning[0].value
