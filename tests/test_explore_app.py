"""The explorer app: its map layers as deck.gl sees them, and the page itself, run by
Streamlit's AppTest over clean layers built from the fixtures and an agent stood in for
the services (Ollama, Qdrant, the prediction API) the real one needs."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import mlops_core.adapter
from domains.coffee.adapter import CoffeeAdapter
from mlops_core import cli
from mlops_core.agent.sql import QueryResult
from mlops_core.config import MapView
from mlops_core.data.clean import build_clean
from mlops_core.explore import app as app_module
from mlops_core.explore.charts import Areas
from mlops_core.explore.maps import (
    HIGH,
    LOW,
    NO_VALUE,
    PALETTE,
    area_layer,
    deck,
    point_layer,
    ramp,
)

APP = Path(app_module.__file__)
SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
AREAS = Areas("area_id", "area", {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"id": key, "name": name}, "geometry": SQUARE}
    for key, name in (("a1", "North"), ("a2", "South"), ("a3", "East"))
]})  # fmt: skip


# --- The map's layers ------------------------------------------------------------------------


def test_areas_rise_and_deepen_with_their_number_and_the_silent_stay_flat() -> None:
    values = pl.DataFrame({"area": ["North", "South"], "shops": [10, 5]})

    layer = area_layer(AREAS, values, "shops", "density")

    features = {f["properties"]["name"]: f["properties"] for f in layer.data["features"]}
    assert features["North"]["elevation"] == 2 * features["South"]["elevation"]
    assert features["North"]["fill"] == list(HIGH)
    assert features["East"] == features["East"] | {"fill": list(NO_VALUE), "elevation": 0.0}
    assert features["East"]["tooltip"] == "East: no data (shops)"
    assert features["North"]["tooltip"] == "North: 10 (shops)"


def test_areas_are_found_by_their_key_too() -> None:
    values = pl.DataFrame({"area_id": ["a3"], "price": [380.5]})

    layer = area_layer(AREAS, values, "price", "price")

    east = layer.data["features"][2]["properties"]
    assert east["tooltip"] == "East: 380.50 (price)" and east["fill"] == list(HIGH)


def test_points_are_coloured_by_category_and_the_unplaced_are_left_out() -> None:
    places = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
            "source": ["osm", "denue", "osm"],
            "latitude": [19.4, 19.5, None],
            "longitude": [-99.1, -99.2, -99.3],
        }
    )

    layer, legend = point_layer(places, "source", "shops")
    plain, no_legend = point_layer(places, None, "shops")

    assert legend == {"denue": PALETTE[0], "osm": PALETTE[1]}
    assert [r["color"] for r in layer.data] == [list(PALETTE[1]), list(PALETTE[0])]
    assert layer.data[0]["position"] == [-99.1, 19.4]
    assert layer.data[0]["tooltip"] == "name: a · source: osm"
    assert no_legend == {} and plain.data[1]["color"] == list(PALETTE[0])


def test_the_map_opens_where_the_domain_says() -> None:
    view = MapView(latitude=19.39, longitude=-99.14, zoom=10.2, pitch=40)

    drawn = deck(view, [])

    state = drawn.initial_view_state
    assert (state.latitude, state.longitude, state.zoom, state.pitch) == (19.39, -99.14, 10.2, 40)
    assert drawn._tooltip == {"text": "{tooltip}"}  # pydeck keeps it for the widget only


def test_the_ramp_runs_from_light_to_deep_and_stays_inside() -> None:
    assert ramp(0.0) == LOW and ramp(1.0) == HIGH
    assert ramp(-1.0) == LOW and ramp(2.0) == HIGH


# --- The page --------------------------------------------------------------------------------

BY_BOROUGH = (
    "SELECT borough, count(*) AS shops FROM clean.coffee_shops "
    "WHERE borough IS NOT NULL GROUP BY borough ORDER BY shops DESC"
)


class StoodIn:
    """An agent that answers every question with the same query and a sentence."""

    def __init__(self, sql: str | None = BY_BOROUGH, fails: bool = False):
        self.sql, self.fails, self.asked = sql, fails, []

    def ask(self, question: str) -> Any:
        self.asked.append(question)
        if self.fails:
            raise ConnectionError("Ollama is not answering at http://127.0.0.1:11434")
        answer = None
        if self.sql:
            answer = SimpleNamespace(sql=self.sql, result=QueryResult(self.sql, ["x"], [], False))
        unverified = ["The figure 9 is not in the evidence"]
        return SimpleNamespace(
            text="Cuauhtémoc [sql].", sources=["[sql] the tables"], sql=answer, problems=unverified
        )


@pytest.fixture
def data_dir(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> Path:
    """The clean layer built from the fixtures: boroughs, shops, shelf prices."""
    build_clean(coffee_adapter, raw_dir.parent)
    return raw_dir.parent.parent


def explorer(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, agent: StoodIn | None = None
) -> AppTest:
    stood_in = agent or StoodIn()

    @contextmanager
    def session(adapter: object, settings: object) -> Iterator[tuple[StoodIn, str]]:
        yield stood_in, "stood-in@0"

    monkeypatch.setattr(cli, "agent_session", session)
    # The page caches its session and its agent for the process: one test's must not
    # answer the next one's questions.
    st.cache_resource.clear()
    monkeypatch.setenv("MLOPS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MLOPS_DOMAIN", "coffee")
    monkeypatch.setenv("MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{(data_dir / 'm.db').as_posix()}")
    page = AppTest.from_file(str(APP), default_timeout=120)
    page.run()
    return page


def test_the_map_opens_with_an_area_layer_and_a_layer_of_places(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch)

    assert not page.exception
    assert page.title[0].value == "Coffee in Mexico City"
    assert page.multiselect[0].value == ["Coffee shops per km² (DENUE)", "Coffee shops"]
    assert "**Coffee shops**: DENUE and OpenStreetMap, side by side" in [
        c.value for c in page.caption
    ]
    assert len(page.button) == 4  # the YAML's example questions, before the first one
    assert not page.warning


def test_an_answer_comes_with_its_query_drawn_as_a_chart_you_can_change(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = StoodIn()
    page = explorer(data_dir, monkeypatch, agent)

    page.chat_input[0].set_value("Where are the coffee shops?").run()

    assert not page.exception
    assert agent.asked == ["Where are the coffee shops?"]
    assert "Cuauhtémoc [sql]." in [m.value for m in page.markdown]
    assert page.code[0].value == BY_BOROUGH
    assert "Unverified: The figure 9 is not in the evidence" in [w.value for w in page.warning]
    # The rows name boroughs and count shops: a map of areas, until the viewer says bars.
    kinds = {box.label: box.value for box in page.selectbox}
    assert kinds == {"Kind": "areas", "x": "(none)", "y": "shops", "color": "(none)"}
    page.selectbox(key="kind0").set_value("bar").run()
    page.selectbox(key="x0").set_value("borough").run()
    assert not page.warning[1:] and page.get("vega_lite_chart")
    # Bars whose height is a name: refused, and the page says why.
    page.selectbox(key="y0").set_value("borough").run()
    assert "y must be a number; 'borough' is not" in [w.value for w in page.warning]


def test_an_answer_about_places_can_go_on_the_map(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch)
    page.button(key="example0").click().run()

    page.button(key="map0").click().run()

    assert not page.exception
    captions = [c.value for c in page.caption]
    assert any(c.startswith("**On the map from an answer**") for c in captions)


def test_an_answer_without_a_query_has_no_chart(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, StoodIn(sql=None))

    page.chat_input[0].set_value("Why does altitude matter?").run()

    assert not page.exception and not page.subheader


def test_without_its_services_the_agent_says_what_it_needs_and_the_map_stays(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, StoodIn(fails=True))

    page.chat_input[0].set_value("Anything?").run()

    assert not page.exception
    assert "Ollama is not answering" in page.error[0].value
    assert "make services-up" in page.error[0].value
    assert page.multiselect[0].value  # the map is still there


def test_a_layer_whose_table_is_not_built_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(tmp_path, monkeypatch)

    assert not page.exception
    assert any("run `make data`?" in w.value for w in page.warning)


def test_a_domain_without_a_map_says_so(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, coffee_adapter: CoffeeAdapter
) -> None:
    mapless = CoffeeAdapter(coffee_adapter.config.model_copy(update={"explore": None}))
    monkeypatch.setattr(mlops_core.adapter, "load_adapter", lambda name: mapless)

    page = explorer(data_dir, monkeypatch)

    assert "declares no `explore:` section" in page.error[0].value
