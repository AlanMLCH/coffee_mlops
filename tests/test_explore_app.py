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
from mlops_core.analysis import pipeline
from mlops_core.analysis.pipeline import build_analysis
from mlops_core.config import MapView
from mlops_core.data.clean import build_clean
from mlops_core.explore import app as app_module
from mlops_core.explore.charts import Areas
from mlops_core.explore.layers import ranked
from mlops_core.explore.maps import (
    NO_VALUE,
    PALETTE,
    STOPS,
    area_layer,
    deck,
    density_layer,
    hexagons,
    point_layer,
    ramp,
)
from mlops_core.ml.features import build_features
from mlops_core.ml.registry import ServedModel
from tests.fakes import ConstantModel

APP = Path(app_module.__file__)
MAP, ASK, SEGMENTS, FINDINGS, MODELS, ABOUT = app_module.TABS
SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
AREAS = Areas("area_id", "area", {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"id": key, "name": name}, "geometry": SQUARE}
    for key, name in (("a1", "North"), ("a2", "South"), ("a3", "East"))
]})  # fmt: skip


# --- The map's layers ------------------------------------------------------------------------


def test_areas_rise_and_darken_with_their_number_and_the_silent_stay_flat() -> None:
    values = pl.DataFrame({"area": ["North", "South"], "shops": [10, 5]})

    layer = area_layer(AREAS, values, "shops", "density")

    features = {f["properties"]["name"]: f["properties"] for f in layer.data["features"]}
    assert features["North"]["elevation"] == 2 * features["South"]["elevation"]
    assert features["North"]["fill"] == list(STOPS[-1])
    assert features["East"] == features["East"] | {"fill": list(NO_VALUE), "elevation": 0.0}
    assert features["East"]["tooltip"] == "East: no data (shops)"
    assert features["North"]["tooltip"] == "North: 10 (shops)"


def test_areas_are_found_by_their_key_too() -> None:
    values = pl.DataFrame({"area_id": ["a3"], "price": [380.5]})

    layer = area_layer(AREAS, values, "price", "price")

    east = layer.data["features"][2]["properties"]
    assert east["tooltip"] == "East: 380.50 (price)" and east["fill"] == list(STOPS[-1])


def test_a_ranking_names_its_areas_largest_first() -> None:
    by_key = pl.DataFrame({"area_id": ["a1", "a2", "a3"], "shops": [3.0, 9.0, None]})
    by_name = pl.DataFrame({"area": ["East"], "shops": [1.0]})

    assert ranked(by_key, AREAS, "shops").rows() == [("South", 9.0), ("North", 3.0)]
    assert ranked(by_name, AREAS, "shops").rows() == [("East", 1.0)]


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


def test_places_are_counted_in_hexagons_of_the_radius_asked() -> None:
    """A place's hexagon is centred within the radius of it; three places a few metres from
    that centre share it, and one two kilometres east has its own. (Two places a metre
    apart can still straddle an edge: the grid is fixed, not laid around them.)"""
    alone = pl.DataFrame({"latitude": [19.4], "longitude": [-99.1]})
    centre = hexagons(alone, radius=400)[0]
    assert abs(centre["latitude"] - 19.4) * 111_320 <= 400
    assert abs(centre["longitude"] + 99.1) * 111_320 * 0.943 <= 400  # cos(19.4°)
    near = [(centre["latitude"] + d, centre["longitude"] - d) for d in (0.0, 1e-5, -1e-5)]
    places = pl.DataFrame(
        [*near, (centre["latitude"], centre["longitude"] + 0.02), (None, -99.0)],
        schema=["latitude", "longitude"],
        orient="row",
    )

    cells = hexagons(places, radius=400)

    assert [cell["count"] for cell in cells] == [3, 1]
    assert cells[0]["latitude"] == pytest.approx(centre["latitude"])
    assert hexagons(places.clear()) == []


def test_hexagons_rise_with_their_count_and_say_it() -> None:
    places = pl.DataFrame({"lat": [19.4, 19.4, 19.5], "lon": [-99.1, -99.1, -99.2]})

    layer = density_layer(places, "shops", radius=300)

    first, second = layer.data
    assert first["elevation"] == 2 * second["elevation"]
    assert first["fill"] == list(STOPS[-1])
    assert first["tooltip"] == "2 places within about 300 m (shops)"
    assert second["tooltip"] == "1 place within about 300 m (shops)"


def test_the_map_opens_where_the_domain_says() -> None:
    view = MapView(latitude=19.39, longitude=-99.14, zoom=10.2, pitch=40)

    drawn = deck(view, [])

    state = drawn.initial_view_state
    assert (state.latitude, state.longitude, state.zoom, state.pitch) == (19.39, -99.14, 10.2, 40)
    assert drawn._tooltip == {"text": "{tooltip}"}  # pydeck keeps it for the widget only


def test_the_ramp_runs_along_its_stops_and_stays_inside() -> None:
    assert ramp(0.0) == STOPS[0] and ramp(1.0) == STOPS[-1]
    assert ramp(-1.0) == STOPS[0] and ramp(2.0) == STOPS[-1]
    assert ramp(0.5) == STOPS[len(STOPS) // 2]  # seven stops: the middle one


# --- The page --------------------------------------------------------------------------------

BY_BOROUGH = (
    "SELECT borough, count(*) AS shops FROM clean.coffee_shops "
    "WHERE borough IS NOT NULL GROUP BY borough ORDER BY shops DESC"
)


class StoodIn:
    """An agent that answers every question with the same query and a sentence."""

    def __init__(
        self,
        sql: str | None = BY_BOROUGH,
        fails: bool = False,
        query_fails: bool = False,
        prediction: Any = None,
        answered: bool = True,
    ):
        self.sql, self.fails, self.query_fails = sql, fails, query_fails
        self.prediction, self.answered = prediction, answered
        self.asked: list[str] = []

    def ask(self, question: str) -> Any:
        self.asked.append(question)
        if self.fails:
            raise ConnectionError("Ollama is not answering at http://127.0.0.1:11434")
        answer = None
        if self.sql:
            result = None if self.query_fails else QueryResult(self.sql, ["x"], [], False)
            error = 'Parser Error: syntax error at or near "price_mxn_per_kg"'
            answer = SimpleNamespace(sql=self.sql, result=result, attempts=3,
                                     error=error if self.query_fails else None)  # fmt: skip
        unverified = ["The figure 9 is not in the evidence"]
        return SimpleNamespace(
            text="Cuauhtémoc [sql].", route="data", sources=["[sql] the tables"], sql=answer,
            problems=unverified if self.answered else [], prediction=self.prediction,
            answered=self.answered,
        )  # fmt: skip


@pytest.fixture
def data_dir(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> Path:
    """The clean layer built from the fixtures: boroughs, shops, shelf prices."""
    build_clean(coffee_adapter, raw_dir.parent)
    return raw_dir.parent.parent


def explorer(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, agent: StoodIn | None = None, view: str = MAP
) -> AppTest:
    page = prepared(data_dir, monkeypatch, agent)
    page.session_state["view"] = view
    page.run()
    return page


def prepared(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, agent: StoodIn | None = None
) -> AppTest:
    """The page, not yet run, over `data_dir` and with `agent` standing in."""
    stood_in = agent or StoodIn()

    @contextmanager
    def session(adapter: object, settings: object) -> Iterator[tuple[StoodIn, str]]:
        yield stood_in, "stood-in@0"

    monkeypatch.setattr(cli, "agent_session", session)
    # The page caches its session, its queries and its agent for the process: one test's
    # must not answer the next one's.
    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setenv("MLOPS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MLOPS_DOMAIN", "coffee")
    monkeypatch.setenv("MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{(data_dir / 'm.db').as_posix()}")
    return AppTest.from_file(str(APP), default_timeout=120)


def test_the_page_opens_on_the_map_under_its_headline_numbers(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch)

    assert not page.exception
    assert [tab.label for tab in page.tabs] == app_module.TABS
    labels = [metric.label for metric in page.metric]
    assert labels[:2] == ["Cherry at the farm gate", "Green coffee at the port"]
    shops = next(m for m in page.metric if m.label == "Coffee shops in the city")
    assert shops.value not in ("", "-")  # the fixtures' register has coffee shops
    assert page.selectbox(key="map_area").value == "Coffee shops per km² (DENUE)"
    assert page.multiselect(key="map_places").value == ["Coffee shops"]
    assert page.get("deck_gl_json_chart") and page.get("vega_lite_chart")  # the map, the ranking
    assert not page.warning


def test_places_can_be_dots_or_hexagons(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page = explorer(data_dir, monkeypatch)

    page.button_group(key="map_style").set_value(app_module.DOTS).run()

    assert not page.exception and not page.warning


def test_an_answer_comes_with_its_query_drawn_as_a_chart_you_can_change(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = StoodIn()
    page = explorer(data_dir, monkeypatch, agent, view=ASK)

    page.chat_input(key="question").set_value("Where are the coffee shops?").run()

    assert not page.exception
    assert agent.asked == ["Where are the coffee shops?"]
    assert "Cuauhtémoc [sql]." in [m.value for m in page.markdown]
    assert page.code[0].value == BY_BOROUGH
    assert "Unverified: The figure 9 is not in the evidence" in [w.value for w in page.warning]
    # The rows name boroughs and count shops: a map of areas, until the viewer says bars.
    kinds = {box.key: box.value for box in page.selectbox}
    assert kinds == {"kind0": "areas", "x0": "(none)", "y0": "shops", "color0": "(none)"}
    page.selectbox(key="kind0").set_value("bar").run()
    page.selectbox(key="x0").set_value("borough").run()
    assert page.get("vega_lite_chart")
    # Bars whose height is a name: refused, and the page says why.
    page.selectbox(key="y0").set_value("borough").run()
    assert "y must be a number; 'borough' is not" in [w.value for w in page.warning]


def test_the_box_stays_open_for_the_next_question(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = StoodIn()
    page = explorer(data_dir, monkeypatch, agent, view=ASK)

    page.chat_input(key="question").set_value("First?").run()
    page.chat_input(key="question").set_value("Second?").run()

    assert agent.asked == ["First?", "Second?"]
    questions = [m.value for m in page.markdown if m.value.startswith("#####")]
    assert questions == ["##### Second?", "##### First?"]  # the newest on top


def test_an_example_is_asked_once_and_can_be_asked_again(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = StoodIn()
    page = explorer(data_dir, monkeypatch, agent, view=ASK)
    example = page.button_group(key="example").options[0]

    page.button_group(key="example").set_value(example).run()
    page.run()  # a later rerun does not ask again
    page.button_group(key="example").set_value(example).run()

    assert agent.asked == [example, example]


def test_a_query_that_did_not_run_says_why_and_shows_itself(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, StoodIn(query_fails=True), view=ASK)

    page.chat_input(key="question").set_value("Median by fortnight?").run()

    assert not page.exception
    assert "after 3 tries: Parser Error" in page.error[0].value
    assert "Explore by segment" in page.error[0].value
    assert page.code[0].value == BY_BOROUGH


def test_a_prediction_is_shown_with_the_item_it_was_made_for(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predicted = SimpleNamespace(model="offer", request={"shop": "almanegra"},
                                response={"prediction": 1324.93}, error=None)  # fmt: skip
    failed = SimpleNamespace(model="offer", request={}, response=None, error="the API is down")
    page = explorer(data_dir, monkeypatch, StoodIn(sql=None, prediction=predicted), view=ASK)

    page.chat_input(key="question").set_value("What would it cost?").run()

    assert [(m.label, m.value) for m in page.metric][-1] == ("The offer model predicts", "1,324.93")
    page = explorer(data_dir, monkeypatch, StoodIn(sql=None, prediction=failed), view=ASK)
    page.chat_input(key="question").set_value("What would it cost?").run()
    assert "No prediction from offer: the API is down" in [w.value for w in page.warning]


def test_an_answer_about_places_can_go_on_the_map(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, view=ASK)
    page.chat_input(key="question").set_value("Where?").run()

    page.button(key="map0").click().run()

    assert not page.exception
    assert page.session_state["view"] == MAP
    assert any("From your question" in h.proto.body for h in page.get("html"))


def test_an_answer_taken_off_the_map_leaves_it(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, view=ASK)
    page.chat_input(key="question").set_value("Where?").run()
    page.button(key="map0").click().run()

    page.button(key="off_map").click().run()

    assert not page.exception and "on_map" not in page.session_state
    assert not any("From your question" in h.proto.body for h in page.get("html"))


def test_no_answer_is_never_shown_as_a_checked_one(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, StoodIn(answered=False), view=ASK)

    page.chat_input(key="question").set_value("How much did Jalisco grow?").run()

    chips = next(h.proto.body for h in page.get("html") if "explore-chip" in h.proto.body
                 and "<style>" not in h.proto.body)  # fmt: skip
    assert "no answer found" in chips and "every figure checked" not in chips


def test_an_answer_without_a_query_has_no_chart(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, StoodIn(sql=None), view=ASK)

    page.chat_input(key="question").set_value("Why does altitude matter?").run()

    assert not page.exception and not page.get("vega_lite_chart")


def test_without_its_services_the_agent_says_what_it_needs(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, StoodIn(fails=True), view=ASK)

    page.chat_input(key="question").set_value("Anything?").run()

    assert not page.exception
    assert "Ollama is not answering" in page.error[0].value
    assert "make services-up" in page.error[0].value


def test_a_table_is_sliced_by_hand_without_the_agent(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = StoodIn()
    page = explorer(data_dir, monkeypatch, agent, view=SEGMENTS)

    page.selectbox(key="dataset").set_value("Shelf prices of packaged coffee (PROFECO)").run()
    page.selectbox(key="by:Shelf prices of packaged coffee (PROFECO)").set_value("fortnight").run()

    assert not page.exception and not agent.asked
    assert page.get("vega_lite_chart")
    assert page.code[0].value.startswith("SELECT fortnight, median(price_mxn_per_kg)")
    assert any(m.value.startswith("**median mxn per kg:") for m in page.markdown)


def test_an_empty_value_is_named_and_can_be_picked(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shelf prices outside the city have no borough: a bar says so instead of "null"."""
    page = explorer(data_dir, monkeypatch, view=SEGMENTS)
    name = "Shelf prices of packaged coffee (PROFECO)"
    page.selectbox(key="dataset").set_value(name).run()

    page.selectbox(key=f"by:{name}").set_value("borough").run()

    assert not page.exception and page.get("vega_lite_chart")
    assert "outside Mexico City" in page.dataframe[-1].value["borough"].to_list()


def linked(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, agent: StoodIn, **params: str
) -> AppTest:
    """The page as a link from the MCP server opens it."""
    page = prepared(data_dir, monkeypatch, agent)
    for key, value in params.items():
        page.query_params[key] = value
    page.run()
    return page


def test_a_link_opens_the_page_on_the_slice_it_names(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "Shelf prices of packaged coffee (PROFECO)"

    page = linked(data_dir, monkeypatch, StoodIn(), view="segments", dataset=name,
                  measure="prices", by="borough", color="nonsense")  # fmt: skip

    assert not page.exception and page.session_state["view"] == SEGMENTS
    assert page.selectbox(key="dataset").value == name
    assert page.selectbox(key=f"measure:{name}").value == "prices"
    assert page.selectbox(key=f"by:{name}").value == "borough"
    assert page.selectbox(key=f"color:{name}").value == app_module.NONE  # not a dimension


def test_a_link_can_ask_a_question_or_open_a_map_layer(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = StoodIn()

    asked = linked(data_dir, monkeypatch, agent, q="Where are the coffee shops?")
    mapped = linked(data_dir, monkeypatch, StoodIn(), view="map",
                    layer="Median shelf price of ground coffee")  # fmt: skip
    places = linked(data_dir, monkeypatch, StoodIn(), view="map", layer="Shelves PROFECO priced")

    assert agent.asked == ["Where are the coffee shops?"]
    assert asked.session_state["view"] == ASK
    assert mapped.selectbox(key="map_area").value == "Median shelf price of ground coffee"
    assert places.multiselect(key="map_places").value == ["Shelves PROFECO priced"]


def test_the_findings_and_the_about_page_draw_from_the_tables(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    findings = explorer(data_dir, monkeypatch, view=FINDINGS)
    about = explorer(data_dir, monkeypatch, view=ABOUT)

    assert not findings.exception and not about.exception
    assert next(h.value for h in findings.subheader) == "A kilogram, from the farm to the shelf"
    assert any("**review**" in m.value for m in about.markdown)


@pytest.fixture
def analysed(
    data_dir: Path, coffee_adapter: CoffeeAdapter, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The fixtures' layers with the analysis built over them, and a verdict that asks for
    a retraining, as the monitor would leave it."""
    config = coffee_adapter.config
    adapter = CoffeeAdapter(
        config.model_copy(
            update={
                "analysis": config.analysis.model_copy(update={"min_rows": 1}),
                "market_analysis": config.market_analysis.model_copy(update={"market_year": 2022}),
            }
        )
    )
    monkeypatch.setattr(
        pipeline, "load_champion", lambda *a, **k: ServedModel(ConstantModel(), "1", "cache")
    )
    domain_dir = data_dir / "coffee"
    build_features(adapter, "review", domain_dir)
    build_analysis(adapter, domain_dir, "sqlite:///unused")
    drift = domain_dir / "monitoring" / "review_drift" / "built_at=20260929T000000Z"
    drift.mkdir(parents=True)
    (drift / "verdict.json").write_text(
        '{"model": "review", "current": "cqi_2023", "retrain": true, '
        '"reasons": ["the target drifted"], "data_version": "abc"}',
        encoding="utf-8",
    )
    (drift / "manifest.json").write_text("{}", encoding="utf-8")
    return data_dir


def test_a_models_studies_are_shown_with_what_they_came_from(
    analysed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(analysed, monkeypatch, view=MODELS)

    assert not page.exception
    # The stamp says which partitions are on screen: without it last week's numbers pass
    # for today's.
    assert any("review_features: built_at=" in c.value for c in page.caption)
    assert "the target drifted" in page.warning[0].value  # the monitor's verdict
    shown = {m.value for m in page.markdown}
    assert {"**The target, period by period**", "**What each feature is worth**"} <= shown
    assert page.get("image")  # the figures, found in the newest complete drawing
    assert len(page.dataframe) >= 4


def test_a_model_with_nothing_predicted_says_why_there_is_no_error_to_show(
    analysed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixtures' model has studies but was never used to predict: no residuals."""
    page = explorer(analysed, monkeypatch, view=MODELS)

    assert not page.exception
    assert any("has no batch predictions" in i.value for i in page.info)
    assert not any("Catalog Error" in w.value for w in page.warning)


def test_a_model_without_studies_says_what_to_run(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(data_dir, monkeypatch, view=MODELS)

    assert not page.exception
    assert "run `make analysis`" in page.info[0].value


def test_every_study_the_analysis_wrote_is_one_pick_away(
    analysed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(analysed, monkeypatch, view=FINDINGS)

    studies = page.selectbox(key="study")
    assert "market summary" in studies.options and "review residuals" not in studies.options
    studies.set_value("market_summary").run()
    assert not page.exception
    assert page.get("download_button")


def test_a_layer_whose_table_is_not_built_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = explorer(tmp_path, monkeypatch)

    assert not page.exception
    assert any("`make status` says what to build" in w.value for w in page.warning)
    assert {m.value for m in page.metric} == {"-"}  # nothing built, nothing to show


def test_a_domain_without_a_map_says_so(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, coffee_adapter: CoffeeAdapter
) -> None:
    mapless = CoffeeAdapter(coffee_adapter.config.model_copy(update={"explore": None}))
    monkeypatch.setattr(mlops_core.adapter, "load_adapter", lambda name: mapless)

    page = explorer(data_dir, monkeypatch)

    assert "declares no `explore:` section" in page.error[0].value
