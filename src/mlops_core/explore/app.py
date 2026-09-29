"""The explorer: a map of the domain's places, questions to its agent, the tables sliced
by hand, and what the data already says.

Six tabs under a band of headline numbers:

- **Map**: the layers the domain declares in its YAML - a number per area raised as
  columns, places as dots or counted in hexagons - with what each one is, a legend, and
  the areas ranked beside it.
- **Ask the agent**: a question in plain words; the agent answers from the tables, its
  models and its documents, cites each, and its query's rows are drawn as a chart the
  viewer can change. A failed query says why, and shows the query.
- **Explore by segment**: a measure, what to segment it by, what to colour it by, and
  filters - all picked from the YAML's lists, so the query is assembled, never written.
- **Findings**: the results worth showing unasked, each a chart and a sentence, and
  every study and figure the analysis wrote, to read or download.
- **Models**: how each model was trained and how it does - its target period by period,
  what each feature is worth, its error by period, and the monitor's last verdict - as
  the analysis and the monitor last wrote it, stamped with the partitions it came from.
- **About**: the sources, the models and the agent, and their limits.

Everything drawn is a SELECT run in the agent's locked session, so the app can show
nothing a question could not ask for. Run it with `make explore` (or `mlops explore`).
The map, the segments and the findings need only the built layers; the questions need
what `mlops agent ask` needs: Ollama, Qdrant with an index, and the prediction API.
"""

import html
import threading
import time
from contextlib import ExitStack
from typing import Any

import duckdb
import polars as pl
import streamlit as st

from mlops_core.adapter import load_adapter
from mlops_core.agent.sql import Refused, read_only
from mlops_core.config import ExploreDataset, ExploreFinding, MapLayer, Settings
from mlops_core.explore.charts import (
    Areas,
    Chart,
    ChartKind,
    check_chart,
    frame,
    infer_chart,
    vega_lite,
)
from mlops_core.explore.layers import areas_if_built, ranked, run_layer
from mlops_core.explore.maps import area_layer, deck, density_layer, point_layer
from mlops_core.explore.segments import segment_sql, summary, top_segments, values_sql
from mlops_core.explore.studies import domain_figures, domain_studies, figure, lineage
from mlops_core.explore.style import CSS

KINDS = ["bar", "line", "scatter", "points", "areas", "table"]
NONE = "(none)"
MAP_HEIGHT = 620
TABS = [
    ":material/map: Map",
    ":material/forum: Ask the agent",
    ":material/bar_chart: Explore by segment",
    ":material/lightbulb: Findings",
    ":material/model_training: Models",
    ":material/info: About",
]
MAP, ASK, SEGMENTS, FINDINGS, MODELS, ABOUT = TABS
DOTS, DENSITY = "dots", "density"
ROUTES = {
    "data": "answered from the tables",
    "prediction": "answered by a model",
    "knowledge": "answered from the documents",
    "mixed": "answered with several tools",
}

settings = Settings()
adapter = load_adapter(settings.domain)
config = adapter.config
data_dir = settings.data_dir / config.name
st.set_page_config(
    page_title=config.explore.title if config.explore else config.name,
    page_icon=":material/explore:",
    layout="wide",
)
if config.explore is None:
    st.error(f"{config.name} declares no `explore:` section in its YAML: there is no map.")
    st.stop()
assert config.explore is not None  # st.stop() ends the run; the type checker does not know
explore = config.explore


# --- The data: one locked session, and every query's rows cached ----------------------------


@st.cache_resource
def session() -> tuple[duckdb.DuckDBPyConnection, threading.Lock]:
    """One locked, read-only session for the page, and a lock: a DuckDB connection serves
    one query at a time, and every browser tab shares it."""
    return read_only(data_dir), threading.Lock()


def query(sql: str) -> tuple[pl.DataFrame, bool]:
    con, lock = session()
    with lock:
        return run_layer(con, sql)


@st.cache_data(show_spinner=False)
def cached(sql: str) -> tuple[pl.DataFrame, bool]:
    """A query's rows, kept for the app's life: the layers do not change while it runs."""
    return query(sql)


def rows_or_warning(name: str, sql: str) -> pl.DataFrame | None:
    """A query's rows, or None and a warning that says which table is missing."""
    try:
        return cached(sql)[0]
    except (Refused, duckdb.Error) as failed:
        st.warning(f"{name}: {first_line(failed)} (run `make data`?)")
        return None


@st.cache_resource
def areas() -> Areas | None:
    con, lock = session()
    with lock:
        return areas_if_built(con, explore)


@st.cache_resource(show_spinner="Starting the agent: Ollama, the index, the prediction API")
def agent() -> tuple[Any, ExitStack]:
    """The agent `mlops agent ask` uses, kept open for the app's life. Its clients live in
    an exit stack cached beside it: dropped, the stack's generator would be collected and
    would close them. A failure is not cached, so the next question tries again."""
    import mlflow

    from mlops_core.cli import agent_session

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{config.name}-agent")
    stack = ExitStack()
    found, _ = stack.enter_context(agent_session(adapter, settings))
    return found, stack


def first_line(error: BaseException | str) -> str:
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


def note(text: str) -> None:
    """A sentence set apart: what the viewer is looking at."""
    st.html(f'<div class="explore-note">{text}</div>')


def label(name: str) -> str:
    return name.replace("_", " ")


# --- The band over the tabs ---------------------------------------------------------------


def header() -> None:
    st.html(CSS)
    intro = f"<p>{html.escape(explore.intro)}</p>" if explore.intro else ""
    st.html(f'<div class="explore-hero"><h1>{html.escape(explore.title)}</h1>{intro}</div>')
    if not explore.metrics:
        return
    for column, metric in zip(st.columns(len(explore.metrics)), explore.metrics, strict=True):
        try:
            value = cached(metric.sql)[0].row(0)[0]
            trend = (
                cached(metric.trend)[0].to_series(0).drop_nulls().to_list()
                if metric.trend
                else None
            )
            shown = (
                f"{value:,.{metric.decimals}f} {metric.unit}".strip() if value is not None else "-"
            )
        except (Refused, duckdb.Error, IndexError):  # not built yet, or no rows
            shown, trend = "-", None
        column.metric(metric.label, shown, help=metric.help or None, border=True,
                      chart_data=trend or None, chart_type="area")  # fmt: skip


# --- Map ----------------------------------------------------------------------------------


def map_tab() -> None:
    area_layers = [layer for layer in explore.layers if layer.kind == "areas"]
    place_layers = [layer for layer in explore.layers if layer.kind == "points"]
    picks = st.columns([3, 3, 2])
    area_names = [NONE, *(layer.name for layer in area_layers)]
    picked_area = picks[0].selectbox(
        "Colour the areas by", area_names, index=1 if area_layers else 0, key="map_area"
    )
    picked_places = picks[1].multiselect(
        "Places on the map",
        [layer.name for layer in place_layers],
        default=[layer.name for layer in place_layers[:1]],
        key="map_places",
    )
    style = picks[2].segmented_control(
        "Places as", [DOTS, DENSITY], default=DENSITY, key="map_style", required=True
    )
    drawn: list[Any] = []
    explained: list[str] = []
    ranking: tuple[pl.DataFrame, str, str] | None = None
    for layer in area_layers:
        if layer.name == picked_area:
            ranking = draw_areas(layer, drawn, explained, raised=not picked_places)
    for layer in place_layers:
        if layer.name in picked_places:
            draw_places(layer, style or DENSITY, drawn, explained)
    if "on_map" in st.session_state:
        chart, rows, title = st.session_state["on_map"]
        drawn += answer_on_map(chart, rows, title)
        explained.append(f"<b>From your question</b>: {html.escape(title)}")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.pydeck_chart(deck(explore.view, drawn), height=MAP_HEIGHT)
        st.caption(
            "Drag to move, right-drag or Ctrl-drag to tilt, scroll to zoom; hover for the numbers."
        )
    with right:
        st.subheader("What you are looking at")
        for text in explained:
            note(text)
        if ranking is not None:
            rows, value, _ = ranking
            st.markdown(f"**{picked_area}, area by area**")
            ranked_chart = Chart(kind="bar", x=rows.columns[0], y=value)
            spec = vega_lite(ranked_chart, rows)
            if spec is not None:
                st.vega_lite_chart(spec | {"height": 300}, width="stretch")


def draw_areas(
    layer: MapLayer, drawn: list[Any], explained: list[str], raised: bool
) -> tuple[pl.DataFrame, str, str] | None:
    """The areas coloured and raised by the layer's number; the ranking beside the map."""
    rows = rows_or_warning(layer.name, layer.sql)
    known = areas()
    if rows is None:
        return None
    value = infer_chart(rows, known).y
    if known is None or value is None:
        st.warning(f"{layer.name}: the areas are not built yet, or the rows name none")
        return None
    drawn.append(area_layer(known, rows, value, layer.name, raised))
    numbers = rows[value].drop_nulls().cast(pl.Float64).to_list()
    low, high = (f"{min(numbers):,.1f}", f"{max(numbers):,.1f}") if numbers else ("", "")
    explained.append(
        f"<b>{html.escape(layer.name)}</b>: {html.escape(layer.description)}"
        f'<div class="explore-ramp"></div>'
        f'<div class="explore-legend"><span>{low}</span><span>{html.escape(layer.unit)}</span>'
        f"<span>{high}</span></div>"
        + ("Taller and darker is more" if raised else "Darker is more (flat, under the places)")
        + "; a grey area has no value."
    )
    return ranked(rows, known, value), value, layer.unit


def draw_places(layer: MapLayer, style: str, drawn: list[Any], explained: list[str]) -> None:
    rows = rows_or_warning(layer.name, layer.sql)
    if rows is None:
        return
    text = (
        f"<b>{html.escape(layer.name)}</b>: {html.escape(layer.description)} "
        f"({rows.height:,} places)"
    )
    if style == DENSITY:
        drawn.append(density_layer(rows, layer.name))
        text += ". Each hexagon counts the places within about 400 m: taller and darker is more."
    else:
        color = infer_chart(rows).color
        layer_drawn, legend = point_layer(rows, color, layer.name)
        drawn.append(layer_drawn)
        text += "".join(
            f'<br><span style="color: rgb{rgb}">&#9679;</span> {html.escape(value)}'
            for value, rgb in legend.items()
        )
    explained.append(text)


def answer_on_map(chart: Chart, rows: pl.DataFrame, title: str) -> list[Any]:
    """An answer's rows, drawn as the answer's chart said: areas or points."""
    known = areas()
    if chart.kind == "areas" and known is not None and chart.y is not None:
        return [area_layer(known, rows, chart.y, title)]
    if chart.kind == "points":
        return [point_layer(rows, chart.color, title)[0]]
    return []


# --- Ask the agent ------------------------------------------------------------------------


def ask_tab() -> None:
    note(
        "Ask in plain words, in English. The agent decides whether the question needs the "
        "tables, one of the models or the documents; it writes the SQL, calls the prediction "
        "API or searches the documents, and cites each. Every figure in its answer is "
        "checked against that evidence before it is shown, and the rows its query returned "
        "are drawn below the answer. The first question also starts the local model."
    )
    if explore.examples:
        st.pills("Try one", explore.examples, key="example", on_change=example_picked)
    question = st.chat_input("Ask about the data, the models or the documents", key="question")
    pending = st.session_state.pop("pending", None) or question
    if pending:
        ask(pending)
    turns = st.session_state.get("turns", [])
    for index in reversed(range(len(turns))):  # the newest answer first
        show_turn(turns[index], index)


def example_picked() -> None:
    """An example asks once: the pick is moved to `pending` and the pills cleared, so the
    same example can be asked again."""
    st.session_state["pending"] = st.session_state.get("example")
    st.session_state["example"] = None


def ask(question: str) -> None:
    """Ask the agent, and keep its answer with the rows its query returns - all of them up
    to the map's limit, since a chart draws more than the fifty a model reads."""
    turns = st.session_state.setdefault("turns", [])
    started = time.monotonic()
    try:
        with st.spinner(f"Answering: {question}"):
            reply = agent()[0].ask(question)
    except Exception as failed:  # the services are the usual cause; say which, keep going
        turns.append({"question": question, "error": first_line(failed)})
        return
    sql = reply.sql
    rows = None
    if sql is not None and sql.result is not None:
        try:
            rows, _ = query(sql.sql)
        except (Refused, duckdb.Error):
            rows = frame(sql.result.columns, sql.result.rows)
    turns.append(
        {
            "question": question,
            "text": reply.text,
            "route": getattr(reply, "route", None),
            "sources": reply.sources,
            "problems": reply.problems,
            "sql": sql.sql if sql else None,
            "sql_error": getattr(sql, "error", None) if sql and sql.result is None else None,
            "attempts": getattr(sql, "attempts", None),
            "prediction": getattr(reply, "prediction", None),
            "rows": rows,
            "seconds": time.monotonic() - started,
        }
    )


def show_turn(turn: dict[str, Any], index: int) -> None:
    with st.container(border=True):
        st.markdown(f"##### {turn['question']}")
        if "error" in turn:
            st.error(
                f"The agent could not answer: {turn['error']}. It needs Ollama, Qdrant with an "
                "index and the prediction API (`make services-up`, `make index`)."
            )
            return
        chips = [ROUTES.get(turn["route"] or "", "answered"), f"{turn['seconds']:.0f} s"]
        chips.append("every figure checked" if not turn["problems"] else "not fully verified")
        st.html("".join(f'<span class="explore-chip">{html.escape(c)}</span>' for c in chips))
        st.markdown(turn["text"])
        for problem in turn["problems"]:
            st.warning(f"Unverified: {problem}")
        if turn["sql_error"]:
            st.error(
                f"The query did not run, after {turn['attempts']} tries: {turn['sql_error']}. "
                "A small local model writes it; rephrasing often helps, and the same numbers "
                "are a few clicks away under **Explore by segment**."
            )
        show_prediction(turn["prediction"])
        rows = turn["rows"]
        if rows is not None and rows.is_empty():
            st.info("The query ran, and no rows matched.")
        elif rows is not None:
            show_rows(turn, rows, index)
        if turn["sources"]:
            st.caption("Sources: " + " · ".join(turn["sources"]))
        if turn["sql"]:
            with st.expander("The query"):
                st.code(turn["sql"], language="sql")


def show_prediction(prediction: Any) -> None:
    if prediction is None:
        return
    if prediction.error:
        st.warning(f"No prediction from {prediction.model}: {prediction.error}")
        return
    value = (prediction.response or {}).get("prediction")
    shown = f"{value:,.2f}" if isinstance(value, float) else str(value)
    st.metric(f"The {prediction.model} model predicts", shown, border=True)
    with st.expander("The item, as the model was given it"):
        st.json(prediction.request)


def show_rows(turn: dict[str, Any], rows: pl.DataFrame, index: int) -> None:
    """One value as a number; anything more as a chart the viewer can change."""
    numbers = [c for c in rows.columns if rows.schema[c].is_numeric()]
    if rows.height == 1 and numbers and rows.width <= 4:
        columns = st.columns(len(numbers))
        labels = [c for c in rows.columns if c not in numbers]
        prefix = " · ".join(str(rows[c][0]) for c in labels)
        for column, name in zip(columns, numbers, strict=True):
            value = rows[name][0]
            shown = f"{value:,.4g}" if isinstance(value, float) else f"{value:,}"
            column.metric(f"{prefix} {label(name)}".strip(), shown, border=True)
        return
    chart_of(turn, rows, index)


def chart_of(turn: dict[str, Any], rows: pl.DataFrame, index: int) -> None:
    """The answer's rows as a chart, checked before drawn; its kind and columns can be
    changed, and places can go on the map."""
    inferred = infer_chart(rows, areas())
    kind = st.session_state.get(f"kind{index}", inferred.kind)
    picked = {
        role: st.session_state.get(f"{role}{index}", getattr(inferred, role) or NONE)
        for role in ("x", "y", "color")
    }
    chart = Chart(kind=kind, **{r: None if v == NONE else v for r, v in picked.items()})  # type: ignore[arg-type]
    problems = check_chart(chart, rows, areas())
    for problem in problems:
        st.warning(problem)
    if not problems:
        spec = vega_lite(chart, rows, areas())
        if spec is None:
            st.dataframe(rows, hide_index=True)
        else:
            st.vega_lite_chart(spec, width="stretch")
    with st.expander("Change the chart"):
        columns = [NONE, *rows.columns]
        kind_box, x_box, y_box, color_box = st.columns(4)
        kind_box.selectbox("Kind", KINDS, index=KINDS.index(inferred.kind), key=f"kind{index}")
        for role, box in (("x", x_box), ("y", y_box), ("color", color_box)):
            default = getattr(inferred, role) or NONE
            box.selectbox(role, columns, index=columns.index(default), key=f"{role}{index}")
    if not problems and chart.kind in ("points", "areas"):
        st.button(
            "Put it on the map", key=f"map{index}", icon=":material/add_location:",
            on_click=put_on_map, args=(chart, rows, turn["question"]),
        )  # fmt: skip
    with st.expander(f"The rows ({rows.height:,})"):
        st.dataframe(rows, hide_index=True)


def put_on_map(chart: Chart, rows: pl.DataFrame, title: str) -> None:
    st.session_state["on_map"] = (chart, rows, title)
    st.session_state["view"] = MAP


# --- Explore by segment ------------------------------------------------------------------


def segments_tab() -> None:
    if not explore.datasets:
        st.info("The domain's YAML lists no `explore.datasets` to slice.")
        return
    note(
        "Pick a table, a measure and what to split it by: the query is assembled from the "
        "domain's own lists, runs in the same locked session as the agent's, and is shown "
        "under the chart. No model, so the same choice always gives the same numbers."
    )
    by_name = {dataset.name: dataset for dataset in explore.datasets}
    name = st.selectbox("What to explore", list(by_name), key="dataset")
    dataset = by_name[name]
    if dataset.description:
        st.caption(dataset.description)
    picks = st.columns(3)
    measure = picks[0].selectbox("Measure", list(dataset.measures), format_func=label,
                                 key=f"measure:{name}")  # fmt: skip
    by = picks[1].selectbox("Split by", dataset.dimensions, format_func=label, key=f"by:{name}")
    colours = [NONE, *(d for d in dataset.dimensions if d != by)]
    color = picks[2].selectbox("Colour by", colours, format_func=label, key=f"color:{name}")
    filters = pick_filters(dataset)
    sql = segment_sql(dataset, measure, by, None if color == NONE else color, filters)
    rows = rows_or_warning(dataset.name, sql)
    if rows is None:
        return
    shown = top_segments(rows, by, measure)
    if shown.height < rows.height:
        st.caption(
            f"The {shown[by].n_unique()} largest values of {label(by)} of {rows[by].n_unique()}."
        )
    st.markdown(f"**{summary(shown, by, measure, None if color == NONE else color)}**")
    chart = segment_chart(shown, by, measure, None if color == NONE else color)
    spec = vega_lite(chart, shown)
    if spec is not None and not check_chart(chart, shown):
        st.vega_lite_chart(spec, width="stretch")
    else:
        st.dataframe(shown, hide_index=True)
    with st.expander("The query"):
        st.code(sql, language="sql")
    with st.expander(f"The rows ({rows.height:,})"):
        st.dataframe(rows, hide_index=True)
        st.download_button("Download as CSV", rows.write_csv(), file_name=f"{name}.csv")


def pick_filters(dataset: ExploreDataset) -> dict[str, list[object]]:
    if not dataset.filters:
        return {}
    picked: dict[str, list[object]] = {}
    boxes = st.columns(min(len(dataset.filters), 4))
    for number, column in enumerate(dataset.filters):
        values = rows_or_warning(dataset.name, values_sql(dataset, column))
        options = values[column].to_list() if values is not None else []
        picked[column] = boxes[number % len(boxes)].multiselect(
            f"Only {label(column)}", options, key=f"filter:{dataset.name}:{column}",
            placeholder="any",
        )  # fmt: skip
    return picked


def segment_chart(rows: pl.DataFrame, by: str, measure: str, color: str | None) -> Chart:
    """A line over time, bars otherwise; a colour with too many values is left out."""
    kind: ChartKind = "line" if rows.schema[by].is_temporal() else "bar"
    if color is not None and rows[color].n_unique() > 12:
        st.caption(f"{label(color)} has too many values to colour by.")
        color = None
    return Chart(kind=kind, x=by, y=measure, color=color)


# --- Findings and about -------------------------------------------------------------------


def findings_tab() -> None:
    if not explore.findings:
        st.info("The domain's YAML lists no `explore.findings`.")
        return
    note("What the data already says, each from a query over the published tables.")
    for start in range(0, len(explore.findings), 2):
        for column, finding in zip(
            st.columns(2, gap="large"), explore.findings[start : start + 2], strict=False
        ):
            with column, st.container(border=True):
                show_finding(finding)
    every_study()


def every_study() -> None:
    """Every study and figure the analysis wrote, beyond the findings: to read or keep."""
    models = [model.name for model in config.models]
    studies = domain_studies(data_dir, models)
    if studies:
        st.subheader("Every study")
        name = st.selectbox("Study", studies, format_func=label, key="study")
        show_study(name)
    drawn = domain_figures(data_dir, models)
    if drawn:
        with st.expander(f"The figures the analysis drew ({len(drawn)})"):
            for path in drawn:
                st.image(str(path))


def show_finding(finding: ExploreFinding) -> None:
    st.subheader(finding.title)
    st.markdown(finding.text)
    rows = rows_or_warning(finding.title, finding.sql)
    if rows is None:
        return
    chart = (
        Chart(kind=finding.kind, x=finding.x, y=finding.y, color=finding.color)
        if finding.kind
        else infer_chart(rows, areas())
    )
    problems = check_chart(chart, rows, areas())
    spec = None if problems else vega_lite(chart, rows, areas())
    if spec is None:
        st.dataframe(rows, hide_index=True)
    else:
        st.vega_lite_chart(spec | {"height": 280}, width="stretch")
    with st.expander("The query"):
        st.code(finding.sql, language="sql")


def show_study(name: str, caption: str | None = None, figure_name: str | None = None) -> None:
    """One study as the analysis wrote it: its figure, its rows and the CSV of both."""
    rows = rows_or_warning(name, f"SELECT * FROM analysis.{name}")
    if rows is None:
        return
    if caption:
        st.markdown(f"**{caption}**")
    image = figure(data_dir, figure_name) if figure_name else None
    if image is not None:
        st.image(str(image))
    st.dataframe(rows, hide_index=True)
    st.download_button(
        "Download as CSV", rows.write_csv(), file_name=f"{name}.csv", key=f"csv:{name}"
    )


def models_tab() -> None:
    note(
        "What the analysis and the monitor last wrote about each model, read from the saved "
        "tables, not recomputed. A model is replaced only when a candidate beats both the "
        "baseline and the current champion, 95% sure, on the same rows."
    )
    names = [model.name for model in config.models]
    picked = st.segmented_control("Model", names, default=names[0], key="model", required=True)
    model = config.model_named(picked or names[0])
    st.markdown(f"**{model.name}**: {model.description}")
    stamp = lineage(data_dir, model.name)
    if not stamp:
        st.info(f"No study of {model.name} has been built yet: run `make analysis`.")
        return
    st.caption(" · ".join(f"{key}: {value}" for key, value in stamp.items()))
    from mlops_core.monitoring.drift import latest_verdict  # MLflow: only when the tab opens

    found = latest_verdict(data_dir, model.name)
    if found is None:
        st.caption("The monitor has not compared its periods yet (`make monitor`).")
    elif found.retrain:
        st.warning(f"The monitor asks for retraining ({found.current}): {'; '.join(found.reasons)}")
    else:
        st.success(f"The monitor compared {found.current} with the periods before: nothing due.")
    named = model.name + "_{}"
    data, features, errors = st.tabs(["The target", "The features", "The error"])
    with data:
        show_study(named.format("target_distribution"), "The target, period by period",
                   named.format("target_distribution"))  # fmt: skip
        show_study(named.format("categorical_profile"), "What each period is made of")
    with features:
        show_study(named.format("feature_recommendation"), "What each feature is worth",
                   named.format("feature_importance"))  # fmt: skip
        st.caption(
            "`suggested_action` is a prompt to look, never an instruction: a feature with no "
            "correlation of its own can still be useful to the model."
        )
        show_study(named.format("numeric_profile"), "The numeric features in detail",
                   named.format("numeric_signal"))  # fmt: skip
    with errors:
        show_residuals(model.name, model.items.period)


def show_residuals(model: str, period: str) -> None:
    rows = rows_or_warning(model, f"SELECT * FROM analysis.{model}_residuals")
    if rows is None:
        st.info("No batch predictions yet: run `make predict`, then `make analysis`.")
        return
    image = figure(data_dir, f"{model}_residual_bias")
    if image is not None:
        st.image(str(image))
    periods = sorted(rows[period].unique().to_list())
    chosen = st.selectbox("Period", periods, index=len(periods) - 1, key=f"period:{model}")
    st.dataframe(rows.filter(pl.col(period) == chosen), hide_index=True)
    st.caption(
        "The error on a period the model trained on forecasts nothing; the newest period is "
        "the one to read."
    )


def about_tab() -> None:
    if explore.about:
        st.markdown(explore.about)
    st.subheader("The models")
    for model in config.models:
        st.markdown(f"- **{model.name}** - {model.description}")
    st.subheader("How an answer is made")
    st.markdown(
        "1. **Route**: the local model decides what the question needs - the tables, a "
        "model's prediction, the documents, or several.\n"
        "2. **Tools**: it writes one SELECT, run in a read-only session that cannot touch "
        "anything but the published tables (a failed query goes back to it, twice at most); "
        "or it describes the item for the prediction API; or it searches the documents.\n"
        "3. **Answer and check**: every figure in the answer must be in the evidence and "
        "every citation must point at evidence it was given; if not, it rewrites once, and "
        "what is still unverified is shown.\n\n"
        "Every step is traced in MLflow. The model is small and local: it can write a "
        "query that runs and answers a slightly different question, which is why the query "
        "is always shown."
    )


# --- The page -----------------------------------------------------------------------------

header()
views = st.tabs(TABS, key="view", on_change="rerun")
for tab, render in zip(
    views, (map_tab, ask_tab, segments_tab, findings_tab, models_tab, about_tab), strict=True
):
    if tab.open:
        with tab:
            render()
