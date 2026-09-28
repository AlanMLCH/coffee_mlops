"""The explorer: a map of the domain's places, questions to its agent, and a chart of each
answer.

Three things on one page. The map draws the layers the domain declares in its YAML - a
number per area raised as columns, places as points - each one a SELECT run in the
agent's locked session. The chat asks the agent, which answers from the tables, its
models and its documents and cites each. And each answer's query result is drawn: its
shape picks the chart (`explore.charts`), a person can change the kind and what goes on
each axis and in the colour, every choice is checked against the columns before it is
drawn, and an answer about places can be put on the map beside the layers.

Run it with `make explore` (or `mlops explore`). The map needs only the built layers; the
questions need what `mlops agent ask` needs: Ollama, Qdrant with an index, and the
prediction API.
"""

import html
import threading
from contextlib import ExitStack
from typing import Any

import duckdb
import polars as pl
import streamlit as st

from mlops_core.adapter import load_adapter
from mlops_core.agent.sql import Refused, read_only
from mlops_core.config import Settings
from mlops_core.explore.charts import Areas, Chart, check_chart, frame, infer_chart, vega_lite
from mlops_core.explore.layers import areas_if_built, run_layer
from mlops_core.explore.maps import area_layer, deck, point_layer

KINDS = ["bar", "line", "scatter", "points", "areas", "table"]
NONE = "(none)"
MAP_HEIGHT = 560

settings = Settings()
adapter = load_adapter(settings.domain)
config = adapter.config
st.set_page_config(
    page_title=config.explore.title if config.explore else config.name, layout="wide"
)
if config.explore is None:
    st.error(f"{config.name} declares no `explore:` section in its YAML: there is no map.")
    st.stop()
assert config.explore is not None  # st.stop() ends the run; the type checker does not know
explore = config.explore


@st.cache_resource
def session() -> tuple[duckdb.DuckDBPyConnection, threading.Lock]:
    """One locked, read-only session for the page, and a lock: a DuckDB connection serves
    one query at a time, and every browser tab shares it."""
    return read_only(settings.data_dir / config.name), threading.Lock()


def query(sql: str) -> tuple[pl.DataFrame, bool]:
    con, lock = session()
    with lock:
        return run_layer(con, sql)


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


def ask(question: str) -> None:
    """Ask the agent, and keep its answer with the rows its query returns - all of them up
    to the map's limit, since a chart draws more than the fifty a model reads."""
    turns = st.session_state.setdefault("turns", [])
    try:
        with st.spinner("Answering from the tables, the models and the documents"):
            reply = agent()[0].ask(question)
    except Exception as failed:  # the services are the usual cause; say which, keep going
        turns.append({"question": question, "error": str(failed).strip().splitlines()[0]})
        return
    rows = None
    if reply.sql is not None and reply.sql.result is not None:
        try:
            rows, _ = query(reply.sql.sql)
        except (Refused, duckdb.Error):
            rows = frame(reply.sql.result.columns, reply.sql.result.rows)
    turns.append({"question": question, "text": reply.text, "sources": reply.sources,
                  "sql": reply.sql.sql if reply.sql else None, "problems": reply.problems,
                  "rows": rows})  # fmt: skip


def show_turn(turn: dict[str, Any]) -> None:
    st.chat_message("user").write(turn["question"])
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(
                f"The agent could not answer: {turn['error']}. It needs Ollama, Qdrant with an "
                "index and the prediction API (`make services-up`, `make index`)."
            )
            return
        st.markdown(turn["text"])
        for source in turn["sources"]:
            st.caption(source)
        for problem in turn["problems"]:
            st.warning(f"Unverified: {problem}")
        if turn["sql"]:
            with st.expander("The query"):
                st.code(turn["sql"], language="sql")


def map_layers() -> list[Any]:
    """The layers the viewer picked, and the answer put on the map, if any."""
    names = [layer.name for layer in explore.layers]
    first = [next((lay.name for lay in explore.layers if lay.kind == kind), None)
             for kind in ("areas", "points")]  # fmt: skip
    shown = st.multiselect("Layers", names, default=[name for name in first if name])
    drawn = []
    for layer in explore.layers:
        if layer.name not in shown:
            continue
        try:
            rows, cut = query(layer.sql)
        except (Refused, duckdb.Error) as failed:
            st.warning(f"{layer.name}: {str(failed).strip().splitlines()[0]} (run `make data`?)")
            continue
        drawn += draw_on_map(layer.name, layer.kind, rows)
        more = " - more rows than the map draws" if cut else ""
        st.caption(f"**{layer.name}**: {layer.description}{more}")
    if "on_map" in st.session_state:
        chart, rows, title = st.session_state["on_map"]
        drawn += draw_on_map(title, chart.kind, rows, chart)
        st.caption(f"**On the map from an answer**: {title}")
    return drawn


def draw_on_map(name: str, kind: str, rows: pl.DataFrame, chart: Chart | None = None) -> list[Any]:
    chart = chart or infer_chart(rows, areas())
    if kind == "areas":
        known = areas()
        if known is None or chart.y is None:
            st.warning(f"{name}: the areas are not built yet, or the rows name none")
            return []
        return [area_layer(known, rows, chart.y, name)]
    if chart.kind != "points":
        st.warning(f"{name}: the rows have no latitude and longitude to put on the map")
        return []
    layer, legend = point_layer(rows, chart.color, name)
    if legend:
        swatches = "&nbsp;&nbsp;".join(
            f'<span style="color: rgb{rgb}">&#9679;</span> {html.escape(value)}'
            for value, rgb in legend.items()
        )
        st.html(f'<div style="font-size: 0.85rem">{html.escape(name)}: {swatches}</div>')
    return [layer]


def chart_of(turn: dict[str, Any], index: int) -> None:
    """The latest answer's rows as a chart the viewer can change, checked before drawn."""
    rows: pl.DataFrame = turn["rows"]
    inferred = infer_chart(rows, areas())
    columns = [NONE, *rows.columns]
    st.subheader(f"Chart: {turn['question']}")
    kind_box, x_box, y_box, color_box = st.columns(4)
    kind = kind_box.selectbox("Kind", KINDS, index=KINDS.index(inferred.kind), key=f"kind{index}")
    picked = {
        role: box.selectbox(
            role,
            columns,
            key=f"{role}{index}",
            index=columns.index(getattr(inferred, role) or NONE),
        )
        for role, box in (("x", x_box), ("y", y_box), ("color", color_box))
    }
    chart = Chart(kind=kind, **{r: None if v == NONE else v for r, v in picked.items()})  # type: ignore[arg-type]
    problems = check_chart(chart, rows, areas())
    for problem in problems:
        st.warning(problem)
    if not problems:
        spec = vega_lite(chart, rows, areas())
        if spec is None:
            st.dataframe(rows)
        else:
            st.vega_lite_chart(spec, width="stretch")
        if chart.kind in ("points", "areas") and st.button("Put it on the map", key=f"map{index}"):
            st.session_state["on_map"] = (chart, rows, turn["question"])
            st.rerun()
    with st.expander(f"The rows ({rows.height})"):
        st.dataframe(rows)


st.title(explore.title)
st.caption(
    "The map draws the domain's layers; ask the agent a question and its answer comes with "
    "the query behind it, drawn as a chart you can change."
)
map_column, chat_column = st.columns([3, 2], gap="large")
with map_column:
    st.pydeck_chart(deck(explore.view, map_layers()), height=MAP_HEIGHT)
with chat_column:
    turns = st.session_state.setdefault("turns", [])
    for turn in turns:
        show_turn(turn)
    if not turns and explore.examples:
        st.caption("Try one of these:")
        for number, example in enumerate(explore.examples):
            if st.button(example, key=f"example{number}"):
                ask(example)
                st.rerun()
    question = st.chat_input("Ask about the data, the models or the documents (in English)")
    if question:
        ask(question)
        st.rerun()

charted = [
    (i, t) for i, t in enumerate(st.session_state.get("turns", [])) if t.get("rows") is not None
]
if charted:
    st.divider()
    chart_of(charted[-1][1], charted[-1][0])
