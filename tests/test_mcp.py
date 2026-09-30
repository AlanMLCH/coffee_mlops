"""What the MCP server offers beyond the agent's tools: the agent's notes on a query, the
explorer's slices and map layers without SQL, and the project's findings, studies, model
cards and freshness as resources - over a small built domain, called in process."""

import asyncio
import json
from pathlib import Path
from typing import Any

import duckdb
import httpx
import polars as pl
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import domains.coffee
from mlops_core.agent.mcp_server import build_server
from mlops_core.agent.sql import read_only
from mlops_core.explore.charts import Areas
from mlops_core.storage import write_table

SQUARE = {"type": "Polygon", "coordinates": [[[-99.2, 19.3], [-99.1, 19.3], [-99.1, 19.4],
                                              [-99.2, 19.3]]]}  # fmt: skip
AREAS = Areas("borough_id", "borough", {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"id": "09015", "name": "Cuauhtémoc"}, "geometry": SQUARE}
]})  # fmt: skip
SHOPS = "Coffee shops and what else the register calls a cafeteria"


@pytest.fixture
def built(tmp_path: Path) -> Path:
    """Shops, boroughs, a study, a model's studies and the monitor's verdict on it."""
    data = tmp_path / "coffee"
    shops = pl.DataFrame({
        "name": ["Café A", "Jugos B", "Café C", "Nieves D"],
        "source": ["denue", "denue", "osm", "denue"],
        "borough": ["Cuauhtémoc", "Cuauhtémoc", None, "Cuauhtémoc"],
        "borough_id": ["09015", "09015", None, "09015"],
        "latitude": [19.35, 19.36, 19.37, 19.38],
        "longitude": [-99.15, -99.15, -99.14, -99.13],
        "kind": ["coffee", "juice", "coffee", "ice_cream"],
    })  # fmt: skip
    write_table(shops, data / "clean" / "coffee_shops", {"denue_cafes": "ingested_at=x"})
    write_table(pl.DataFrame({"borough_id": ["09015"], "borough": ["Cuauhtémoc"],
                              "area_km2": [32.4], "population": [545884]}),
                data / "clean" / "boroughs", {})  # fmt: skip
    write_table(pl.DataFrame({"state": ["Chiapas", "Veracruz"], "share_pct": [36.6, 24.9]}),
                data / "analysis" / "production_by_state", {})  # fmt: skip
    write_table(pl.DataFrame({"period": ["cqi_2018"], "mean": [82.2]}),
                data / "analysis" / "review_target_distribution",
                {"review_features": "built_at=20260929T000000Z"})  # fmt: skip
    verdict = data / "monitoring" / "review_drift" / "built_at=20260929T000000Z"
    verdict.mkdir(parents=True)
    (verdict / "verdict.json").write_text(
        json.dumps({"model": "review", "current": "cqi_2023", "retrain": False,
                    "reasons": ["the target drifted"], "data_version": "abc"}),
        encoding="utf-8",
    )  # fmt: skip
    (verdict / "manifest.json").write_text("{}", encoding="utf-8")
    return data


def server(data: Path, areas: Areas | None = AREAS) -> Any:
    return build_server(
        domains.coffee.adapter(),
        read_only(data),
        "## `clean.coffee_shops` — one place",
        lambda question, k: [],
        httpx.Client(base_url="http://api.test"),
        lambda passage: "",
        areas,
        data,
        "http://localhost:8502",
    )


def call(built: Any, tool: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(built.call_tool(tool, arguments))


# --- The agent's notes on a query -----------------------------------------------------------


def test_a_query_comes_back_with_what_the_agent_would_be_told(built: Path) -> None:
    served = server(built)

    everything = "SELECT count(*) AS n FROM clean.coffee_shops"
    unguarded = call(served, "query_tables", {"sql": everything})
    guarded = call(served, "query_tables", {
        "sql": "SELECT count(*) AS n FROM clean.coffee_shops WHERE kind = 'coffee'"})  # fmt: skip
    misspelt = call(served, "query_tables", {
        "sql": "SELECT name FROM clean.coffee_shops WHERE kind = 'coffe'"})  # fmt: skip
    none = call(served, "query_tables", {
        "sql": "SELECT name FROM clean.coffee_shops WHERE kind = 'coffee' AND false"})  # fmt: skip

    assert "juice stands" in unguarded.structured_content["notes"][0]
    assert guarded.structured_content["notes"] == [] and guarded.structured_content["rows"] == [[2]]
    assert misspelt.structured_content["notes"] == [
        "kind = 'coffe' matches no row of clean.coffee_shops; its values include: coffee, "
        "ice_cream, juice"
    ]
    assert none.structured_content["notes"] == ["The query found no rows."]


# --- The explorer, without SQL ----------------------------------------------------------------


def test_a_curated_slice_needs_no_sql_and_links_to_the_explorer(built: Path) -> None:
    served = server(built)

    sliced = call(served, "explore_segment", {
        "dataset": SHOPS, "measure": "places", "by": "borough", "color": "kind",
        "filters": {"source": ["denue"]}})  # fmt: skip

    found = sliced.structured_content
    assert "source IN ('denue')" in found["sql"]
    assert found["columns"] == ["borough", "kind", "places"]
    assert sorted(found["rows"]) == [["Cuauhtémoc", "coffee", 1], ["Cuauhtémoc", "ice_cream", 1],
                                     ["Cuauhtémoc", "juice", 1]]  # fmt: skip
    assert found["summary"].startswith("Highest places: Cuauhtémoc")
    assert found["explorer_url"] == (
        "http://localhost:8502/?view=segments&dataset=Coffee+shops+and+what+else+the+register"
        "+calls+a+cafeteria&measure=places&by=borough&color=kind"
    )
    tool = next(t for t in asyncio.run(served.list_tools()) if t.name == "explore_segment")
    assert f"- {SHOPS}: measures places; split or colour by borough" in tool.description
    assert tool.output_schema is not None


def test_a_slice_names_its_empty_values_and_refuses_what_the_dataset_lacks(built: Path) -> None:
    served = server(built)

    sliced = call(served, "explore_segment", {"dataset": SHOPS, "measure": "places",
                                              "by": "borough"})  # fmt: skip

    assert ["(no value)", 1] in sliced.structured_content["rows"]
    with pytest.raises(ToolError, match="No dataset 'Tea'; there are"):
        call(served, "explore_segment", {"dataset": "Tea", "measure": "x", "by": "y"})
    with pytest.raises(ToolError, match="has no measure 'price'"):
        call(served, "explore_segment", {"dataset": SHOPS, "measure": "price", "by": "kind"})


def test_a_map_layer_is_drawn_with_what_it_shows(built: Path) -> None:
    served = server(built)

    places = call(served, "map_layer", {"name": "Coffee shops"})
    density = call(served, "map_layer", {"name": "Coffee shops per km² (DENUE)"})

    image, about = places.content
    told = json.loads(about.text)
    assert image.mime_type == "image/png"
    assert told["chart"]["kind"] == "points" and told["rows"] == 2
    assert told["explorer_url"] == "http://localhost:8502/?view=map&layer=Coffee+shops"
    assert json.loads(density.content[1].text)["chart"] == {
        "kind": "areas", "y": "coffee_shops_per_km2", "title": "Coffee shops per km² (DENUE)"
    }  # fmt: skip
    with pytest.raises(ToolError, match="No map layer 'Tea'"):
        call(served, "map_layer", {"name": "Tea"})
    with pytest.raises(ToolError, match="areas need a column naming the area"):
        call(server(built, areas=None), "map_layer", {"name": "Coffee shops per km² (DENUE)"})


def test_a_large_result_s_chart_comes_without_its_spec(built: Path, tmp_path: Path) -> None:
    served = server(built)

    drawn = call(served, "draw", {"sql": "SELECT range AS x, range * 2 AS y FROM range(600)"})

    assert "vega_lite" not in json.loads(drawn.content[1].text)


# --- Resources and prompts --------------------------------------------------------------------


def test_the_findings_studies_models_and_freshness_are_resources(built: Path) -> None:
    served = server(built)

    def read(uri: str) -> str:
        return str(next(iter(asyncio.run(served.read_resource(uri)))).content)

    listed = {str(r.uri) for r in asyncio.run(served.list_resources())}
    templates = {t.uri_template for t in asyncio.run(served.list_resource_templates())}
    assert {"dictionary://tables", "findings://all", "studies://index",
            "status://freshness"} <= listed  # fmt: skip
    assert {"studies://{name}", "models://{name}"} <= templates
    assert "## A kilogram, from the farm to the shelf" in read("findings://all")
    assert read("studies://index").splitlines() == ["production_by_state",
                                                    "review_target_distribution"]  # fmt: skip
    assert read("studies://production_by_state").splitlines()[0] == "state,share_pct"
    card = json.loads(read("models://review"))
    assert card["target"] == "total_cup_points" and card["tool"] == "predict_review"
    assert card["monitor"]["reasons"] == ["the target drifted"]
    assert card["studies_built_from"]["review_features"] == "built_at=20260929T000000Z"
    fresh = json.loads(read("status://freshness"))
    assert fresh["clean.coffee_shops"]["inputs"] == {"denue_cafes": "ingested_at=x"}
    with pytest.raises(Exception, match="studies://nope"):
        read("studies://nope")
    with pytest.raises(Exception, match="models://nope"):
        read("models://nope")


def test_a_model_card_without_the_data_dir_says_only_what_the_config_does(built: Path) -> None:
    served = build_server(domains.coffee.adapter(), read_only(built), "", lambda q, k: [],
                          httpx.Client(), lambda p: "")  # fmt: skip

    card = json.loads(next(iter(asyncio.run(served.read_resource("models://offer")))).content)

    assert "monitor" not in card and card["target"] == "price_mxn_per_kg"
    assert json.loads(next(iter(asyncio.run(served.read_resource("status://freshness"))))
                      .content) == {}  # fmt: skip
    # Pointed at a folder the tables are not in: nothing to say about any of them.
    elsewhere = build_server(
        domains.coffee.adapter(), read_only(built), "", lambda q, k: [], httpx.Client(),
        lambda p: "", None, built.parent / "other",
    )  # fmt: skip
    fresh = next(iter(asyncio.run(elsewhere.read_resource("status://freshness")))).content
    assert json.loads(fresh) == {}


def test_each_finding_is_a_prompt_to_reproduce_it(built: Path) -> None:
    served = server(built)

    prompts = {p.name for p in asyncio.run(served.list_prompts())}
    got = asyncio.run(served.get_prompt("finding_a_kilogram_from_the_farm_to_the_shelf", {}))

    assert "finding_who_grows_mexico_s_coffee" in prompts
    text = got.messages[0].content.text
    assert "SELECT step, mxn_per_kg FROM analysis.price_ladder" in text
    assert text.startswith("Reproduce this finding")


def test_a_domain_without_an_explorer_has_neither_slices_nor_maps(built: Path) -> None:
    adapter = domains.coffee.adapter()
    bare = type(adapter)(adapter.config.model_copy(update={"explore": None}))

    served = build_server(bare, read_only(built), "", lambda q, k: [], httpx.Client(),
                          lambda p: "")  # fmt: skip
    tools = {t.name for t in asyncio.run(served.list_tools())}

    assert "explore_segment" not in tools and "map_layer" not in tools
    assert "findings://all" not in {str(r.uri) for r in asyncio.run(served.list_resources())}


def test_a_link_is_left_out_when_there_is_no_explorer_to_open(built: Path) -> None:
    served = build_server(domains.coffee.adapter(), read_only(built), "", lambda q, k: [],
                          httpx.Client(), lambda p: "", AREAS)  # fmt: skip

    sliced = call(served, "explore_segment", {"dataset": SHOPS, "measure": "places", "by": "kind"})

    assert sliced.structured_content["explorer_url"] is None


def test_the_same_session_answers_calls_at_once(built: Path) -> None:
    """Every query takes a cursor of its own: no lock, and no call waits for another's."""
    from concurrent.futures import ThreadPoolExecutor

    served = server(built)

    sql = {"sql": "SELECT count(*) FROM clean.coffee_shops WHERE kind IS NOT NULL"}
    with ThreadPoolExecutor(8) as pool:
        counts = list(pool.map(lambda _: call(served, "query_tables", sql), range(8)))

    assert {c.structured_content["rows"][0][0] for c in counts} == {4}


def test_a_query_that_fails_is_a_tool_error(built: Path) -> None:
    with pytest.raises(ToolError, match="Only SELECT may run"):
        call(server(built), "query_tables", {"sql": "DROP VIEW clean.boroughs"})
    with pytest.raises(ToolError):
        call(server(built), "query_tables", {"sql": "SELECT nope FROM clean.boroughs"})
    assert isinstance(duckdb.__version__, str)
