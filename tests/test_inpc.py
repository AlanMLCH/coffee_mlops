"""INEGI's consumer price index: read from its indicators API with the token in the path,
cleaned to a month a row, and used to state the ladder's prices in pesos of one month.
And the tables the local agent is not shown.

The API is replayed by `tests.fakes.inpc_response`.
"""

import json
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest
from matplotlib.figure import Figure
from pydantic import ValidationError

from domains.coffee.analysis import real_prices, real_prices_figure
from domains.coffee.config import CoffeeConfig
from domains.coffee.prices import clean_consumer_price_index
from domains.coffee.schemas import CONSUMER_PRICE_INDEX, INPC_RAW
from domains.coffee.sources.inpc import ingest_index, to_frame
from mlops_core.agent.dictionary import reads_any, shown
from mlops_core.agent.routing import routing_context
from mlops_core.config import AgentConfig
from mlops_core.contracts import check_contract
from mlops_core.data.api import ApiClient
from tests.fakes import INPC_OBSERVATIONS

TOKEN = "fixture-inpc-token"


@pytest.fixture
def api(client: httpx.Client, tmp_path: Path) -> ApiClient:
    return ApiClient(client=client, cache_dir=tmp_path / "cache", min_interval_s=0.0, max_age_s=0.0)


def test_the_index_is_stored_oldest_first_and_the_token_nowhere(
    api: ApiClient, coffee_config: CoffeeConfig, tmp_path: Path
) -> None:
    assert coffee_config.inpc is not None
    artifact = ingest_index(api, coffee_config.inpc, TOKEN, tmp_path / "raw")
    document = json.loads(artifact.path.read_text(encoding="utf-8"))

    periods = [o["period"] for o in document["observations"]]
    assert periods == sorted(p for p, _ in INPC_OBSERVATIONS)
    assert document["indicator"] == "910392"
    # The token is in the request's path, and in nothing the project writes.
    assert TOKEN not in artifact.manifest.url
    assert TOKEN not in artifact.path.read_text(encoding="utf-8")
    assert all(TOKEN not in path.name for path in (tmp_path / "cache").rglob("*"))
    frame = check_contract(INPC_RAW, to_frame(document))
    assert frame["value"].to_list()[-1] == pytest.approx(145.462)


def test_an_answer_for_another_series_is_refused(
    coffee_config: CoffeeConfig, tmp_path: Path
) -> None:
    def other(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Series": [{"INDICADOR": "628194", "UNIT": "1012",
                                                     "FREQ": "8", "LASTUPDATE": "",
                                                     "OBSERVATIONS": []}]})  # fmt: skip

    with httpx.Client(transport=httpx.MockTransport(other)) as http:
        api = ApiClient(client=http, cache_dir=tmp_path, min_interval_s=0.0, max_age_s=0.0)
        assert coffee_config.inpc is not None
        with pytest.raises(RuntimeError, match="628194 with no observations"):
            ingest_index(api, coffee_config.inpc, TOKEN, tmp_path / "raw")


def test_a_month_a_row_or_none_without_the_token() -> None:
    raw = pl.DataFrame({"period": ["2026/08", "1969/01"], "value": [145.462, 0.0116]})

    index = check_contract(CONSUMER_PRICE_INDEX, clean_consumer_price_index(raw))

    assert index["month"].to_list() == [date(1969, 1, 1), date(2026, 8, 1)]
    empty = clean_consumer_price_index(None)
    assert empty.is_empty() and empty.columns == ["month", "index"]


def test_prices_in_pesos_of_the_latest_month() -> None:
    """A price is moved by the index's ratio: a year by its mean, a month by its own."""
    cpi = pl.DataFrame({"month": [date(2024, 1, 1), date(2024, 7, 1), date(2026, 8, 1)],
                        "index": [100.0, 110.0, 150.0]})  # fmt: skip
    production = pl.DataFrame({"year": [2024, 2024], "value_mxn": [6000.0, 4000.0],
                               "production_t": [1.0, 1.0]})  # fmt: skip
    green = pl.DataFrame({"period": [date(2024, 1, 1)], "indicator": ["other_milds"],
                          "mxn_per_kg": [100.0]})  # fmt: skip
    prices = pl.DataFrame({"date": [date(2024, 1, 3), date(2024, 1, 20)],
                           "product": ["ground", "ground"], "sweetened": [False, True],
                           "decaf": [False, False],
                           "price_mxn_per_kg": [300.0, 900.0]})  # fmt: skip

    real = real_prices(production, green, prices, cpi)
    by_step = {row["step"]: row for row in real.iter_rows(named=True)}

    cherry = by_step["cherry at the farm gate"]  # 5 pesos a kg, a year whose mean index is 105
    assert cherry["nominal_mxn_per_kg"] == 5.0
    assert cherry["real_mxn_per_kg"] == pytest.approx(5.0 * 150 / 105)
    assert by_step["green coffee at the port"]["real_mxn_per_kg"] == pytest.approx(150.0)
    shelf = by_step["ground coffee on a shelf"]  # plain ground only: the sweetened is left out
    assert shelf["nominal_mxn_per_kg"] == 300.0
    assert set(real["pesos_of"]) == {date(2026, 8, 1)}
    assert isinstance(real_prices_figure(real), Figure)
    assert real_prices(production, green, prices, cpi.clear()).is_empty()


def test_the_agent_is_not_shown_what_the_domain_keeps_from_it(
    coffee_config: CoffeeConfig,
) -> None:
    hidden = coffee_config.agent.hidden_tables

    assert "clean.consumer_price_index" in hidden
    assert shown({"clean.boroughs", "clean.borough_profile"}, hidden) == {"clean.boroughs"}
    assert reads_any("SELECT * FROM CLEAN.Borough_Profile WHERE x", hidden)
    assert not reads_any("SELECT * FROM clean.boroughs", hidden)  # a prefix is not a name
    with pytest.raises(ValidationError, match=r"write them as schema\.table"):
        AgentConfig(hidden_tables=["borough_profile"])


def test_the_agent_is_offered_only_the_models_the_domain_shows_it(
    coffee_config: CoffeeConfig,
) -> None:
    offered = [model.name for model in coffee_config.agent.shown_models(coffee_config.models)]
    context = routing_context(coffee_config, "", set())

    assert offered == ["review", "offer", "green_price"]
    assert "  - offer: " in context["models"] and "zones" not in context["models"]
    with pytest.raises(ValidationError, match="names no model of the domain"):
        coffee_config.model_validate(
            coffee_config.model_dump() | {"agent": {"hidden_models": ["tasting"]}}
        )
