"""Which models a question is shown: the cards of the models the API serves, retrieved by
how close each is to the question, each with the inputs its request takes."""

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
import pytest

import domains.coffee
from mlops_core.agent.graph import Agent
from mlops_core.agent.model_cards import (
    SHOWN_MODELS,
    ModelCard,
    ModelFinder,
    model_cards,
    model_query,
    served_models,
)
from mlops_core.agent.tools import literal_values, predict
from mlops_core.config import DomainConfig
from tests.test_agent import PASSAGE, PREDICTED, Scripted, api, session  # noqa: F401

# Words that pull a text toward one axis each: a stand-in embedder whose geometry is known.
AXES = ["shelf", "household", "zone"]


def embed(texts: Sequence[str]) -> np.ndarray:
    """A unit vector per text, along the axes whose word it holds."""
    vectors = np.array(
        [[float(word in text.casefold()) for word in AXES] + [0.01] for text in texts]
    )
    norms: np.ndarray = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def cards() -> list[ModelCard]:
    return [
        ModelCard("review", "A lot's cup score.", ["country"]),
        ModelCard("shelf_price", "A jar's fair price on a shelf.", ["brand", "grams"]),
        ModelCard("households", "Whether a household buys coffee.", ["state", "members"]),
        ModelCard("zones", "Coffee shops a zone should have.", ["zone_id"]),
    ]


def test_a_card_is_read_from_the_models_request_body_required_inputs_first() -> None:
    adapter = domains.coffee.adapter()

    shelf, households = model_cards(adapter, ["shelf_price", "households"])

    assert shelf.inputs[:3] == ["brand", "grams", "product"]  # the required ones
    assert "observed_on" in shelf.inputs
    assert households.listing().startswith("  - households: The probability that a Mexican")
    assert households.listing().endswith(
        "Asked with: state, members, income_month_mxn, grows_coffee."
    )
    assert "income month mxn" in households.text  # embedded as words


def test_a_question_is_shown_the_closest_cards_most_similar_first() -> None:
    finder = ModelFinder(cards(), embed)

    shown = finder.closest("What is a fair price for a jar on a shelf, for a household?")

    assert len(shown) == SHOWN_MODELS
    assert {card.name for card in shown} == {"shelf_price", "households"}
    assert "zones" not in finder.listing("A jar on a shelf?")
    assert model_query("Q?").startswith("Instruct: ") and model_query("Q?").endswith("Query:Q?")


def test_with_no_more_cards_than_are_shown_every_one_is_shown_and_nothing_embedded() -> None:
    def never(texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("nothing to rank")

    finder = ModelFinder(cards()[:2], never)

    assert [card.name for card in finder.closest("Anything?")] == ["review", "shelf_price"]


def test_only_the_models_the_api_serves_get_a_card() -> None:
    adapter = domains.coffee.adapter()

    def health(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "partial", "models": {"review": "1", "zones": "2", "auction": None}},
        )

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    served = served_models(adapter, api(health))
    every = served_models(adapter, api(down))

    assert served == ["review", "zones"]
    assert every == [model.name for model in adapter.config.models]  # it cannot ask: all


def test_the_router_and_the_prediction_are_shown_only_the_closest_models(
    session: Any,  # noqa: F811
) -> None:
    generator = Scripted(
        RouteReply=lambda p: {"route": "prediction"},
        Household=lambda p: {"state": "Veracruz", "members": 3, "income_month_mxn": 15000},
        AnswerReply=lambda p: {"text": "Its chance is 0.16 [prediction].", "citations": []},
    )
    finder = ModelFinder(
        model_cards(domains.coffee.adapter(), ["review", "households", "zones"]),
        lambda texts: np.array(
            [[1.0, 0.0] if "household" in t.casefold() else [0.0, 1.0] for t in texts]
        ),
        shown=1,
    )
    agent = Agent(
        generator,
        domains.coffee.adapter(),
        session,
        "## `clean.mexico_production` — coffee grown",
        {"subject": "coffee", "tables": "", "models": "  - review: a cup score", "topics": ""},
        lambda question, k: [PASSAGE],
        api(lambda request: httpx.Response(200, json=PREDICTED)),
        {},
        models=finder,
    )

    reply = agent.ask("How likely is a household of 3 in Veracruz on 15,000 a month to buy it?")

    routed = generator.asked("RouteReply")[0]
    assert "  - households: The probability" in routed and "review" not in routed
    assert "Asked with: state, members" in generator.asked("NeedsReply")[0]
    # One model shown: no choice to make, and the item is read in its body's schema.
    assert generator.asked("ModelChoice") == [] and reply.prediction is not None
    assert reply.prediction.model == "households"
    assert reply.prediction.request["members"] == 3


@pytest.mark.parametrize("shown", [None, []])
def test_without_a_finder_or_a_served_model_the_agent_still_answers(
    session: Any,  # noqa: F811
    shown: list[str] | None,
) -> None:
    """No finder: the router's list, as before. No model served: the prediction says so."""
    generator = Scripted(
        RouteReply=lambda p: {"route": "prediction"},
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya"},
        AnswerReply=lambda p: {"text": "84 points [prediction].", "citations": []},
    )
    finder = None if shown is None else ModelFinder([], embed)
    agent = Agent(
        generator,
        domains.coffee.adapter(),
        session,
        "",
        {"subject": "coffee", "tables": "", "models": "  - review: a cup score", "topics": ""},
        lambda question, k: [],
        api(lambda request: httpx.Response(200, json=PREDICTED)),
        {},
        models=finder,
    )

    reply = agent.ask("A Kenyan lot?")

    assert reply.prediction is not None
    if shown is None:
        assert "  - review: a cup score" in generator.asked("RouteReply")[0]
    else:
        assert reply.prediction.error == "No model the API serves answers it"
        assert not reply.answered


def test_a_key_the_schema_gives_the_form_of_is_read_off_the_question() -> None:
    """The 4B wrote "null" for a 13-digit key the question gave; the body's own pattern
    finds it. Two keys in the question name neither."""
    adapter = domains.coffee.adapter()
    respond = api(lambda request: httpx.Response(200, json=PREDICTED))
    lost = Scripted(Zone=lambda p: {"zone_id": "null"})  # what the 4B wrote

    found = predict(lost, adapter, respond, "Coffee shops in AGEB 0901600010482?", shown=["zones"])
    none = predict(lost, adapter, respond, "Coffee shops in my block?", shown=["zones"])
    place = Scripted(Place=lambda p: {"name": "Antojitos Mary"})

    named = predict(
        place, adapter, respond, "Antojitos Mary, AGEB 0901500010010?", shown=["shop_kind"]
    )

    assert found.request == {"zone_id": "0901600010482"} and found.response is not None
    assert none.response is None and none.error is not None
    assert none.error.startswith("The item did not fit zones's request: zone_id")
    assert named.request == {"name": "Antojitos Mary", "zone_id": "0901500010010"}
    assert literal_values(adapter.request_model("zones"), "0901600010482 or 0901500010010") == {}


def test_the_router_is_told_only_the_models_the_api_serves(coffee_config: DomainConfig) -> None:
    """A model with no champion is not offered: routed to, it could only refuse, and its
    description would draw questions the tables answer."""
    from mlops_core.agent.routing import routing_context

    served = routing_context(coffee_config, "", set(), served=["zones"])["models"]
    every = routing_context(coffee_config, "", set())["models"]

    assert served.startswith("  - zones:") and "review:" not in served
    assert "review:" in every and "zones:" in every
