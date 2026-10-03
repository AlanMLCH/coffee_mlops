"""Which models a question may be asking for, found the way the documents are: by retrieval.

The router used to be told every model of the domain in one list. A small model reads a
list, not a catalogue: offered nine models instead of three, it routed two of ten
prediction questions to the tables, and a list that grows with the domain cannot be the
prompt. So each model the API serves has a card - what it predicts, and the inputs its
request takes, read from the request body the API validates - embedded once. A question
is embedded and set against the cards, and the steps that choose a tool (the router, the
second opinion, the plan, the choice of model) are shown the closest few only, each with
its inputs: a question that states a model's inputs is easier to tell apart from one that
asks the tables when the inputs are in front of the model that decides.

Only models the API serves get a card: one the gate never let through answers 503, and
offering it only invites the call.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx
import numpy as np

from mlops_core.adapter import DomainAdapter

# The closest cards a question is shown: the two most similar. One would leave a
# question between two models no choice; the whole list is what failed.
SHOWN_MODELS = 2
# How a question is embedded for this search (qwen3-embedding takes the task as an
# instruction): a different task from finding the passage that answers it.
MODEL_TASK = "Given a question, retrieve the predictive model whose estimate answers it"

Embed = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class ModelCard:
    """What a model is told by: its name, what it predicts, and what it is asked with."""

    name: str
    description: str
    inputs: list[str]  # the request body's fields, required ones first

    def listing(self) -> str:
        """The card as a prompt lists it."""
        return f"  - {self.name}: {self.description} Asked with: {', '.join(self.inputs)}."

    @property
    def text(self) -> str:
        """What is embedded: the description and the inputs, as words."""
        inputs = ", ".join(field.replace("_", " ") for field in self.inputs)
        return f"{self.description} Inputs: {inputs}."


def model_cards(adapter: DomainAdapter, names: Sequence[str]) -> list[ModelCard]:
    """A card per named model, in the order given."""
    cards = []
    for name in names:
        body = adapter.request_model(name)
        fields = body.model_fields
        required = [field for field, info in fields.items() if info.is_required()]
        optional = [field for field in fields if field not in required]
        description = adapter.config.model_named(name).description
        cards.append(ModelCard(name, description, required + optional))
    return cards


def served_models(adapter: DomainAdapter, api: httpx.Client) -> list[str]:
    """The models the API serves now, among those the domain offers its agent; every one
    it offers when the API cannot be asked (the call will say why, question by question)."""
    agent = adapter.config.agent
    offered = [model.name for model in agent.shown_models(adapter.config.models)]
    try:
        response = api.get("/health")
        response.raise_for_status()
        versions = response.json()["models"]
    except (httpx.HTTPError, KeyError, ValueError):
        return offered
    return [name for name in offered if versions.get(name) is not None]


def model_query(question: str) -> str:
    return f"Instruct: {MODEL_TASK}\nQuery:{question}"


class ModelFinder:
    """The cards closest to a question, most similar first."""

    def __init__(self, cards: list[ModelCard], embed: Embed, shown: int = SHOWN_MODELS):
        self.cards, self._embed, self._shown = cards, embed, shown
        self._vectors = embed([card.text for card in cards]) if len(cards) > shown else None

    def closest(self, question: str) -> list[ModelCard]:
        if self._vectors is None:  # no more cards than are shown: all of them, as declared
            return list(self.cards)
        similarity = self._vectors @ self._embed([model_query(question)])[0]
        return [self.cards[i] for i in np.argsort(-similarity)[: self._shown]]

    def listing(self, question: str) -> str:
        return "\n".join(card.listing() for card in self.closest(question))
