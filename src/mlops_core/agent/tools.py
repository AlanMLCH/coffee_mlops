"""The prediction tool, and how a passage is cited.

The prediction: the model describes the item, the prediction API prices or scores it.

Two small steps rather than one large one, because a 4B model does each reliably and the
pair less so: first which of the domain's models answers the question - a choice
constrained to their names - then the item, in the model's own request body, whose
JSON schema (field descriptions included) is what Ollama constrains the reply to. The
API validates it again and answers; whatever the question did not state is left for the
API to fill, and the request is shown with the answer, so an assumption is visible.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from pydantic import create_model

from mlops_core.adapter import DomainAdapter
from mlops_core.agent.prompts import CHOOSE_MODEL, DESCRIBE_ITEM
from mlops_core.agent.text_to_sql import Generator


@dataclass(frozen=True)
class PredictionAnswer:
    model: str
    request: dict[str, Any]  # what was sent: the item as the model understood it
    response: dict[str, Any] | None  # the API's answer
    error: str | None


def cite(passage: dict[str, Any], documents: dict[str, dict[str, Any]]) -> str:
    """Where a passage comes from, as a reader can find it: publisher, title, year, and
    the page or section. Written by the code, never by the model."""
    document = documents.get(passage["document_id"], {})
    title = document.get("title", passage["document_id"])
    year = f" ({document['year']})" if document.get("year") else ""
    where = (
        f"section '{passage['part_title']}'"
        if passage.get("part_title")
        else f"page {passage['part']}"
    )
    return f'{document.get("publisher", "")}, "{title}"{year}, {where}'.lstrip(", ")


def predict(
    generator: Generator, adapter: DomainAdapter, api: httpx.Client, question: str
) -> PredictionAnswer:
    """Choose the model, describe the item, and ask the API."""
    config = adapter.config
    names = Enum("ModelName", {model.name: model.name for model in config.models})  # type: ignore[misc]
    choice = create_model("ModelChoice", model=(names, ...))
    listing = "\n".join(f"- {model.name}: {model.description}" for model in config.models)
    chosen = generator.ask(CHOOSE_MODEL.format(models=listing, question=question), choice)
    name = str(chosen.model.value)  # type: ignore[attr-defined]

    described = generator.ask(
        DESCRIBE_ITEM.format(description=config.model_named(name).description, question=question),
        adapter.request_model(name),
    )
    request = described.model_dump(mode="json", exclude_none=True)
    try:
        response = api.post(f"/models/{name}/predict", json=request)
        response.raise_for_status()
    except httpx.HTTPError as failed:
        return PredictionAnswer(name, request, None, f"The prediction service failed: {failed}")
    return PredictionAnswer(name, request, response.json(), None)
