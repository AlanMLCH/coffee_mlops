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
from pydantic import BaseModel, create_model

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


def fields(body: type[BaseModel]) -> str:
    """A request body's fields as the model is told them: name, type, whether required,
    and the description. Ollama constrains the reply to the schema but never shows it to
    the model, so a description that gives a field's vocabulary reaches it only here.
    Measured: without it, 10 of 13 predictions arrived with a field as the question worded
    it - a nationality for a country, a hyphen for an underscore, a place left out - which
    the model takes for a category it never saw, or never gets."""
    schema = body.model_json_schema()
    required = set(schema.get("required", ()))
    lines = []
    for name, spec in schema["properties"].items():
        kinds = [s.get("format", s.get("type")) for s in spec.get("anyOf", [spec])]
        kind = "/".join(k for k in kinds if k and k != "null")
        note = f": {spec['description']}" if "description" in spec else ""
        lines.append(f"- {name} ({kind}{', required' if name in required else ''}){note}")
    return "\n".join(lines)


def predict(
    generator: Generator, adapter: DomainAdapter, api: httpx.Client, question: str
) -> PredictionAnswer:
    """Choose the model, describe the item, and ask the API."""
    config = adapter.config
    names = Enum("ModelName", {model.name: model.name for model in config.models})  # type: ignore[misc]
    choice = create_model("ModelChoice", model=(names, ...))
    # A model's name need not say what it predicts: its target does. Measured: without
    # it, a question about one model's target was sent to the other model.
    listing = "\n".join(
        f"- {model.name} (predicts {model.spec.target}): {model.description}"
        for model in config.models
    )
    chosen = generator.ask(CHOOSE_MODEL.format(models=listing, question=question), choice)
    name = str(chosen.model.value)  # type: ignore[attr-defined]

    body = adapter.request_model(name)
    prompt = DESCRIBE_ITEM.format(
        description=config.model_named(name).description, fields=fields(body), question=question
    )
    described = generator.ask(prompt, body)
    request = described.model_dump(mode="json", exclude_none=True)
    try:
        response = api.post(f"/models/{name}/predict", json=request)
        response.raise_for_status()
    except httpx.HTTPError as failed:
        return PredictionAnswer(name, request, None, f"The prediction service failed: {failed}")
    return PredictionAnswer(name, request, response.json(), None)
