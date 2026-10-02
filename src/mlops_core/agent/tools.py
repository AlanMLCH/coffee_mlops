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


def unstated_dates(body: type[BaseModel], request: dict[str, Any], question: str) -> dict[str, Any]:
    """The request without the optional dates the question never states.

    Told a date field defaults to today, the 4B filled it anyway, with a day of its own -
    2023-10-07, across questions that named no date at all - and the model was then asked
    about a day nobody asked about. A date whose year the question does not give is the
    model's invention: it is dropped, and the API applies its default."""
    schema = body.model_json_schema()
    required = set(schema.get("required", ()))
    kept = dict(request)
    for name, spec in schema["properties"].items():
        is_date = any(s.get("format") == "date" for s in spec.get("anyOf", [spec]))
        value = kept.get(name)
        unstated = isinstance(value, str) and value[:4] not in question
        if is_date and name not in required and unstated:
            del kept[name]
    return kept


def described(response: dict[str, Any]) -> str:
    """A prediction in words, its range and its level included: what the answer is
    written from. A 4B does not multiply a price by a percent change reliably, so the
    level comes computed."""
    text = f"{response['target']} = {response['prediction']:.2f}"
    if response.get("lower") is not None and response.get("upper") is not None:
        text += (
            f", between {response['lower']:.2f} and {response['upper']:.2f} "
            f"{response['coverage']:.0%} of the time"
        )
    level = response.get("level")
    if level:
        text += f"; in {level['of']}, from {level['now']:.2f} now to {level['prediction']:.2f}"
        if level.get("lower") is not None and level.get("upper") is not None:
            text += f" (between {level['lower']:.2f} and {level['upper']:.2f})"
    return text


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
    generator: Generator,
    adapter: DomainAdapter,
    api: httpx.Client,
    question: str,
    asked: str | None = None,
) -> PredictionAnswer:
    """Choose the model for `question`, describe the item, and ask the API. The item is
    read from `asked` - the words of whoever asked, when `question` is a plan's part of
    them: a plan restates its part, and a detail it drops is a field left empty (a bag's
    shop, once, and the price came back for no shop at all)."""
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
        description=config.model_named(name).description,
        fields=fields(body),
        question=asked or question,
    )
    item = generator.ask(prompt, body)
    request = unstated_dates(
        body, item.model_dump(mode="json", exclude_none=True), asked or question
    )
    try:
        response = api.post(f"/models/{name}/predict", json=request)
        response.raise_for_status()
    except httpx.HTTPError as failed:
        return PredictionAnswer(name, request, None, f"The prediction service failed: {failed}")
    return PredictionAnswer(name, request, response.json(), None)
