"""The prediction tool, and how a passage is cited.

The prediction: the model describes the item, the prediction API prices or scores it.

Two small steps rather than one large one, because a 4B model does each reliably and the
pair less so: first which of the domain's models answers the question - a choice
constrained to their names - then the item, in the model's own request body, whose
JSON schema (field descriptions included) is what Ollama constrains the reply to. The
API validates it again and answers; whatever the question did not state is left for the
API to fill, and the request is shown with the answer, so an assumption is visible.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, create_model

from mlops_core.adapter import DomainAdapter
from mlops_core.agent.model_cards import model_cards
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


# Small numbers a question spells out ("a household of three").
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve"]  # fmt: skip
SPELLED = {word: n for n, word in enumerate(NUMBER_WORDS)}


def unstated_values(
    body: type[BaseModel], request: dict[str, Any], question: str
) -> dict[str, Any]:
    """The request without the optional dates and numbers the question never states.

    The request body's schema constrains what the small model writes, not where it gets
    it. Told a date field defaults to today, the 4B filled it anyway, with a day of its own
    - 2023-10-07, across questions that named no date at all - and the model was asked
    about a day nobody asked about. A date whose year the question does not give, or a
    number it does not state, is the model's invention: an optional field holding one is
    dropped, and the API applies its default. Text is not checked this way: putting the
    question's words into a field's vocabulary ("up to five employees" as "0 a 5
    personas") is the reading asked of it. A required field is kept: the model needs it."""
    schema = body.model_json_schema()
    required = set(schema.get("required", ()))
    numbers = stated_numbers(question)
    kept = dict(request)
    for name, spec in schema["properties"].items():
        if name in required or name not in kept:
            continue
        kinds = spec.get("anyOf", [spec])
        value = kept[name]
        is_date = any(s.get("format") == "date" for s in kinds)
        is_number = any(s.get("type") in ("integer", "number") for s in kinds)
        invented_date = is_date and isinstance(value, str) and value[:4] not in question
        a_number = is_number and isinstance(value, int | float) and not isinstance(value, bool)
        if invented_date or (a_number and float(value) not in numbers):
            del kept[name]
    return kept


def literal_values(body: type[BaseModel], question: str) -> dict[str, str]:
    """The fields whose schema gives the form of their value - a key, a code - read off the
    question, where it holds exactly one string of that form. A small model copies a
    13-digit key less reliably than a pattern finds it: the 4B wrote "null" for one the
    question gave. What the question states in the schema's own form is taken as written."""
    found = {}
    for name, spec in body.model_json_schema()["properties"].items():
        for kind in spec.get("anyOf", [spec]):
            pattern = kind.get("pattern")
            if not pattern:
                continue
            bare = pattern.removeprefix("^").removesuffix("$")
            matches = set(re.findall(rf"(?<![0-9A-Za-z]){bare}(?![0-9A-Za-z])", question))
            if len(matches) == 1:
                found[name] = matches.pop()
    return found


def stated_numbers(question: str) -> set[float]:
    """Every number a question states: in figures, thousands separators aside, or spelled."""
    figures = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", question)}
    words = {float(SPELLED[w]) for w in re.findall(r"[a-z]+", question.casefold()) if w in SPELLED}
    return figures | words


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
    if str(response.get("model_gate", "")).startswith("provisional"):
        # Served because nothing passed the gate: the answer must not sound surer than that.
        text += f" [the model is provisional - {response['model_gate']}]"
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
    shown: Sequence[str] | None = None,
) -> PredictionAnswer:
    """Choose the model for `question` among those `shown` (the router's, else every one
    the domain offers), describe the item in its request body's schema, and ask the API.
    The item is read from `asked` - the words of whoever asked, when `question` is a plan's
    part of them: a plan restates its part, and a detail it drops is a field left empty (a
    bag's shop, once, and the price came back for no shop at all)."""
    config = adapter.config
    names = (
        list(shown)
        if shown is not None
        else [model.name for model in config.agent.shown_models(config.models)]
    )
    if not names:
        return PredictionAnswer("", {}, None, "No model the API serves answers it")
    if len(names) == 1:  # one model shown: there is nothing to choose
        name = names[0]
    else:
        choice = create_model(
            "ModelChoice",
            model=(Enum("ModelName", {n: n for n in names}), ...),
        )
        # A model's name need not say what it predicts: its target does. Measured: without
        # it, a question about one model's target was sent to the other model.
        listing = "\n".join(
            f"- {card.name} (predicts {config.model_named(card.name).spec.target}): "
            f"{card.description} Asked with: {', '.join(card.inputs)}."
            for card in model_cards(adapter, names)
        )
        chosen = generator.ask(CHOOSE_MODEL.format(models=listing, question=question), choice)
        name = str(chosen.model.value)  # type: ignore[attr-defined]

    body = adapter.request_model(name)
    prompt = DESCRIBE_ITEM.format(
        description=config.model_named(name).description,
        fields=fields(body),
        question=asked or question,
    )
    literal = literal_values(body, asked or question)
    try:
        written = generator.ask(prompt, body).model_dump(mode="json", exclude_none=True)
    except ValueError as unread:  # pydantic's ValidationError: an item its body refuses
        try:  # what the schema read off the question may be the whole item
            written = body.model_validate(literal).model_dump(mode="json", exclude_none=True)
        except ValueError:
            reason = str(unread).splitlines()
            return PredictionAnswer(name, {}, None, f"The item did not fit {name}'s request: "
                                    + " ".join(line.strip() for line in reason[1:3]))  # fmt: skip
    request = unstated_values(body, written | literal, asked or question)
    try:
        response = api.post(f"/models/{name}/predict", json=request)
        response.raise_for_status()
    except httpx.HTTPError as failed:
        return PredictionAnswer(name, request, None, f"The prediction service failed: {failed}")
    return PredictionAnswer(name, request, response.json(), None)
