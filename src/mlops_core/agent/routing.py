"""Which tool answers a question: the tables, a model, the documents, or more than one.

The router is told what each tool covers in the domain's own terms, read from what the
domain already declares: the headings of its data dictionary, the description of each
of its models, and the topics of its corpus. A new domain gets a router without writing
one.
"""

from collections.abc import Collection

from mlops_core.agent.dictionary import table_sections
from mlops_core.agent.prompts import ROUTER, Route, RouteReply
from mlops_core.agent.text_to_sql import Generator
from mlops_core.config import DomainConfig


def routing_context(
    config: DomainConfig, dictionary: str, views: Collection[str]
) -> dict[str, str]:
    """What the router prompt says about each tool, for this domain."""
    headings = [
        "  - " + section.splitlines()[0].removeprefix("## ").replace("`", "")
        for view, section in table_sections(dictionary).items()
        if view in views
    ]
    topics = config.corpus.topics if config.corpus else {}
    return {
        "subject": config.name,
        "tables": "\n".join(headings),
        "models": "\n".join(f"  - {m.name}: {m.description}" for m in config.models),
        "topics": "\n".join(f"  - {name}: {t.description}" for name, t in topics.items()),
    }


def route(generator: Generator, context: dict[str, str], question: str) -> Route:
    return generator.ask(ROUTER.format(question=question, **context), RouteReply).route
