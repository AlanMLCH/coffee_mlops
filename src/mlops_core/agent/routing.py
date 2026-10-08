"""Which tool answers a question: the tables, a model, the documents, or more than one.

The router is told what each tool covers in the domain's own terms, read from what the
domain already declares: the headings of its data dictionary, the description of each
of its models, and the topics of its corpus. A new domain gets a router without writing
one.
"""

from collections.abc import Collection

from mlops_core.agent.dictionary import offered
from mlops_core.agent.prompts import ROUTER, Route, RouteReply
from mlops_core.agent.text_to_sql import Generator
from mlops_core.config import DomainConfig


def routing_context(
    config: DomainConfig,
    dictionary: str,
    views: Collection[str],
    served: Collection[str] | None = None,
) -> dict[str, str]:
    """What the router prompt says about each tool, for this domain. `served` names the
    models the API serves now: a model with no champion is not offered - routed to, it
    could only refuse, and its description would draw questions the tables answer."""
    models = [
        m for m in config.agent.shown_models(config.models) if served is None or m.name in served
    ]
    headings = [
        "  - " + section.splitlines()[0].removeprefix("## ").replace("`", "")
        for section in offered(dictionary, views).values()
    ]
    topics = config.corpus.topics if config.corpus else {}
    return {
        "subject": config.name,
        "tables": "\n".join(headings),
        "models": "\n".join(f"  - {m.name}: {m.description}" for m in models),
        "topics": "\n".join(f"  - {name}: {t.description}" for name, t in topics.items()),
        "studies": "",  # the studies closest to a question, set per question when shown
    }


def route(generator: Generator, context: dict[str, str], question: str) -> Route:
    # No studies unless the context shows some: a context built before they existed reads.
    shown = {"studies": ""} | dict(context)
    return generator.ask(ROUTER.format(question=question, **shown), RouteReply).route
