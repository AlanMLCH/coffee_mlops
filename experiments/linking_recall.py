"""Which way of choosing the dictionary's sections gives the SQL writer every table it needs?

Before measuring SQL, measure the linking: each benchmark question's reference query names
the tables an answer reads, so a linker can be judged on its own - did the sections it
chose include every one of them (recall), and how much of the dictionary did it pass on
(the prompt the SQL writer reads)? Three linkers, on the default and the held-out SQL
questions:

- `embed:k`: the k sections most similar to the question (`dictionary.SchemaLinker`), and
  the ones they name.
- `model`: the generator picks the tables from a short list of their headings, its reply
  constrained to the tables that exist, and the ones they name are added.
- `model+embed:k`: both, united.

Run with Ollama up:  uv run python experiments/linking_recall.py
"""

import re
from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, Field, create_model

from mlops_core.adapter import domain_dir, load_adapter
from mlops_core.agent.benchmark import (
    GENERATOR_OPTIONS,
    SQL_CASES_FILE,
    SqlCase,
    case_file,
    load_cases,
)
from mlops_core.agent.dictionary import SchemaLinker, dictionary_path, named_by, offered
from mlops_core.agent.graph import AGENT_GENERATOR
from mlops_core.agent.sql import read_only, views
from mlops_core.agent.text_to_sql import Generator
from mlops_core.config import Settings
from mlops_core.rag.llm import LocalModel, ollama_client
from mlops_core.rag.vectors import EMBEDDING_MODEL, QUERY_OPTIONS, query_text

# The tables a SQL query needs, picked from their headings before the query is written:
# the writer then reads only their sections (`ModelLinker`, below).
TABLES = """A SQL query will answer a question about {subject}. Which of these tables must
it read? Choose every table the query needs - both sides of a join - and no other.

{tables}

Question: {question}
"""


def heading(section: str) -> str:
    """A section's first line, without its Markdown: what one row of the table is."""
    return section.splitlines()[0].removeprefix("## ").replace("`", "")


class ModelLinker:
    """The sections a question needs, chosen by the generator itself from a list of the
    tables' headings - its reply constrained to tables that exist - and the ones those
    name. A short prompt to pick from, instead of a long one to read past."""

    def __init__(self, sections: dict[str, str], generator: Generator, subject: str):
        self._sections, self._generator, self._subject = sections, generator, subject
        names = Enum("Table", {name: name for name in sections})  # type: ignore[misc]
        self._reply: type[BaseModel] = create_model(
            "TablesReply",
            tables=(list[names], Field(min_length=1)),
        )
        self._listing = "\n".join(f"- {heading(section)}" for section in sections.values())

    def chosen(self, question: str) -> list[str]:
        prompt = TABLES.format(subject=self._subject, tables=self._listing, question=question)
        reply = self._generator.ask(prompt, self._reply)
        picked = {str(table.value) for table in reply.tables}  # type: ignore[attr-defined]
        picked |= named_by(self._sections, picked)
        return [name for name in self._sections if name in picked]

    def __call__(self, question: str) -> str:
        return "\n\n".join(self._sections[name] for name in self.chosen(question))


TABLE = re.compile(r"\b((?:clean|features|predictions|analysis|evaluations|embeddings)\.\w+)")


def needed(sql: str) -> set[str]:
    return set(TABLE.findall(sql))


def united(*linkers: Callable[[str], list[str]]) -> Callable[[str], list[str]]:
    """The sections any of the linkers chose."""
    return lambda question: sorted({name for link in linkers for name in link(question)})


def main() -> None:
    config = load_adapter("coffee").config
    home = domain_dir(config.name)
    con = read_only(Settings().data_dir / config.name)
    sections = offered(dictionary_path(home).read_text(encoding="utf-8"), views(con))
    whole = sum(len(s) for s in sections.values())
    print(f"{len(sections)} sections, {whole:,} characters in all", flush=True)
    with ollama_client(Settings().ollama_url) as http:
        generator = LocalModel(http, AGENT_GENERATOR, GENERATOR_OPTIONS)
        embedder = LocalModel(http, EMBEDDING_MODEL, QUERY_OPTIONS)
        by_model = ModelLinker(sections, generator, config.name)
        linkers: dict[str, Callable[[str], list[str]]] = {"model": by_model.chosen}
        for k in (2, 3, 4, 6):
            embedded = SchemaLinker(sections, embedder.embed, k, query_text)
            linkers[f"embed:{k}"] = embedded.chosen
        for k in (1, 2):
            linkers[f"model+embed:{k}"] = united(
                by_model.chosen, SchemaLinker(sections, embedder.embed, k, query_text).chosen
            )
        for cases in (None, "holdout"):
            questions = load_cases(case_file(home, SQL_CASES_FILE, cases), SqlCase)
            print(f"\n{cases or 'default'}: {len(questions)} questions", flush=True)
            for name, choose in linkers.items():
                hits, size, missed = 0, 0, []
                for case in questions:
                    chosen = set(choose(case.question)) | named_by(sections, set())
                    need = needed(case.sql)
                    if need <= chosen:
                        hits += 1
                    else:
                        missed.append(f"{case.id}:{','.join(sorted(need - chosen))}")
                    size += sum(len(sections[s]) for s in chosen)
                print(f"  {name:14s} recall {hits}/{len(questions)} "
                      f"({hits / len(questions):.0%}), {size / len(questions) / whole:.0%} of the "
                      f"dictionary; missed {missed[:6]}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
