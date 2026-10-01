"""Does the SQL writer do better seeing only the dictionary sections a question needs?

The whole data dictionary is most of the SQL writer's prompt (about 8,000 tokens by
September 2026), and a small model reads past what it was told in a long prompt. Schema
linking is the usual remedy in text to SQL. This measures it on the benchmark's own SQL
questions, the default set and the held-out one, with the agent's guards, each question
judged as the benchmark judges it: the whole dictionary against each linker, paired,
question by question. A linker is named as `experiments/linking_recall.py` names it:

- `embed:k`: the k sections most similar to the question, and the ones they name;
- `model+embed:k`: the tables the generator picks from their headings, united with those.

Run with Ollama up:  uv run python experiments/schema_linking.py embed:4 model+embed:1
"""

import sys
from collections.abc import Callable

import numpy as np
from linking_recall import ModelLinker  # run from experiments/, as its docstring says

from mlops_core.adapter import domain_dir, load_adapter
from mlops_core.agent.benchmark import (
    COMPARED_ROWS,
    GENERATOR_OPTIONS,
    SQL_CASES_FILE,
    SqlCase,
    case_file,
    load_cases,
    same_answer,
)
from mlops_core.agent.dictionary import SchemaLinker, dictionary_path, offered, schema_context
from mlops_core.agent.graph import AGENT_GENERATOR
from mlops_core.agent.sql import read_only, run_select, views
from mlops_core.agent.text_to_sql import write_sql
from mlops_core.config import Settings
from mlops_core.rag.llm import LocalModel, ollama_client
from mlops_core.rag.vectors import EMBEDDING_MODEL, QUERY_OPTIONS, query_text
from mlops_core.stats import compare

SETS = (None, "holdout")


def main() -> None:
    names = sys.argv[1:] or ["embed:4"]
    config = load_adapter("coffee").config
    home = domain_dir(config.name)
    con = read_only(Settings().data_dir / config.name)
    dictionary = dictionary_path(home).read_text(encoding="utf-8")
    whole = schema_context(dictionary, views(con))
    sections = offered(dictionary, views(con))
    guards = config.agent.sql_guards
    with ollama_client(Settings().ollama_url) as http:
        generator = LocalModel(http, AGENT_GENERATOR, GENERATOR_OPTIONS)
        embedder = LocalModel(http, EMBEDDING_MODEL, QUERY_OPTIONS)
        by_model = ModelLinker(sections, generator, config.name)

        def linker(name: str) -> Callable[[str], str]:
            kind, k = name.split(":")
            embedded = SchemaLinker(sections, embedder.embed, int(k), query_text)
            if kind == "embed":
                return embedded

            def both(question: str) -> str:
                chosen = set(by_model.chosen(question)) | set(embedded.chosen(question))
                return "\n\n".join(text for view, text in sections.items() if view in chosen)

            return both

        for cases in SETS:
            questions = load_cases(case_file(home, SQL_CASES_FILE, cases), SqlCase)

            def right(
                schema_for: Callable[[str], str] | None, questions: list[SqlCase] = questions
            ) -> list[float]:
                verdicts = []
                for case in questions:
                    schema = schema_for(case.question) if schema_for else whole
                    answer = write_sql(generator, con, schema, case.question, COMPARED_ROWS, guards)
                    expected = run_select(con, case.sql, COMPARED_ROWS)
                    verdicts.append(
                        float(answer.result is not None and same_answer(expected, answer.result))
                    )
                return verdicts

            baseline = right(None)
            label = cases or "default"
            print(f"{label}: whole dictionary {sum(baseline) / len(baseline):.0%} "
                  f"of {len(baseline)}", flush=True)  # fmt: skip
            for name in names:
                linked = right(linker(name))
                versus = compare(np.array(linked), np.array(baseline), higher_is_better=True)
                wrong = [c.id for c, v in zip(questions, linked, strict=True) if not v]
                print(f"{label}: {name} {sum(linked) / len(linked):.0%} "
                      f"({versus.difference:+.0%}, {versus.ci_low:+.0%} to {versus.ci_high:+.0%}, "
                      f"{versus.probability_better:.0%} sure); wrong {wrong}",
                      flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
