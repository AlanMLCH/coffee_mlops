"""Does the SQL writer answer more questions right when several queries vote on the answer?

Execution-guided self-consistency, as text-to-SQL systems use it: the usual query (the
agent's settings: temperature 0, its seed) and a few more sampled at a higher temperature,
each with the agent's repairs, guards and checks; every one that runs gives a result, and
the result most of them give wins (a tie keeps the usual query's). A small model's wrong
queries tend to disagree with each other, its right ones to agree.

Measured on the benchmark's SQL questions, the default set and the held-out one, judged as
the benchmark judges them, paired question by question against the usual query alone.

Run with Ollama up:  uv run python experiments/sql_voting.py 5 0.7 [default] [holdout]
(the number of queries, the temperature of the sampled ones, and the sets; both unnamed)
"""

import sys
from collections import Counter

import numpy as np

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
from mlops_core.agent.dictionary import dictionary_path, schema_context
from mlops_core.agent.graph import AGENT_GENERATOR
from mlops_core.agent.sql import QueryResult, read_only, run_select, views
from mlops_core.agent.text_to_sql import SqlAnswer, voted, write_sql
from mlops_core.config import Settings
from mlops_core.rag.llm import LocalModel, ollama_client
from mlops_core.stats import compare


def key(result: QueryResult) -> tuple[tuple[object, ...], ...]:
    """A result as `text_to_sql.voted` counts it, for the agreement the log shows."""

    def value(v: object) -> object:
        return float(f"{v:.4g}") if isinstance(v, int | float) and not isinstance(v, bool) else v

    return tuple(sorted((tuple(value(v) for v in row) for row in result.rows), key=repr))


def right(expected: QueryResult, answer: SqlAnswer) -> float:
    return float(answer.result is not None and same_answer(expected, answer.result))


def main() -> None:
    queries, temperature = int(sys.argv[1]), float(sys.argv[2])
    config = load_adapter("coffee").config
    home = domain_dir(config.name)
    con = read_only(Settings().data_dir / config.name)
    schema = schema_context(dictionary_path(home).read_text(encoding="utf-8"), views(con))
    guards = config.agent.sql_guards
    with ollama_client(Settings().ollama_url) as http:
        usual = LocalModel(http, AGENT_GENERATOR, GENERATOR_OPTIONS)
        sampled = [
            LocalModel(http, AGENT_GENERATOR,
                       GENERATOR_OPTIONS | {"temperature": temperature, "seed": 100 + n})
            for n in range(queries - 1)
        ]  # fmt: skip
        for cases in [None if s == "default" else s for s in sys.argv[3:]] or (None, "holdout"):
            questions = load_cases(case_file(home, SQL_CASES_FILE, cases), SqlCase)
            alone, vote = [], []
            for case in questions:
                expected = run_select(con, case.sql, COMPARED_ROWS)
                answers = [
                    write_sql(model, con, schema, case.question, COMPARED_ROWS, guards)
                    for model in (usual, *sampled)
                ]
                rights = [right(expected, a) for a in answers]
                alone.append(rights[0])
                vote.append(right(expected, voted(answers)))
                agree = Counter(key(a.result) for a in answers if a.result is not None)
                largest = max(agree.values(), default=0)
                print(f"  {case.id}: alone {alone[-1]:.0f}, voted {vote[-1]:.0f}, "
                      f"{sum(rights):.0f}/{queries} right, largest agreement {largest}",
                      flush=True)  # fmt: skip
            versus = compare(np.array(vote), np.array(alone), higher_is_better=True)
            label = cases or "default"
            print(f"{label}: alone {np.mean(alone):.0%}, {queries} voting {np.mean(vote):.0%} "
                  f"({versus.difference:+.0%}, {versus.ci_low:+.0%} to {versus.ci_high:+.0%}, "
                  f"{versus.probability_better:.0%} sure)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
