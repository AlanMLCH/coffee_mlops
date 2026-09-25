"""What the agent's model is asked, and the shape of what it may answer.

Each reply is a pydantic model, sent to Ollama as the JSON schema its output is
constrained to, so a reply that parses is one the code can use. Each prompt is
versioned by its content - template and schema together - and every run records the
version it used: a changed prompt is a new candidate, judged like a new model.
"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel

Route = Literal["data", "prediction", "knowledge", "mixed"]
ROUTES: tuple[Route, ...] = ("data", "prediction", "knowledge", "mixed")


class SqlReply(BaseModel):
    sql: str


class RouteReply(BaseModel):
    route: Route


SQL = """You write one DuckDB SQL query that answers a question about the tables below.

Rules:
- One SELECT statement. Use only the tables and columns described below, named as
  their headings name them (schema.table).
- Return the columns the answer needs, and nothing the question did not ask for.
- For "the most", "the highest" or "the top N", order and limit accordingly.
- Null means not reported. Leave nulls out of rankings, and count them only when the
  question asks about missing values.

{schema}

Question: {question}
"""

REPAIR = """
Your previous query was:
{sql}

It failed with:
{error}

Write a corrected query.
"""

ROUTER = """You route questions about {subject} to the tool that can answer them.

- data: figures, counts, rankings and comparisons that can be read from these tables:
{tables}
- prediction: what a model would predict for an item the question describes, which is
  not in the tables. The models:
{models}
- knowledge: how and why - explanations from a library of documents on:
{topics}
- mixed: the question needs two of the above, such as a figure and an explanation, or a
  prediction and a figure to compare it with.

A question about what a model predicted for items already in the tables is data.

Question: {question}
"""


def version(template: str, reply: type[BaseModel]) -> str:
    """Eight characters that change whenever the template or the reply's schema does."""
    schema = json.dumps(reply.model_json_schema(), sort_keys=True)
    return hashlib.sha256((template + schema).encode()).hexdigest()[:8]


SQL_VERSION = version(SQL + REPAIR, SqlReply)
ROUTER_VERSION = version(ROUTER, RouteReply)
