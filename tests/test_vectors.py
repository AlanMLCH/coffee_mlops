"""The semantic index and the searches it serves, against Qdrant's in-process mode.

`QdrantClient(":memory:")` runs the same collection, sparse-vector, fusion and alias
logic the server does, without a server; the local model is replayed from a recorded
Ollama exchange, and in the command tests it answers with vectors derived from the text,
so the same text always lands in the same place.
"""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest
import typer
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from mlops_core import cli
from mlops_core.config import CHUNKS_TABLE, DOCUMENTS_TABLE
from mlops_core.data.corpus import CHUNKS_COLUMNS
from mlops_core.rag import llm
from mlops_core.rag.lexical import Bm25
from mlops_core.rag.llm import LocalModel, ollama_client
from mlops_core.rag.questions import (
    PROMPT_VERSION,
    Judgment,
    Question,
    questions_path,
    save_questions,
)
from mlops_core.rag.vectors import (
    IndexSearch,
    build_index,
    embedding_table,
    index_alias,
    index_metadata,
    query_text,
)
from mlops_core.storage import read_table, write_table

OLLAMA = Path(__file__).parent / "fixtures" / "ollama"
TEXTS = [
    "A light roast keeps the acidity of the origin.",
    "A dark roast turns bitter as the sugars burn.",
    "The cupping form scores eight affective sections.",
]


def corpus() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "chunk_id": f"doc-{n:04d}",
                "document_id": "doc",
                "chunk": n,
                "part": n,
                "part_title": None,
                "text": text,
                "characters": len(text),
                "topics": ["roasting"],
                "topics_basis": "terms",
            }
            for n, text in enumerate(TEXTS, start=1)
        ],
        schema=CHUNKS_COLUMNS,
    )


def vector_of(text: str, size: int = 8) -> list[float]:
    """A stand-in embedding: the same text always gives the same unit vector."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    raw = np.random.default_rng(seed).normal(size=size)
    unit: list[float] = (raw / np.linalg.norm(raw)).tolist()
    return unit


def built(
    client: QdrantClient, stamp: str = "20260924T000000Z", domain: str = "test", **metadata: str
) -> str:
    chunks = corpus()
    vectors = np.array([vector_of(t) for t in TEXTS], dtype=np.float32)
    return build_index(
        client, domain, chunks, embedding_table(chunks, vectors), metadata or {"k": "v"}, stamp
    )


# --- The index -------------------------------------------------------------------------------


def test_a_build_is_searched_through_an_alias_that_moves_only_when_it_is_complete() -> None:
    client = QdrantClient(":memory:")

    first = built(client, "20260924T000000Z", chunks_partition="p1")
    second = built(client, "20260925T000000Z", chunks_partition="p2")

    assert (first, second) == ("test-chunks-20260924T000000Z", "test-chunks-20260925T000000Z")
    assert [c.name for c in client.get_collections().collections] == [second]
    assert index_metadata(client, "test") == {"chunks_partition": "p2"}


def test_a_chunk_without_a_vector_is_refused() -> None:
    chunks = corpus()
    partial = embedding_table(chunks.head(2), np.zeros((2, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="no embedding"):
        build_index(QdrantClient(":memory:"), "test", chunks, partial, {}, "stamp")


def test_the_three_searches_answer_with_positions_in_the_chunk_table() -> None:
    client = QdrantClient(":memory:")
    built(client)
    asked: list[str] = []

    def embed(text: str) -> list[float]:
        asked.append(text)
        return vector_of(TEXTS[1])  # the question means the dark roast

    served = IndexSearch(client, "test", corpus(), embed)

    assert served.dense("Why is a dark roast bitter?", 1) == [1]
    assert asked[0] == query_text("Why is a dark roast bitter?")
    assert served.hybrid("Why is a dark roast bitter?", 2)[0] == 1
    # Qdrant's keyword half ranks exactly as the in-process BM25 does.
    keyword = Bm25(TEXTS)
    assert served.keyword("light roast acidity", 3) == keyword.search("light roast acidity", 3)


def test_an_index_of_other_chunks_is_refused_rather_than_misread() -> None:
    client = QdrantClient(":memory:")
    built(client)
    recut = corpus().with_columns(pl.col("chunk_id").str.replace("doc", "new"))

    with pytest.raises(LookupError, match="no longer has"):
        IndexSearch(client, "test", recut, vector_of).dense("roast", 1)


def test_the_query_carries_the_instruction_the_model_was_trained_with() -> None:
    assert query_text("Why?") == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query"
        "\nQuery:Why?"
    )


# --- The local model's vectors ---------------------------------------------------------------


def recorded_embeddings(seen: list[httpx.Request]) -> httpx.MockTransport:
    reply = json.loads((OLLAMA / "embed.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply)

    return httpx.MockTransport(handler)


def test_texts_are_embedded_in_batches_as_unit_vectors() -> None:
    seen: list[httpx.Request] = []
    with ollama_client("http://ollama.test", recorded_embeddings(seen)) as client:
        embedder = LocalModel(client, "qwen3-embedding:0.6b", {})
        vectors = embedder.embed(["a", "b", "c", "d"], batch=2)

    assert len(seen) == 2
    assert json.loads(seen[0].content)["input"] == ["a", "b"]
    assert vectors.shape == (4, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)


# --- The commands ----------------------------------------------------------------------------


def question(qid: str, excerpt: str) -> Question:
    return Question(
        id=qid,
        topic="roasting",
        question="Why does a dark roast taste bitter?",
        answer="The sugars burn.",
        relevant=[Judgment(document_id="doc", part=2, excerpt=excerpt, grade=2)],
        status="draft",
        drafted_by="model@abc",
        prompt=PROMPT_VERSION,
        drafted_on=date(2026, 9, 24),
        source_chunk="doc-0002",
    )


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> QdrantClient:
    """A clean corpus, a question set outside the source tree, an in-process Qdrant and
    an Ollama that embeds by hashing the text."""
    clean = tmp_path / "data" / "coffee" / "clean"
    write_table(corpus(), clean / CHUNKS_TABLE, {})
    write_table(
        pl.DataFrame({"document_id": ["doc"], "title": ["On roasting"], "publisher": ["P"]}),
        clean / DOCUMENTS_TABLE,
        {},
    )
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "domain_dir", lambda domain: tmp_path / "domain")
    monkeypatch.chdir(tmp_path)  # MLflow writes local artifacts under the working dir
    save_questions(
        questions_path(tmp_path / "domain"),
        [
            question("roasting-01", "A dark roast turns bitter as the sugars burn."),
            question("roasting-02", "A light roast keeps the acidity of the origin."),
        ],
    )
    client = QdrantClient(":memory:")
    monkeypatch.setattr(cli, "_qdrant", lambda url: client)

    tags = json.loads((OLLAMA / "tags.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [vector_of(t) for t in texts]})

    @contextmanager
    def served(url: str) -> Iterator[httpx.Client]:
        with ollama_client(url, httpx.MockTransport(handler)) as c:
            yield c

    monkeypatch.setattr(llm, "ollama_client", served)
    return client


def test_index_writes_the_vectors_to_parquet_and_loads_them_into_qdrant(
    workspace: QdrantClient, tmp_path: Path
) -> None:
    result = CliRunner().invoke(cli.app, ["rag", "index"])

    assert result.exit_code == 0, result.output
    vectors = read_table(tmp_path / "data" / "coffee" / "embeddings" / "chunk_embeddings")
    assert vectors.height == len(TEXTS)
    assert workspace.count(index_alias("coffee")).count == len(TEXTS)
    assert index_metadata(workspace, "coffee")["embedding_model"].startswith(
        "qwen3-embedding:0.6b@"
    )


def test_evaluate_walks_the_ladder_and_judges_each_step(workspace: QdrantClient) -> None:
    assert CliRunner().invoke(cli.app, ["rag", "index"]).exit_code == 0

    result = CliRunner().invoke(cli.app, ["rag", "evaluate"])

    assert result.exit_code == 0, result.output
    assert (
        result.output.index("bm25:")
        < result.output.index("dense:")
        < result.output.index("hybrid:")
    )
    assert "vs bm25:" in result.output and "vs dense:" in result.output
    assert "the gate" in result.output


def test_evaluate_refuses_an_index_built_from_other_chunks(workspace: QdrantClient) -> None:
    # Built before the corpus was cut again: its chunks are not the ones on disk.
    built(workspace, "20270101T000000Z", "coffee", chunks_partition="built_at=long-ago")

    result = CliRunner().invoke(cli.app, ["rag", "evaluate"])

    assert result.exit_code == 1
    assert "make index" in result.output


def test_qdrant_not_running_is_said_plainly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit):
        cli._qdrant("http://127.0.0.1:9")  # the discard port: nothing listens

    assert "docker compose --profile ai up -d" in capsys.readouterr().err


def test_a_qdrant_that_answers_is_handed_over(monkeypatch: pytest.MonkeyPatch) -> None:
    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda url: QdrantClient(":memory:"))

    assert cli._qdrant("http://127.0.0.1:6333").get_collections().collections == []


def test_tied_chunks_come_back_in_chunk_order_every_time() -> None:
    """Fusing ranks makes exact ties, and Qdrant orders tied points as it likes; a
    ranking that changes between two identical calls cannot be compared with another."""
    twins = (
        corpus()
        .head(2)
        .with_columns(
            pl.lit(TEXTS[0]).alias("text"), pl.Series("chunk_id", ["doc-0002", "doc-0001"])
        )
    )
    vectors = np.array([vector_of(TEXTS[0])] * 2, dtype=np.float32)
    client = QdrantClient(":memory:")
    build_index(client, "test", twins, embedding_table(twins, vectors), {}, "stamp")
    served = IndexSearch(client, "test", twins, lambda text: vector_of(TEXTS[0]))

    assert served.dense("light roast", 1) == [1]  # doc-0001, at position 1
    assert served.hybrid("light roast", 2) == [1, 0]
    assert served.keyword("light roast", 2) == [1, 0]
