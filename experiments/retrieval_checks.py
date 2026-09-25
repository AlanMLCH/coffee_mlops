"""Two things the retrieval ladder takes on trust, measured.

1. That the keyword half of the hybrid search is the BM25 the baseline was: Qdrant
   stores each chunk's term-frequency weights and applies the inverse document frequency
   itself, so for every question its top ten should be the in-process BM25's top ten.
2. That the query instruction Qwen3-Embedding's model card asks for earns its place: the
   card puts leaving it out at 1-5% of retrieval. Here, dense search with and without it,
   on the same questions, through the gate's paired bootstrap.

Needs Qdrant (`make services-up PROFILE=ai`), Ollama and a built index (`make index`):

    uv run python experiments/retrieval_checks.py
"""

import logging
from functools import cache

import numpy as np
from qdrant_client import QdrantClient

from mlops_core.adapter import domain_dir, load_adapter
from mlops_core.config import CHUNKS_TABLE, Settings
from mlops_core.rag.evaluate import GATE_METRIC, score_search
from mlops_core.rag.lexical import Bm25
from mlops_core.rag.llm import LocalModel, ollama_client
from mlops_core.rag.questions import load_questions, questions_path
from mlops_core.rag.vectors import DENSE, EMBEDDING_MODEL, IndexSearch, index_alias
from mlops_core.stats import compare
from mlops_core.storage import read_table


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    config = load_adapter("coffee").config
    assert config.corpus is not None
    settings = Settings()
    chunks = read_table(settings.data_dir / config.name / "clean" / CHUNKS_TABLE)
    questions = [
        q
        for q in load_questions(questions_path(domain_dir(config.name)), config.corpus.topics)
        if q.status != "rejected"
    ]
    client = QdrantClient(url=settings.qdrant_url)
    keyword = Bm25(chunks["text"].to_list())

    with ollama_client(settings.ollama_url) as http:
        embedder = LocalModel(http, EMBEDDING_MODEL, {})

        @cache
        def embed(text: str) -> list[float]:
            vector: list[float] = embedder.embed([text])[0].tolist()
            return vector

        served = IndexSearch(client, config.name, chunks, embed)

        same = sum(
            keyword.search(q.question, 10) == served.keyword(q.question, 10) for q in questions
        )
        print(f"Qdrant's BM25 ranks as the in-process BM25 on {same} of {len(questions)} questions")

        def plain(question: str, k: int) -> list[int]:
            found = client.query_points(
                index_alias(config.name),
                query=embed(question),
                using=DENSE,
                limit=k,
                with_payload=["chunk_id"],
            )
            positions = {c: p for p, c in enumerate(chunks["chunk_id"])}
            return [positions[str((p.payload or {})["chunk_id"])] for p in found.points]

        with_instruction = score_search(served.dense, questions, chunks)
        without = score_search(plain, questions, chunks)

    instructed = with_instruction[GATE_METRIC].to_numpy().astype(np.float64)
    bare = without[GATE_METRIC].to_numpy().astype(np.float64)
    verdict = compare(instructed, bare, higher_is_better=True)
    print(
        f"{GATE_METRIC}: {instructed.mean():.3f} with the instruction, {bare.mean():.3f} "
        f"without; difference {verdict.difference:+.3f} "
        f"[{verdict.ci_low:+.3f}, {verdict.ci_high:+.3f}], "
        f"{verdict.probability_better:.0%} sure the instruction helps"
    )


if __name__ == "__main__":
    main()
