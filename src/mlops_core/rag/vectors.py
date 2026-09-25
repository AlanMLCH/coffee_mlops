"""Semantic and hybrid search, and the index that serves them.

Every chunk is embedded by a local model and the vectors are written to Parquet - the
source of truth - then loaded into Qdrant beside the chunk's BM25 weights and its
metadata. Qdrant is an index derived from the layers and rebuilt from them: each build
goes into a new collection, and the alias readers use is moved onto it in one operation,
so nobody ever searches half an index and a failed build leaves the old one serving.

Three searches, each a step the gate has to let through:
- dense: the question's embedding against the chunks', by cosine;
- hybrid: dense and BM25 each offer their best candidates, and reciprocal rank fusion
  merges the two lists by rank, not score - a cosine and a BM25 score are not on one
  scale, and RRF never has to pretend they are.
"""

import uuid
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl
from qdrant_client import QdrantClient, models

from mlops_core.rag.lexical import Bm25, query_vector

# Decided 2026-09-22 (CLAUDE.md): multilingual, 1,024 dimensions, 639 MB of VRAM.
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
# Qwen3-Embedding is trained with an instruction on the query side and none on the
# documents'; its model card puts the cost of leaving it out at 1-5% of retrieval, and
# asks for it in English. The format is the card's, to the character.
QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"

# A question is a few dozen tokens. With the default context of 4,096 the embedding
# model took 2.37 GB of VRAM and did not fit beside the agent's generator (3.27 GB) on a
# 6 GB card; at 512 it takes 1.01 GB and both stay loaded. Indexing keeps the default: a
# chunk can pass 512 tokens.
QUERY_OPTIONS = {"num_ctx": 512}

EMBEDDINGS = "embeddings"  # the layer
EMBEDDINGS_TABLE = "chunk_embeddings"
DENSE, SPARSE = "dense", "bm25"  # the collection's two named vectors
# RRF's constant, from the paper that introduced it (Cormack et al., 2009). Set rather
# than left to the server's default, so the run can record it.
RRF_K = 60
# How many candidates each half of a hybrid search offers the fusion.
PREFETCH = 50
UPSERT_BATCH = 256
# Fusing ranks makes exact ties (rank 2 in one list and 5 in the other scores as 5 and 2),
# and Qdrant orders tied points arbitrarily: 17 of 108 hybrid rankings changed between
# two identical calls. So a few more are asked for, ordered here - by score, then by
# chunk id - and cut, which makes a ranking repeatable unless a tie runs past the margin.
TIE_MARGIN = 10


def query_text(question: str) -> str:
    return f"Instruct: {QUERY_TASK}\nQuery:{question}"


def index_alias(domain: str) -> str:
    """What readers search: it always names one complete build."""
    return f"{domain}-chunks"


def embedding_table(chunks: pl.DataFrame, vectors: np.ndarray) -> pl.DataFrame:
    """One row per chunk: its id and its vector, as a fixed-width array column."""
    return pl.DataFrame({"chunk_id": chunks["chunk_id"], "embedding": vectors})


def build_index(
    client: QdrantClient,
    domain: str,
    chunks: pl.DataFrame,
    embeddings: pl.DataFrame,
    metadata: Mapping[str, str],
    stamp: str,
) -> str:
    """Load every chunk into a new collection - dense vector, BM25 weights, payload -
    move the alias onto it, and drop the builds before it. Returns the collection."""
    aligned = chunks.join(embeddings, on="chunk_id", how="left")
    if aligned["embedding"].null_count():
        raise ValueError("Some chunks have no embedding: embed the chunks being indexed")
    alias = index_alias(domain)
    collection = f"{alias}-{stamp}"
    # An Array column comes out of polars as one 2-D array.
    vectors = np.asarray(aligned["embedding"].to_numpy(), dtype=np.float32)
    client.create_collection(
        collection,
        vectors_config={
            DENSE: models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE)
        },
        # Qdrant completes BM25 at query time with Lucene's inverse document frequency.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata=dict(metadata),
    )
    keyword = Bm25(aligned["text"].to_list())
    rows = aligned.select("chunk_id", "document_id", "part", "part_title", "topics", "text")
    points = []
    for position, row in enumerate(rows.iter_rows(named=True)):
        ids, weights = keyword.document_vector(position)
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{domain}/{row['chunk_id']}")),
                vector={
                    DENSE: vectors[position].tolist(),
                    SPARSE: models.SparseVector(indices=ids, values=weights),
                },
                payload=row,
            )
        )
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection, points[start : start + UPSERT_BATCH])

    # One request: the old alias goes and the new one comes at once, or neither does.
    had_alias = any(a.alias_name == alias for a in client.get_aliases().aliases)
    swap: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
    if had_alias:
        swap.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    swap.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=swap)
    for old in client.get_collections().collections:
        if old.name.startswith(f"{alias}-") and old.name != collection:
            client.delete_collection(old.name)
    return collection


def index_metadata(client: QdrantClient, domain: str) -> dict[str, Any]:
    """What the collection the alias names was built from."""
    return dict(client.get_collection(index_alias(domain)).config.metadata or {})


class IndexSearch:
    """The searches Qdrant serves, answering with positions in the chunk table the
    evaluation holds - the same positions BM25 answers with."""

    def __init__(
        self,
        client: QdrantClient,
        domain: str,
        chunks: pl.DataFrame,
        embed: Callable[[str], list[float]],
    ):
        self._client = client
        self._alias = index_alias(domain)
        self._embed = embed
        self._positions = {chunk_id: p for p, chunk_id in enumerate(chunks["chunk_id"])}

    def dense(self, question: str, k: int) -> list[int]:
        return self._located(self._nearest(question, k), k)

    def passages(self, question: str, k: int) -> list[dict[str, Any]]:
        """The `k` nearest chunks whole - text, document, page or section - best first:
        what the agent reads and cites."""
        nearest = self._nearest(question, k)
        self._located(nearest, k)  # refuses an index built from other chunks
        ranked = sorted(nearest, key=lambda point: (-point.score, _chunk_id(point)))[:k]
        return [dict(point.payload or {}) for point in ranked]

    def _nearest(self, question: str, k: int) -> list[models.ScoredPoint]:
        found = self._client.query_points(
            self._alias,
            query=self._embed(query_text(question)),
            using=DENSE,
            limit=k + TIE_MARGIN,
            with_payload=True,
        )
        return found.points

    def hybrid(self, question: str, k: int) -> list[int]:
        ids, weights = query_vector(question)
        found = self._client.query_points(
            self._alias,
            prefetch=[
                models.Prefetch(
                    query=self._embed(query_text(question)), using=DENSE, limit=PREFETCH
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=ids, values=weights),
                    using=SPARSE,
                    limit=PREFETCH,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
            limit=k + TIE_MARGIN,
            with_payload=["chunk_id"],
        )
        return self._located(found.points, k)

    def keyword(self, question: str, k: int) -> list[int]:
        """BM25 as Qdrant computes it, to check it agrees with the in-process one."""
        ids, weights = query_vector(question)
        found = self._client.query_points(
            self._alias,
            query=models.SparseVector(indices=ids, values=weights),
            using=SPARSE,
            limit=k + TIE_MARGIN,
            with_payload=["chunk_id"],
        )
        return self._located(found.points, k)

    def _located(self, points: list[models.ScoredPoint], k: int) -> list[int]:
        ranked = sorted(points, key=lambda point: (-point.score, _chunk_id(point)))
        chunk_ids = [_chunk_id(point) for point in ranked[:k]]
        stale = [c for c in chunk_ids if c not in self._positions]
        if stale:
            # The corpus was cut again and the index was not rebuilt: its chunks are not
            # the ones the labels are being matched against.
            raise LookupError(f"The index holds chunks the corpus no longer has ({stale[0]}…)")
        return [self._positions[c] for c in chunk_ids]


def _chunk_id(point: models.ScoredPoint) -> str:
    return str((point.payload or {})["chunk_id"])
