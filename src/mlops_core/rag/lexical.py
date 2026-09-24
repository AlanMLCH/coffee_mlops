"""Keyword search: BM25, as Lucene and Elasticsearch ship it.

The baseline every other search has to beat, so it is built to be strong rather than
convenient: Lucene's defaults (k1 1.2, b 0.75), its 33 English stop words and the
Snowball English stemmer, so "roasting" finds "roasted". A weak baseline would make
semantic search look good for the wrong reason.

The weights are written as a sparse vector per chunk - the term-frequency half of BM25,
saturated and normalised by length - and the inverse document frequency is applied at
query time with Lucene's formula. That split is the one Qdrant makes for a sparse vector
with its IDF modifier, so the same encoding can be the keyword half of a hybrid search.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import snowballstemmer

K1 = 1.2
B = 0.75
# Lucene's EnglishAnalyzer stop set. Inverse document frequency already makes these
# nearly worthless; dropping them keeps the postings short.
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
        "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "was", "will", "with",
    }
)  # fmt: skip
_STEMMER = snowballstemmer.stemmer("english")
_WORD = re.compile(r"[^\W_]+")


def terms(text: str) -> list[str]:
    """Words, lower-cased, without stop words, stemmed: what BM25 counts."""
    words = [w for w in _WORD.findall(text.casefold()) if w not in STOP_WORDS]
    stemmed: list[str] = _STEMMER.stemWords(words)
    return stemmed


class Bm25:
    """An index over a fixed list of texts, ranking them for a query."""

    def __init__(self, texts: Sequence[str], k1: float = K1, b: float = B):
        counted = [Counter(terms(text)) for text in texts]
        lengths = np.array([sum(c.values()) for c in counted], dtype=float)
        average = lengths.mean() if len(texts) else 0.0
        postings: dict[str, list[tuple[int, float]]] = {}
        for position, counts in enumerate(counted):
            norm = k1 * (1 - b + b * lengths[position] / average)
            for term, tf in counts.items():
                postings.setdefault(term, []).append((position, tf * (k1 + 1) / (tf + norm)))
        self.size = len(texts)
        self._postings = {
            term: (np.array([p for p, _ in hits]), np.array([w for _, w in hits]))
            for term, hits in postings.items()
        }

    def idf(self, term: str) -> float:
        """Lucene's (and Qdrant's) inverse document frequency: never negative."""
        found = len(self._postings[term][0]) if term in self._postings else 0
        return math.log(1 + (self.size - found + 0.5) / (found + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """One score per text; a repeated query word counts once, as a sparse query does."""
        total = np.zeros(self.size)
        for term in set(terms(query)) & self._postings.keys():
            positions, weights = self._postings[term]
            total[positions] += self.idf(term) * weights
        return total

    def search(self, query: str, k: int) -> list[int]:
        """The positions of the `k` best texts, best first; texts sharing no word with the
        query are never returned, however short the list."""
        scores = self.scores(query)
        best = np.argsort(-scores, kind="stable")[:k]
        return [int(position) for position in best if scores[position] > 0]
