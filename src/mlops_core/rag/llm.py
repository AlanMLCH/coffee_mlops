"""A local model behind Ollama's HTTP API, asked for replies that fit a schema.

Structured output is constrained generation, not a polite request: Ollama's `format`
takes a JSON schema, and the model can only emit tokens that keep the reply valid
against it (Ollama 0.34.2). The schema comes from a pydantic model and the reply is
validated against that same model, so a reply that parses is a reply the code can use.

Local and keyless: no request carries a secret, so no secret can reach a log.

One request at a time, whoever asks: every model here shares one GPU (6 GB on the laptop
this runs on). Asked several things at once, Ollama queues them or splits the card -
and a 4B model's 5,000-token context fits once, not three times; and requests batched
together are computed in another order, so the same prompt can come back otherwise. So
while hosted models answer several questions at once, the local one takes its turns.
"""

import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Loading a model into VRAM is the slow part of a first request: about a minute and a
# half for a 4B model on the laptop this runs on. A reply after that takes seconds.
TIMEOUT = httpx.Timeout(300.0, connect=5.0)
# Texts per embedding request: 32 chunks took 1.4 s on the laptop's GPU (2026-09-24).
EMBEDDING_BATCH = 32
# Every request to a local model holds it: one at a time, from any thread.
LOCAL_TURN = threading.Lock()


@contextmanager
def ollama_client(url: str, transport: httpx.BaseTransport | None = None) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=url, timeout=TIMEOUT, transport=transport) as client:
        yield client


class LocalModel:
    """One model served by Ollama, with the sampling options every request uses."""

    def __init__(self, client: httpx.Client, model: str, options: Mapping[str, float]):
        self._client = client
        self.model = model
        self._options = dict(options)

    def digest(self) -> str:
        """The weights behind the tag, first 12 hex digits. A tag can be pulled again and
        point at other weights; the digest cannot, so it is what a result records."""
        try:
            listed = self._client.get("/api/tags")
        except httpx.ConnectError as down:
            raise ConnectionError(
                f"Ollama is not answering at {self._client.base_url}: start it (ollama serve)"
            ) from down
        listed.raise_for_status()
        for entry in listed.json()["models"]:
            if entry["name"] == self.model:
                return str(entry["digest"])[:12]
        raise LookupError(f"Ollama has no model '{self.model}': ollama pull {self.model}")

    def embed(self, texts: Sequence[str], batch: int = EMBEDDING_BATCH) -> np.ndarray:
        """One unit-length vector per text, in batches. Ollama normalises them, and cuts a
        text longer than the model's context without saying so - the caller keeps its
        texts short (a chunk is at most a few hundred tokens)."""
        vectors = []
        for start in range(0, len(texts), batch):
            chunk = list(texts[start : start + batch])
            body: dict[str, object] = {"model": self.model, "input": chunk}
            if self._options:
                # The context the model is loaded with: without it Ollama loads the
                # default, and a query-sized model is the one that fits beside another.
                body["options"] = self._options
            with LOCAL_TURN:
                response = self._client.post("/api/embed", json=body)
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return np.array(vectors, dtype=np.float32)

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        """One prompt, one reply shaped like `reply`. A reply cut short - it happened once,
        `{"` and nothing more, under load - is asked for once more before it is an error."""
        try:
            return self._ask(prompt, reply)
        except ValidationError as cut:
            logger.warning("%s gave a malformed %s; asking again: %s", self.model,
                           reply.__name__, str(cut).splitlines()[0])  # fmt: skip
            return self._ask(prompt, reply)

    def _ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": reply.model_json_schema(),
            "stream": False,
            # A reasoning model would otherwise think aloud first; the schema already
            # says what the answer must be, and the thinking would only cost time.
            "think": False,
            "options": self._options,
        }
        with LOCAL_TURN:
            response = self._client.post("/api/chat", json=body)
        response.raise_for_status()
        return reply.model_validate_json(response.json()["message"]["content"])
