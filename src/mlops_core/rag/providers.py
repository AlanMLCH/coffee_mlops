"""Models behind other services' APIs, tried in order - free tiers first - before the
local model, which is always last.

The local model is the standard: free, private, offline, and what every result so far
was measured with. A hosted model answers more questions right, faster, for as long as
its free tier lasts, so the agent can ask one first. A chain asks the first provider
that still has quota. A provider that says it is out - HTTP 429, or a quota named in a
402 or 403 - is set aside until the moment it says; when it does not say, until the next
UTC day if the message is about a daily quota, the first of the next month for a monthly
one (Mistral's free mode), and for a minute otherwise. A per-minute
limit that clears within a few seconds is waited out once instead. A key refused, a
provider down or a reply that is not the shape asked for sets it aside too, for longer.
The local model is never set aside: the chain cannot run dry. When no provider has a
key, the chain is the local model alone, as before.

When a provider is set aside - when it ran out, how long it takes to come back, and
when it does - and every token it spends, is written to a small state file, so the next
run does not ask a provider whose quota is gone.

Verified 2026-09-29 against each provider's documentation, and each endpoint answered a
request without a key with 401 (the address is right, nothing was spent):

- Groq, OpenAI-compatible: free tier 30 requests a minute, 1,000 a day, 8,000 tokens a
  minute and 200,000 a day for `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
  `qwen/qwen3.8-27b`; all three take a JSON schema. 429 with `retry-after`.
- Gemini, OpenAI-compatible at `.../v1beta/openai/`: the Flash and Flash-Lite models are
  free with per-model limits shown in AI Studio; free-tier prompts may be used to
  improve Google's products. 429 RESOURCE_EXHAUSTED.
- Mistral, OpenAI-compatible: a free mode with monthly limits shown in its admin panel.
- OpenRouter, OpenAI-compatible: 20 requests a minute and 50 a day on its free models
  (`openrouter/free` routes to one that takes a schema).
- Cerebras: a free trial - $5 of credits for 30 days, and it asks for a card.
- Anthropic (Messages API) and OpenAI: paid, left out unless `MLOPS_PAID_PROVIDERS` is
  true.
NVIDIA's endpoint answered 404 and GitHub Models was retired in July 2026: not listed.

Two APIs cover them. The OpenAI-compatible one takes the reply's JSON schema as
`response_format`; a provider that refuses the schema is asked once more with plain JSON
mode, and from then on. Anthropic's is asked through a tool whose input is the schema.
The schema is also written into the prompt: a hosted model may not enforce it, and the
reply is validated against it either way.

Prompt caching: the long, stable part of a prompt - the rules and the data dictionary,
everything before its "Question:" line - comes first, so a provider that caches prefixes
on its own (OpenAI, Gemini) reuses it; Anthropic is told to (`cache_control`). On top,
`CachedGenerator` keeps every reply by its prompt and shape, so a prompt asked before
costs nothing at all - at temperature 0 the answer would be the same.

A key travels in a header, never in a URL, and is never logged or written anywhere.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import httpx
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, model_validator

logger = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(120.0, connect=10.0)
ANTHROPIC_VERSION = "2023-06-01"
MAX_WAIT = 20.0  # seconds a per-minute limit is waited out rather than moving on
# With nothing to move on to: Groq's 8,000 tokens a minute against a ~5,000-token SQL
# prompt asks for waits of 40 s and more.
ALONE_WAIT, ALONE_WAITS = 120.0, 3
FAILURE_PAUSE = timedelta(minutes=10)  # a provider down, or a reply of the wrong shape
QUOTA_PAUSE = timedelta(minutes=1)  # out of quota, with nothing said about until when
DAILY = re.compile(r"per.?day|daily|\bTPD\b|\bRPD\b|PerDay", re.IGNORECASE)
MONTHLY = re.compile(r"per.?month|monthly|PerMonth", re.IGNORECASE)
QUOTA = re.compile(r"quota|rate.?limit|credit|exhausted|insufficient", re.IGNORECASE)
QUESTION = "\nQuestion:"  # every prompt ends with its question: what comes before is stable
LOCAL = "local"
SHAPE = (
    "Reply with a single JSON object and nothing else - no prose, no code fence. It must "
    "match this JSON schema:\n{schema}"
)


class Generator(Protocol):
    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply: ...


class Ping(BaseModel):
    """The smallest reply there is: what `mlops agent providers --check` asks for."""

    ok: bool


class ProviderConfig(BaseModel):
    """One hosted model: where it is, which key it takes, and whether it costs money."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: Literal["openai", "anthropic"] = "openai"
    base_url: str
    model: str
    key: str  # the environment variable holding the key, never the key
    paid: bool = False
    structured: Literal["json_schema", "json_object"] = "json_schema"
    note: str = ""


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    providers: list[ProviderConfig] = []

    @model_validator(mode="after")
    def _names_are_unique(self) -> Self:
        names = [provider.name for provider in self.providers]
        odd = {name for name in names if names.count(name) > 1 or name == LOCAL}
        if odd:
            raise ValueError(f"provider names must be unique and not {LOCAL!r}: {sorted(odd)}")
        return self


def load_providers(path: Path) -> list[ProviderConfig]:
    """The chain as the file lists it; none if there is no file."""
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProvidersConfig.model_validate(raw).providers


def environment(
    env_file: Path = Path(".env"), env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The variables a key may be in: the process environment over the `.env` file."""
    found = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.is_file() else {}
    return found | dict(os.environ if env is None else env)


def usable(
    configs: Sequence[ProviderConfig],
    env: Mapping[str, str],
    paid: bool = False,
    only: str | None = None,
) -> list[tuple[ProviderConfig, SecretStr]]:
    """The providers the chain may ask, in order, each with its key: those whose key is
    set, the paid ones only if `paid`, and just one if `only` names it."""
    chosen = []
    for config in configs:
        if only is not None and config.name != only:
            continue
        if config.paid and not paid and only is None:
            continue
        key = env.get(config.key)
        if key:
            chosen.append((config, SecretStr(key)))
    if only is not None and only != LOCAL and not chosen:
        known = [c.name for c in configs]
        raise LookupError(
            f"No provider {only!r} with its key set: the providers are {known}, and "
            f"each needs its key's variable in the environment or .env"
        )
    return chosen


# --- What a provider says when it will not answer -------------------------------------------


class OutOfQuota(RuntimeError):
    """The provider will not answer before `until`."""

    def __init__(self, provider: str, until: datetime, reason: str):
        super().__init__(f"{provider}: {reason}")
        self.until, self.reason = until, reason


class Unavailable(RuntimeError):
    """The provider did not answer for a reason that is not its quota."""

    def __init__(self, provider: str, reason: str, pause: timedelta = FAILURE_PAUSE):
        super().__init__(f"{provider}: {reason}")
        self.reason, self.pause = reason, pause


def quota_until(response: httpx.Response, now: datetime) -> datetime:
    """When a provider out of quota will answer again: what its `retry-after` header or its
    error body says; else the first of the next month for a monthly quota, the next UTC
    day for a daily one, and a minute for the rest."""
    after = response.headers.get("retry-after")
    if after is not None:
        try:
            return now + timedelta(seconds=float(after))
        except ValueError:
            pass
    text = response.text
    delay = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', text)  # Gemini's error body
    if delay:
        return now + timedelta(seconds=float(delay.group(1)))
    if MONTHLY.search(text):
        following = (now.date().replace(day=1) + timedelta(days=32)).replace(day=1)
        return datetime.combine(following, datetime.min.time(), tzinfo=UTC)
    if DAILY.search(text):
        return datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    return now + QUOTA_PAUSE


def lasting(pause: timedelta) -> str:
    """How long a pause is, as a person reads it: 45 s, 12 min, 7.5 h, 3.0 d."""
    seconds = pause.total_seconds()
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def raise_for(provider: str, response: httpx.Response, now: datetime) -> None:
    """A provider's refusal as the chain reads it: out of quota, or unavailable."""
    status = response.status_code
    if status < 400:
        return
    first = " ".join(response.text.split())[:200]
    if status == 429 or (status in (402, 403) and QUOTA.search(first)):
        raise OutOfQuota(provider, quota_until(response, now), f"HTTP {status}: {first}")
    if status in (401, 403):
        # A key refused stays refused until someone changes it: a day, not ten minutes.
        raise Unavailable(provider, f"the key was refused (HTTP {status})", timedelta(days=1))
    raise Unavailable(provider, f"HTTP {status}: {first}")


# --- The two APIs ---------------------------------------------------------------------------


def split_stable(prompt: str) -> tuple[str, str]:
    """A prompt's stable head (rules, the data dictionary) and the rest, from its question
    on. No question line: all of it is the rest."""
    at = prompt.find(QUESTION)
    return (prompt[:at], prompt[at:].lstrip("\n")) if at > 0 else ("", prompt)


def json_text(content: str) -> str:
    """The JSON in a reply, without the code fence a model may put round it."""
    text = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return fenced.group(1) if fenced else text


class ApiModel:
    """One hosted model, asked for replies of a given shape."""

    def __init__(
        self,
        client: httpx.Client,
        config: ProviderConfig,
        key: SecretStr,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._client, self.config, self._key, self._clock = client, config, key, clock
        self.name = config.name
        self.model = f"{config.name}/{config.model}"
        self._structured = config.structured
        self._call = threading.local()

    @property
    def spent(self) -> dict[str, int]:
        """What the last `ask` on this thread spent: tokens in, out, and read from cache."""
        return dict(getattr(self._call, "spent", {}))

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        """One prompt, one reply shaped like `reply`; a malformed one is asked for again
        once, then raised."""
        self._call.spent = {"input": 0, "output": 0, "cached": 0, "calls": 0}
        try:
            return self._ask(prompt, reply)
        except ValidationError as cut:
            logger.warning("%s gave a malformed %s; asking again: %s", self.model,
                           reply.__name__, str(cut).splitlines()[0])  # fmt: skip
            return self._ask(prompt, reply)

    def _ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        if self.config.kind == "anthropic":
            return reply.model_validate(self._anthropic(prompt, reply))
        return reply.model_validate_json(json_text(self._openai(prompt, reply)))

    def _post(self, path: str, body: dict[str, object], headers: dict[str, str]) -> httpx.Response:
        try:
            return self._client.post(path, json=body, headers=headers)
        except httpx.TransportError as failed:
            raise Unavailable(self.name, f"no answer: {type(failed).__name__}") from failed

    def _openai(self, prompt: str, reply: type[BaseModel]) -> str:
        schema = reply.model_json_schema()
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SHAPE.format(schema=json.dumps(schema))},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": self._format(reply, schema),
        }
        headers = {"Authorization": f"Bearer {self._key.get_secret_value()}"}
        response = self._post("chat/completions", body, headers)
        if (
            response.status_code == 400
            and self._structured == "json_schema"
            and re.search(r"response_format|json_schema|schema", response.text, re.IGNORECASE)
        ):
            logger.warning("%s refuses a JSON schema; asking in plain JSON mode from now on",
                           self.model)  # fmt: skip
            self._structured = "json_object"
            body["response_format"] = self._format(reply, schema)
            response = self._post("chat/completions", body, headers)
        raise_for(self.name, response, self._clock())
        answer = response.json()
        usage = answer.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        self._count(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cached)
        content = answer["choices"][0]["message"].get("content")
        if not content:
            raise Unavailable(self.name, "an empty reply")
        return str(content)

    def _format(self, reply: type[BaseModel], schema: dict[str, object]) -> dict[str, object]:
        if self._structured == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {"name": reply.__name__, "schema": schema, "strict": False},
        }

    def _anthropic(self, prompt: str, reply: type[BaseModel]) -> dict[str, object]:
        stable, rest = split_stable(prompt)
        body: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "temperature": 0,
            "messages": [{"role": "user", "content": rest}],
            "tools": [
                {
                    "name": "reply",
                    "description": f"Give the {reply.__name__}.",
                    "input_schema": reply.model_json_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": "reply"},
        }
        if stable:
            # Everything before the question is the same for every question: cached, it
            # is read at a tenth of the price on the next call within five minutes.
            body["system"] = [
                {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}
            ]
        headers = {
            "x-api-key": self._key.get_secret_value(),
            "anthropic-version": ANTHROPIC_VERSION,
        }
        response = self._post("messages", body, headers)
        raise_for(self.name, response, self._clock())
        answer = response.json()
        usage = answer.get("usage") or {}
        self._count(
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
        )
        for block in answer.get("content", []):
            if block.get("type") == "tool_use":
                return dict(block["input"])
        raise Unavailable(self.name, "a reply without the tool call asked for")

    def _count(self, prompt_tokens: int, output_tokens: int, cached: int) -> None:
        spent = self._call.spent
        spent["input"] += int(prompt_tokens)
        spent["output"] += int(output_tokens)
        spent["cached"] += int(cached)
        spent["calls"] += 1


@contextmanager
def provider_client(
    config: ProviderConfig, transport: httpx.BaseTransport | None = None
) -> Iterator[httpx.Client]:
    """A client for one provider, its base URL ending in a slash so relative paths extend
    it. Its URLs carry no secret, but request logging stays off anyway, as it does for
    every client of this project's (`data.api.silence_request_urls`)."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    base = config.base_url if config.base_url.endswith("/") else f"{config.base_url}/"
    with httpx.Client(base_url=base, timeout=TIMEOUT, transport=transport) as client:
        yield client


# --- The chain ------------------------------------------------------------------------------


class Cooldowns:
    """Which providers are set aside, until when and why, and what each spent today -
    kept in a file so the next run knows. Safe to use from several threads."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"aside": {}, "spent": {}}
        if path is not None and path.is_file():
            try:
                self._state |= json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("%s could not be read: starting with every provider", path)

    def until(self, name: str, now: datetime) -> datetime | None:
        """When a set-aside provider may be asked again; None if it may be now."""
        entry = self._state["aside"].get(name)
        if not entry:
            return None
        until = datetime.fromisoformat(entry["until"])
        return until if until > now else None

    def set_aside(self, name: str, until: datetime, reason: str, since: datetime) -> None:
        """`name` is not asked from `since`, when it ran out, until `until`, when it said
        it would answer again; how long that is is kept too."""
        with self._lock:
            self._state["aside"][name] = {
                "since": since.isoformat(),
                "until": until.isoformat(),
                "pause_seconds": round((until - since).total_seconds()),
                "reason": reason,
            }
            self._save()

    def spend(self, name: str, usage: Mapping[str, int], today: str) -> None:
        with self._lock:
            spent = self._state["spent"]
            day = spent.setdefault(today, {})
            earlier = day.get(name, {})
            day[name] = {k: earlier.get(k, 0) + v for k, v in usage.items()}
            for old in sorted(spent)[:-7]:  # a week is enough to see a pattern
                del spent[old]
            self._save()

    def report(self) -> dict[str, Any]:
        """What the file says: providers set aside and tokens spent, by day."""
        return dict(json.loads(json.dumps(self._state)))

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._state, indent=1), encoding="utf-8", newline="\n")
        temporary.replace(self._path)


class NoModelLeft(RuntimeError):
    """Every provider was set aside and there is no local model to fall back on."""


class Chain:
    """Hosted models in order, then the local one: each call goes to the first that may
    answer. `last` says which one did (per thread), for the trace."""

    def __init__(
        self,
        members: Sequence[ApiModel],
        local: Generator | None,
        cooldowns: Cooldowns | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
        local_name: str = LOCAL,
    ):
        self._members, self._local = list(members), local
        self._cooldowns = cooldowns or Cooldowns()
        self._clock, self._sleep, self._local_name = clock, sleep, local_name
        self._seen = threading.local()

    @property
    def last(self) -> str | None:
        return getattr(self._seen, "name", None)

    def names(self) -> list[str]:
        return [member.model for member in self._members] + (
            [self._local_name] if self._local is not None else []
        )

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        # Alone, a member set aside for moments is waited for, as `_try` waits: seen
        # 2026-10-07, one set aside for 7 s failed the three questions asked within them.
        rounds = ALONE_WAITS if self._local is None else 0
        while True:
            for member in self._members:
                now = self._clock()
                if self._cooldowns.until(member.name, now) is not None:
                    continue
                answer = self._try(member, prompt, reply)
                if answer is not None:
                    self._seen.name = member.model
                    return answer
            if self._local is not None:
                self._seen.name = self._local_name
                return self._local.ask(prompt, reply)
            wait = self._soonest()
            if rounds > 0 and wait is not None and wait <= ALONE_WAIT:
                rounds -= 1
                self._sleep(wait)
                continue
            raise NoModelLeft(
                "every provider is set aside and the local model is not in the chain: "
                + "; ".join(self._why())
            )

    def _soonest(self) -> float | None:
        """Seconds until the first member set aside may be asked again; None if none will."""
        now = self._clock()
        untils = [self._cooldowns.until(member.name, now) for member in self._members]
        waits = [(until - now).total_seconds() for until in untils if until is not None]
        return min(waits) if waits else None

    def _try[Reply: BaseModel](
        self, member: ApiModel, prompt: str, reply: type[Reply]
    ) -> Reply | None:
        """The member's reply, or None and the member set aside. A limit that clears
        within `MAX_WAIT` seconds is waited out once - or, with no model to fall back on
        (one model measured), within `ALONE_WAIT`, up to `ALONE_WAITS` times: a per-minute
        limit there would otherwise fail the question it fell on."""
        alone = self._local is None
        patience, waits = (ALONE_WAIT, ALONE_WAITS) if alone else (MAX_WAIT, 1)
        while True:
            try:
                answer = member.ask(prompt, reply)
            except OutOfQuota as out:
                wait = (out.until - self._clock()).total_seconds()
                if waits > 0 and wait <= patience:
                    waits -= 1
                    self._sleep(max(wait, 0.0))
                    continue
                self._aside(member, out.until, out.reason)
                return None
            except Unavailable as down:
                self._aside(member, self._clock() + down.pause, down.reason)
                return None
            except ValidationError as wrong:
                first = str(wrong).splitlines()[0]
                reason = f"twice a reply that is not a {reply.__name__}: {first}"
                self._aside(member, self._clock() + FAILURE_PAUSE, reason)
                return None
            finally:
                spent = member.spent
                if any(spent.values()):
                    self._cooldowns.spend(member.name, spent, self._clock().date().isoformat())
            return answer

    def _aside(self, member: ApiModel, until: datetime, reason: str) -> None:
        now = self._clock()
        logger.warning("%s set aside for %s, until %s: %s", member.model, lasting(until - now),
                       until.isoformat(timespec="seconds"), reason)  # fmt: skip
        self._cooldowns.set_aside(member.name, until, reason, since=now)

    def _why(self) -> list[str]:
        aside = self._cooldowns.report()["aside"]
        return [f"{name} until {e['until']} ({e['reason']})" for name, e in aside.items()]


# --- Replies kept -------------------------------------------------------------------------


class ReplyCache:
    """Replies by prompt and shape, in a SQLite file, for `max_age`. Safe across threads."""

    def __init__(self, path: Path, max_age: timedelta = timedelta(days=30)):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._max_age = max_age
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS replies "
                "(key TEXT PRIMARY KEY, reply TEXT NOT NULL, answered_by TEXT, at TEXT NOT NULL)"
            )

    def get(self, key: str, now: datetime) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT reply, at FROM replies WHERE key = ?", (key,)).fetchone()
        if row is None or datetime.fromisoformat(row[1]) < now - self._max_age:
            return None
        return str(row[0])

    def put(self, key: str, reply: str, answered_by: str | None, now: datetime) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?)",
                (key, reply, answered_by, now.isoformat()),
            )

    def close(self) -> None:
        self._db.close()


class CachedGenerator:
    """A generator whose replies are kept: the same prompt for the same shape is answered
    from the cache, and costs nothing. `last` is "cache" for such a reply."""

    def __init__(
        self,
        inner: Generator,
        cache: ReplyCache,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._inner, self._cache, self._clock = inner, cache, clock
        self._seen = threading.local()

    @property
    def last(self) -> str | None:
        return getattr(self._seen, "name", None)

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        key = sha256(
            (prompt + "\n" + json.dumps(reply.model_json_schema(), sort_keys=True)).encode()
        ).hexdigest()
        kept = self._cache.get(key, self._clock())
        if kept is not None:
            try:
                answer = reply.model_validate_json(kept)
            except ValidationError:
                pass  # the shape changed under the same name: ask again
            else:
                self._seen.name = "cache"
                return answer
        answer = self._inner.ask(prompt, reply)
        by = getattr(self._inner, "last", None)
        self._seen.name = by
        self._cache.put(key, answer.model_dump_json(), by, self._clock())
        return answer
