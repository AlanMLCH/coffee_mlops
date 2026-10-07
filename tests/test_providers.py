"""The chain of hosted models before the local one, over stood-in HTTP: what each API is
sent, how a refusal is read, when a provider is set aside and for how long, and replies
kept by prompt. No key and no network: every provider is an `httpx.MockTransport`."""

import json
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr, ValidationError
from typer.testing import CliRunner

from mlops_core import cli
from mlops_core.config import Settings
from mlops_core.rag import providers
from mlops_core.rag.providers import (
    FAILURE_PAUSE,
    ApiModel,
    CachedGenerator,
    Chain,
    Cooldowns,
    NoModelLeft,
    OutOfQuota,
    ProviderConfig,
    ReplyCache,
    Unavailable,
    environment,
    json_text,
    lasting,
    load_providers,
    provider_client,
    quota_until,
    raise_for,
    split_stable,
    usable,
)

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
KEY = SecretStr("sk-test-not-a-real-key")


class Shape(BaseModel):
    sql: str


def provider(name: str = "free", kind: str = "openai", **more: Any) -> ProviderConfig:
    base = {"name": name, "kind": kind, "base_url": "https://llm.test/v1", "model": "m-1"}
    return ProviderConfig.model_validate(base | {"key": f"{name.upper()}_KEY"} | more)


def model(
    respond: Callable[[httpx.Request], httpx.Response], config: ProviderConfig | None = None
) -> tuple[ApiModel, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def recorded(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return respond(request)

    client = httpx.Client(base_url="https://llm.test/v1/", transport=httpx.MockTransport(recorded))
    return ApiModel(client, config or provider(), KEY, clock=lambda: NOW), sent


def chat(content: str, usage: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": usage or {}})  # fmt: skip


class Answers:
    """A stood-in generator: its replies in order, or an exception to raise."""

    def __init__(self, *replies: object):
        self.replies, self.asked = list(replies), 0

    def ask[R: BaseModel](self, prompt: str, reply: type[R]) -> R:
        self.asked += 1
        answer = self.replies.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return reply.model_validate(answer)


class Member(Answers):
    """A stood-in hosted model, as the chain sees one."""

    def __init__(self, name: str, *replies: object, spent: dict[str, int] | None = None):
        super().__init__(*replies)
        self.name, self.model, self.spent = name, f"{name}/m", spent or {}


# --- The file and the keys ------------------------------------------------------------------


def test_the_chain_is_read_from_its_file_and_names_are_its_own(tmp_path: Path) -> None:
    listed = tmp_path / "providers.yaml"
    listed.write_text(
        "providers:\n  - {name: groq, base_url: 'https://g/v1', model: x, key: GROQ_API_KEY}\n",
        encoding="utf-8",
    )
    twice = tmp_path / "twice.yaml"
    twice.write_text(
        "providers:\n"
        "  - {name: local, base_url: 'https://g', model: x, key: K}\n"
        "  - {name: local, base_url: 'https://g', model: y, key: K}\n",
        encoding="utf-8",
    )

    assert [p.name for p in load_providers(listed)] == ["groq"]
    assert load_providers(tmp_path / "none.yaml") == []
    with pytest.raises(ValidationError, match="unique and not 'local'"):
        load_providers(twice)


def test_a_key_is_found_in_the_environment_over_the_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_KEY=from-file\nMISTRAL_KEY=\nGEMINI_KEY=file\n", encoding="utf-8")

    found = environment(env_file, {"GEMINI_KEY": "from-process"})

    assert found == {"GROQ_KEY": "from-file", "GEMINI_KEY": "from-process"}
    assert environment(tmp_path / "missing", {}) == {}


def test_only_providers_with_a_key_are_asked_and_the_paid_only_if_allowed() -> None:
    configs = [provider("groq"), provider("mistral"), provider("claude", paid=True)]
    env = {"GROQ_KEY": "a", "CLAUDE_KEY": "b"}

    assert [c.name for c, _ in usable(configs, env)] == ["groq"]
    assert [c.name for c, _ in usable(configs, env, paid=True)] == ["groq", "claude"]
    assert [c.name for c, _ in usable(configs, env, only="claude")] == ["claude"]
    assert usable(configs, env, only="local") == []
    with pytest.raises(LookupError, match="No provider 'mistral' with its key set"):
        usable(configs, env, only="mistral")
    assert usable(configs, env)[0][1].get_secret_value() == "a"


# --- What a refusal says --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "until"),
    [
        (httpx.Response(429, headers={"retry-after": "7"}), NOW + timedelta(seconds=7)),
        (httpx.Response(429, text='{"retryDelay": "30s"}'), NOW + timedelta(seconds=30)),
        (httpx.Response(429, text="Limit tokens per day (TPD) reached"),
         datetime(2026, 9, 30, tzinfo=UTC)),
        (httpx.Response(429, text="Monthly token limit exceeded"),
         datetime(2026, 10, 1, tzinfo=UTC)),
        (httpx.Response(429, headers={"retry-after": "soon"}), NOW + timedelta(minutes=1)),
    ],
)  # fmt: skip
def test_a_provider_out_of_quota_is_asked_again_when_it_says(
    response: httpx.Response, until: datetime
) -> None:
    assert quota_until(response, NOW) == until


def test_a_monthly_quota_comes_back_on_the_first_of_the_next_month_and_a_pause_reads_well() -> None:
    december = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
    out = httpx.Response(429, text="Requests per month exceeded")

    assert quota_until(out, december) == datetime(2027, 1, 1, tzinfo=UTC)
    assert [lasting(timedelta(seconds=s)) for s in (45, 720, 27_000, 259_200)] == [
        "45 s",
        "12 min",
        "7.5 h",
        "3.0 d",
    ]


def test_a_refusal_is_quota_or_unavailability() -> None:
    with pytest.raises(OutOfQuota):
        raise_for("p", httpx.Response(429), NOW)
    with pytest.raises(OutOfQuota):
        raise_for("p", httpx.Response(402, text="Insufficient credits"), NOW)
    with pytest.raises(Unavailable, match="key was refused") as refused:
        raise_for("p", httpx.Response(401, text="bad key"), NOW)
    assert refused.value.pause == timedelta(days=1)
    with pytest.raises(Unavailable, match="HTTP 503") as down:
        raise_for("p", httpx.Response(503, text="overloaded"), NOW)
    assert down.value.pause == FAILURE_PAUSE
    raise_for("p", httpx.Response(200), NOW)


def test_a_prompt_splits_before_its_question_and_a_fence_comes_off() -> None:
    assert split_stable("Rules\n\ndictionary\n\nQuestion: which?\nMore") == (
        "Rules\n\ndictionary\n",
        "Question: which?\nMore",
    )
    assert split_stable("No question here") == ("", "No question here")
    assert json_text('```json\n{"sql": "x"}\n```') == '{"sql": "x"}'
    assert json_text(' {"sql": "x"} ') == '{"sql": "x"}'


# --- The OpenAI-compatible API ----------------------------------------------------------------


def test_an_openai_style_provider_is_sent_the_schema_and_its_reply_is_read() -> None:
    usage = {"prompt_tokens": 900, "completion_tokens": 12,
             "prompt_tokens_details": {"cached_tokens": 800}}  # fmt: skip
    asked, sent = model(lambda r: chat('```json\n{"sql": "SELECT 1"}\n```', usage))

    answer = asked.ask("Rules\nQuestion: q", Shape)

    body = json.loads(sent[0].content)
    assert answer == Shape(sql="SELECT 1")
    assert sent[0].url == "https://llm.test/v1/chat/completions"
    assert sent[0].headers["authorization"] == f"Bearer {KEY.get_secret_value()}"
    assert body["response_format"]["json_schema"]["schema"] == Shape.model_json_schema()
    assert '"sql"' in body["messages"][0]["content"] and body["temperature"] == 0
    assert asked.spent == {"input": 900, "output": 12, "cached": 800, "calls": 1}


def test_a_provider_that_refuses_the_schema_is_asked_in_json_mode_from_then_on() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text="response_format json_schema is not supported")
        return chat('{"sql": "SELECT 2"}')

    asked, sent = model(respond)

    assert asked.ask("q", Shape).sql == "SELECT 2"
    assert asked.ask("q", Shape).sql == "SELECT 2"
    formats = [json.loads(r.content)["response_format"]["type"] for r in sent]
    assert formats == ["json_schema", "json_object", "json_object"]


def test_a_provider_s_address_is_extended_not_replaced() -> None:
    """`.../v1beta/openai/` + `chat/completions`: without the slash httpx would drop the
    last segment of the base."""
    seen: list[str] = []
    transport = httpx.MockTransport(lambda r: seen.append(str(r.url)) or chat('{"sql": "x"}'))
    gemini = provider(base_url="https://llm.test/v1beta/openai")

    with provider_client(gemini, transport) as client:
        ApiModel(client, gemini, KEY).ask("q", Shape)

    assert seen == ["https://llm.test/v1beta/openai/chat/completions"]


def test_an_empty_or_malformed_reply_and_a_dead_line_are_errors() -> None:
    empty, _ = model(lambda r: chat(""))
    malformed, sent = model(lambda r: chat('{"sq'))

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down, _ = model(dead)

    with pytest.raises(Unavailable, match="empty reply"):
        empty.ask("q", Shape)
    with pytest.raises(ValidationError):
        malformed.ask("q", Shape)
    assert len(sent) == 2  # a malformed reply is asked for twice, then raised
    with pytest.raises(Unavailable, match="no answer: ConnectError"):
        down.ask("q", Shape)


# --- Anthropic's API --------------------------------------------------------------------------


def test_anthropic_is_asked_through_a_tool_and_the_stable_head_is_cached() -> None:
    reply = {
        "content": [{"type": "text", "text": "Here"},
                    {"type": "tool_use", "name": "reply", "input": {"sql": "SELECT 3"}}],
        "usage": {"input_tokens": 40, "output_tokens": 9, "cache_read_input_tokens": 4000,
                  "cache_creation_input_tokens": 0},
    }  # fmt: skip
    asked, sent = model(lambda r: httpx.Response(200, json=reply), provider(kind="anthropic"))

    answer = asked.ask("Rules and the dictionary\nQuestion: which?", Shape)

    body = json.loads(sent[0].content)
    assert answer.sql == "SELECT 3"
    assert sent[0].url == "https://llm.test/v1/messages"
    assert sent[0].headers["x-api-key"] == KEY.get_secret_value()
    assert body["system"] == [{"type": "text", "text": "Rules and the dictionary",
                               "cache_control": {"type": "ephemeral"}}]  # fmt: skip
    assert body["messages"] == [{"role": "user", "content": "Question: which?"}]
    assert body["tool_choice"] == {"type": "tool", "name": "reply"}
    assert asked.spent == {"input": 4040, "output": 9, "cached": 4000, "calls": 1}


def test_anthropic_without_a_head_or_a_tool_call() -> None:
    no_tool, sent = model(lambda r: httpx.Response(200, json={"content": []}),
                          provider(kind="anthropic"))  # fmt: skip

    with pytest.raises(Unavailable, match="without the tool call"):
        no_tool.ask("Just a prompt", Shape)
    assert "system" not in json.loads(sent[0].content)


# --- The chain ------------------------------------------------------------------------------


def test_the_first_provider_with_quota_answers_and_the_rest_wait(tmp_path: Path) -> None:
    out = OutOfQuota("groq", NOW + timedelta(hours=3), "HTTP 429: tokens per day")
    groq = Member("groq", out)
    gemini = Member("gemini", {"sql": "SELECT 1"}, spent={"input": 10, "output": 2})
    local = Answers()
    state = tmp_path / "providers.json"
    chain = Chain([groq, gemini], local, Cooldowns(state), clock=lambda: NOW)  # type: ignore[list-item]

    assert chain.ask("q", Shape).sql == "SELECT 1"
    assert chain.last == "gemini/m" and local.asked == 0
    # The next run reads the file: groq is not asked until its quota comes back.
    later = Chain([Member("groq"), Member("gemini", {"sql": "SELECT 2"})], local,  # type: ignore[list-item]
                  Cooldowns(state), clock=lambda: NOW + timedelta(hours=1))  # fmt: skip
    assert later.ask("q", Shape).sql == "SELECT 2"
    saved = json.loads(state.read_text(encoding="utf-8"))
    # When it ran out, how long it takes to come back, and when it does.
    assert saved["aside"]["groq"] == {
        "since": NOW.isoformat(),
        "until": (NOW + timedelta(hours=3)).isoformat(),
        "pause_seconds": 10_800,
        "reason": "HTTP 429: tokens per day",
    }
    assert saved["spent"]["2026-09-29"]["gemini"] == {"input": 10, "output": 2}


def test_a_short_limit_is_waited_out_once_and_a_long_one_moves_on() -> None:
    slept: list[float] = []
    soon = OutOfQuota("groq", NOW + timedelta(seconds=5), "per minute")
    groq = Member("groq", soon, {"sql": "SELECT 1"})
    chain = Chain([groq], Answers(), clock=lambda: NOW, sleep=slept.append)  # type: ignore[list-item]

    assert chain.ask("q", Shape).sql == "SELECT 1" and slept == [5.0]

    twice = Member("groq", soon, soon)
    local = Answers({"sql": "LOCAL"})
    fallback = Chain([twice], local, clock=lambda: NOW, sleep=slept.append)  # type: ignore[list-item]
    assert fallback.ask("q", Shape).sql == "LOCAL" and fallback.last == "local"


def test_a_provider_down_or_wrong_is_set_aside_and_the_local_model_answers() -> None:
    wrong = ValidationError.from_exception_data("Shape", [])
    down = Member("down", Unavailable("down", "HTTP 503"))
    garbled = Member("garbled", wrong)
    cooldowns = Cooldowns()
    chain = Chain([down, garbled], Answers({"sql": "LOCAL"}), cooldowns,  # type: ignore[list-item]
                  clock=lambda: NOW, local_name="qwen@abc")  # fmt: skip

    assert chain.ask("q", Shape).sql == "LOCAL" and chain.last == "qwen@abc"
    assert cooldowns.until("down", NOW) == NOW + FAILURE_PAUSE
    assert "not a Shape" in cooldowns.report()["aside"]["garbled"]["reason"]
    assert chain.names() == ["down/m", "garbled/m", "qwen@abc"]


def test_without_the_local_model_a_chain_can_run_dry() -> None:
    out = OutOfQuota("groq", NOW + timedelta(days=1), "daily")
    chain = Chain([Member("groq", out)], None, clock=lambda: NOW)  # type: ignore[list-item]

    with pytest.raises(NoModelLeft, match="groq until 2026-09-30"):
        chain.ask("q", Shape)
    assert chain.names() == ["groq/m"]


def test_a_state_file_that_cannot_be_read_starts_afresh(tmp_path: Path) -> None:
    broken = tmp_path / "providers.json"
    broken.write_text("{not json", encoding="utf-8")

    assert Cooldowns(broken).report() == {"aside": {}, "spent": {}}


def test_a_week_of_spending_is_kept() -> None:
    cooldowns = Cooldowns()
    for day in range(1, 10):
        cooldowns.spend("groq", {"input": day}, f"2026-09-{day:02d}")
    cooldowns.spend("groq", {"input": 1}, "2026-09-09")

    spent = cooldowns.report()["spent"]
    assert sorted(spent) == [f"2026-09-{d:02d}" for d in range(3, 10)]
    assert spent["2026-09-09"]["groq"] == {"input": 10}


# --- Replies kept -------------------------------------------------------------------------


def test_a_prompt_asked_before_is_answered_from_the_cache(tmp_path: Path) -> None:
    inner = Answers({"sql": "SELECT 1"}, {"sql": "SELECT 2"})
    inner.last = "groq/m"  # type: ignore[attr-defined]
    cache = ReplyCache(tmp_path / "replies.sqlite")
    cached = CachedGenerator(inner, cache, clock=lambda: NOW)

    first, second = cached.ask("q", Shape), cached.ask("q", Shape)
    other = cached.ask("another", Shape)

    assert first == second == Shape(sql="SELECT 1") and other.sql == "SELECT 2"
    assert inner.asked == 2 and cached.last == "groq/m"
    cached.ask("q", Shape)
    assert cached.last == "cache"
    stale = CachedGenerator(Answers({"sql": "NEW"}), cache,
                            clock=lambda: NOW + timedelta(days=31))  # fmt: skip
    assert stale.ask("q", Shape).sql == "NEW"  # a month old: asked again
    cache.close()


def test_a_kept_reply_of_another_shape_is_asked_again(tmp_path: Path) -> None:
    class Wider(BaseModel):
        sql: str
        why: str

    cache = ReplyCache(tmp_path / "replies.sqlite")
    cache.put("x", '{"sql": "SELECT 1"}', None, NOW)
    inner = Answers({"sql": "SELECT 1", "why": "because"})
    cached = CachedGenerator(inner, cache, clock=lambda: NOW)
    cached._cache.get = lambda key, now: '{"sql": "SELECT 1"}'  # type: ignore[method-assign]

    assert cached.ask("q", Wider).why == "because" and inner.asked == 1
    cache.close()


# --- The chain the agent is given -----------------------------------------------------------

LISTED = """providers:
  - {name: free, base_url: 'https://llm.test/v1', model: m-1, key: FREE_KEY}
  - {name: keyless, base_url: 'https://llm.test/v1', model: m-2, key: NOBODY_KEY}
  - {name: paid, base_url: 'https://llm.test/v1', model: m-3, key: PAID_KEY, paid: true}
"""


@pytest.fixture
def listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    path = tmp_path / "providers.yaml"
    path.write_text(LISTED, encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # no .env here: the keys come from the environment only
    monkeypatch.setenv("FREE_KEY", "k1")
    monkeypatch.setenv("PAID_KEY", "k2")
    monkeypatch.delenv("NOBODY_KEY", raising=False)
    return Settings(providers_file=path, data_dir=tmp_path / "data")


LOCAL_MODEL = SimpleNamespace(model="qwen3.5:4b", digest=lambda: "abc123")


def test_the_agent_asks_the_providers_with_keys_then_its_local_model(listed: Settings) -> None:
    with ExitStack() as stack:
        chained, identity = cli.hosted_chain(listed, stack, LOCAL_MODEL, cache=True)  # type: ignore[arg-type]
        assert isinstance(chained, CachedGenerator)
        assert identity == "free/m-1 > qwen3.5:4b@abc123"
        _, alone = cli.hosted_chain(listed, stack, LOCAL_MODEL, only="local")  # type: ignore[arg-type]
        assert alone == "qwen3.5:4b@abc123"
        only, _ = cli.hosted_chain(listed, stack, LOCAL_MODEL, only="paid")  # type: ignore[arg-type]
        assert isinstance(only, Chain) and only.names() == ["paid/m-3"]  # no fallback
        paid = listed.model_copy(update={"paid_providers": True})
        _, both = cli.hosted_chain(paid, stack, LOCAL_MODEL)  # type: ignore[arg-type]
        assert both == "free/m-1 > paid/m-3 > qwen3.5:4b@abc123"
    assert (listed.data_dir / "llm" / "replies.sqlite").is_file()


def test_the_providers_command_says_who_is_ready_aside_or_keyless(
    listed: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    cooldowns = Cooldowns(listed.data_dir / "llm" / "providers.json")
    now = datetime.now(UTC)
    cooldowns.set_aside("paid", now + timedelta(hours=2), "HTTP 429: daily", since=now)
    cooldowns.spend("free", {"input": 1200, "output": 30, "cached": 1000, "calls": 2},
                    "2026-09-29")  # fmt: skip
    monkeypatch.setenv("MLOPS_PROVIDERS_FILE", str(listed.providers_file))
    monkeypatch.setenv("MLOPS_DATA_DIR", str(listed.data_dir))
    monkeypatch.setenv("MLOPS_PAID_PROVIDERS", "true")

    def served(config: ProviderConfig) -> Any:
        return provider_client(config, httpx.MockTransport(lambda r: chat('{"ok": true}')))

    monkeypatch.setattr(providers, "provider_client", served)
    result = CliRunner().invoke(cli.app, ["agent", "providers", "--check"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "1. free - m-1: ready"
    assert lines[1] == "2. keyless - m-2: no key (NOBODY_KEY is not set)"
    # When it ran out, for how long, and until when.
    assert lines[2].startswith(f"3. paid - m-3: set aside since {now:%Y-%m-%d %H:%M}, for 2.0 h, ")
    assert "until " in lines[2] and lines[2].endswith("UTC: HTTP 429: daily")
    assert lines[3].startswith("4. local")
    assert "2026-09-29 free: 2 calls, 1,200 tokens in (1,000 cached), 30 out" in result.output
    assert "check free: answered" in result.output
    assert "k1" not in result.output and "k2" not in result.output  # never a key

    # A state file written before the start was kept still reads, with what it has.
    state = listed.data_dir / "llm" / "providers.json"
    older = json.loads(state.read_text(encoding="utf-8"))
    del older["aside"]["paid"]["since"]
    state.write_text(json.dumps(older), encoding="utf-8")
    again = CliRunner().invoke(cli.app, ["agent", "providers"])
    assert again.output.splitlines()[2].startswith("3. paid - m-3: set aside until ")


def test_a_paid_provider_waits_for_permission_and_a_bad_key_is_reported(
    listed: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLOPS_PROVIDERS_FILE", str(listed.providers_file))
    monkeypatch.setenv("MLOPS_DATA_DIR", str(listed.data_dir))

    def refused(config: ProviderConfig) -> Any:
        return provider_client(config, httpx.MockTransport(lambda r: httpx.Response(401)))

    monkeypatch.setattr(providers, "provider_client", refused)
    result = CliRunner().invoke(cli.app, ["agent", "providers", "--check"])

    assert "3. paid - m-3: paid: left out (MLOPS_PAID_PROVIDERS is not true)" in result.output
    assert "check free: FAILED - free: the key was refused (HTTP 401)" in result.output
    empty = CliRunner().invoke(cli.app, ["agent", "providers"],
                               env={"MLOPS_PROVIDERS_FILE": "nowhere.yaml"})  # fmt: skip
    assert "no providers; the agent asks its local model" in empty.output


def test_a_traced_call_says_which_model_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    from mlops_core.agent.graph import TracedGenerator

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("providers")
    chain = Chain([Member("groq", {"sql": "SELECT 1"})], Answers(), clock=lambda: NOW)  # type: ignore[list-item]

    with mlflow.start_span("agent"):
        TracedGenerator(chain).ask("q", Shape)
    mlflow.flush_trace_async_logging()

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())  # type: ignore[arg-type]
    assert trace is not None
    called = next(span for span in trace.data.spans if span.name == "Shape")
    assert called.attributes["answered_by"] == "groq/m"
