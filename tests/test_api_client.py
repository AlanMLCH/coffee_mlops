"""The client is what stands between this project and someone else's free service."""

import logging
import time
from pathlib import Path

import httpx
import pytest

from coffee_mlops.data.api import ApiClient

URL = "https://example.test/data"


class Recorder:
    """Replays a list of responses and counts how many requests actually went out."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def build(recorder: Recorder, tmp_path: Path, **kwargs: object) -> tuple[ApiClient, list[float]]:
    slept: list[float] = []
    client = ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(recorder.handler)),
        cache_dir=tmp_path / "cache",
        min_interval_s=kwargs.pop("min_interval_s", 0.0),  # type: ignore[arg-type]
        backoff_s=kwargs.pop("backoff_s", 0.1),  # type: ignore[arg-type]
        sleep=slept.append,
        now=lambda: 0.0,  # time never advances, so the rate limit always has to wait
        **kwargs,  # type: ignore[arg-type]
    )
    return client, slept


def test_a_second_identical_request_is_served_from_disk(tmp_path: Path) -> None:
    recorder = Recorder(httpx.Response(200, json=[{"id": 1}]))
    client, _ = build(recorder, tmp_path)

    first = client.get_json(URL, cache_key="page-1")
    second = client.get_json(URL, cache_key="page-1")

    assert first == second == [{"id": 1}]
    assert recorder.calls == 1  # the service was asked once


@pytest.mark.parametrize(("age_s", "requests"), [(30, 1), (120, 2)])
def test_a_cached_answer_is_only_trusted_while_it_is_young(
    tmp_path: Path, age_s: float, requests: int
) -> None:
    """A cache that never expires turns a live register into a snapshot that keeps calling
    itself a fresh pull: re-runs reported 'unchanged' because nothing was ever asked."""
    recorder = Recorder(httpx.Response(200, json=[{"id": 1}]))
    client, _ = build(recorder, tmp_path, max_age_s=60.0, clock=lambda: time.time() + age_s)

    client.get_json(URL, cache_key="page-1")
    client.get_json(URL, cache_key="page-1")

    assert recorder.calls == requests


def test_the_cache_never_carries_the_credential(tmp_path: Path) -> None:
    """DENUE puts its token in the URL path; a cache keyed on the URL would file it."""
    recorder = Recorder(httpx.Response(200, json=[]))
    client, _ = build(recorder, tmp_path)

    client.get_json(f"{URL}/super-secret-token", cache_key="denue:page-1")

    written = list((tmp_path / "cache").iterdir())
    assert written and all("super-secret-token" not in path.name for path in written)


def test_a_failing_service_is_retried_then_succeeds(tmp_path: Path) -> None:
    recorder = Recorder(httpx.Response(503), httpx.Response(200, json={"ok": True}))
    client, slept = build(recorder, tmp_path)

    assert client.get_json(URL, cache_key="k") == {"ok": True}
    assert recorder.calls == 2
    assert slept == [0.1]  # it waited before trying again


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_what_is_worth_retrying_is_retried(tmp_path: Path, status: int) -> None:
    recorder = Recorder(httpx.Response(status))
    client, _ = build(recorder, tmp_path, max_attempts=3)

    with pytest.raises(RuntimeError, match="Giving up after 3"):
        client.get_json(URL, cache_key="k")

    assert recorder.calls == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_our_own_mistakes_are_not_retried(tmp_path: Path, status: int) -> None:
    """A bad token or a wrong path will not fix itself, and retrying burns someone
    else's capacity to learn nothing."""
    recorder = Recorder(httpx.Response(status))
    client, _ = build(recorder, tmp_path)

    with pytest.raises(httpx.HTTPStatusError):
        client.get_json(URL, cache_key="k")

    assert recorder.calls == 1


def test_a_timeout_is_retried(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("too slow", request=request)
        return httpx.Response(200, json={"ok": True})

    client = ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(flaky)),
        cache_dir=tmp_path / "cache",
        min_interval_s=0.0,
        backoff_s=0.0,
        sleep=lambda _: None,
    )

    assert client.get_json(URL, cache_key="k") == {"ok": True}
    assert calls["n"] == 2


def test_requests_are_spaced_out(tmp_path: Path) -> None:
    recorder = Recorder(httpx.Response(200, json=[]))
    client, slept = build(recorder, tmp_path, min_interval_s=2.0)

    client.get_json(URL, cache_key="a")
    client.get_json(URL, cache_key="b")

    assert slept == [2.0]  # the first request goes out at once, the second waits its turn


def test_building_a_client_stops_request_urls_reaching_the_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DENUE's token is a path segment, so one httpx INFO line leaks a credential.
    The protection belongs to the client: a script that never touches the CLI still
    gets it."""
    logging.getLogger("httpx").setLevel(logging.DEBUG)  # as a stray basicConfig would
    caplog.set_level(logging.DEBUG)
    recorder = Recorder(httpx.Response(200, json=[]))
    client, _ = build(recorder, tmp_path)

    client.get_json(f"{URL}/super-secret-token", cache_key="k")

    assert "super-secret-token" not in caplog.text
    assert logging.getLogger("httpx").level == logging.WARNING
