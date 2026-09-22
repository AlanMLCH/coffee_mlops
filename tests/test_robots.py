"""robots.txt is asked before any page is read, and silence is not permission.

What the rules say is standard; what an unreadable robots.txt means is the part a
scraper gets wrong in its own favour. RFC 9309: none there allows everything, a server
that cannot answer allows nothing.
"""

from pathlib import Path

import httpx
import pytest

from mlops_core.data.api import ApiClient
from mlops_core.data.robots import RobotsDisallowed, RobotsPolicy

AGENT = "coffee-mlops/0.1.0 (+https://github.com/AlanMLCH/coffee_mlops)"


def policy(tmp_path: Path, *responses: httpx.Response) -> tuple[RobotsPolicy, list[str]]:
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return responses[min(len(asked) - 1, len(responses) - 1)]

    client = ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        cache_dir=tmp_path,
        min_interval_s=0.0,
        max_attempts=2,
        sleep=lambda _: None,
    )
    return RobotsPolicy(client, AGENT), asked


def test_what_the_rules_forbid_is_refused(tmp_path: Path) -> None:
    rules = "User-agent: *\nDisallow: /checkout\nDisallow: /admin\n"
    robots, _ = policy(tmp_path, httpx.Response(200, text=rules))

    robots.check("https://shop.test/products.json?limit=250&page=1")
    with pytest.raises(RobotsDisallowed, match="/checkout"):
        robots.check("https://shop.test/checkout")


def test_rules_naming_this_agent_apply_to_it(tmp_path: Path) -> None:
    rules = "User-agent: coffee-mlops\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
    robots, _ = policy(tmp_path, httpx.Response(200, text=rules))

    with pytest.raises(RobotsDisallowed):
        robots.check("https://shop.test/products.json")


def test_a_requested_crawl_delay_is_reported(tmp_path: Path) -> None:
    robots, _ = policy(tmp_path, httpx.Response(200, text="User-agent: *\nCrawl-delay: 7\n"))

    assert robots.crawl_delay("https://shop.test/") == 7.0


def test_the_rules_are_fetched_once_per_site(tmp_path: Path) -> None:
    robots, asked = policy(tmp_path, httpx.Response(200, text="User-agent: *\nDisallow:\n"))

    robots.check("https://shop.test/a")
    robots.check("https://shop.test/b")
    assert robots.crawl_delay("https://shop.test/c") is None

    assert asked == ["https://shop.test/robots.txt"]


def test_no_robots_txt_allows_everything(tmp_path: Path) -> None:
    robots, _ = policy(tmp_path, httpx.Response(404))

    robots.check("https://shop.test/anything")


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(httpx.Response(503), id="gave-up-on-a-failing-server"),
        pytest.param(httpx.Response(501), id="server-error-not-worth-retrying"),
    ],
)
def test_a_server_that_cannot_answer_allows_nothing(tmp_path: Path, answer: httpx.Response) -> None:
    """Silence from a failing server is not permission."""
    robots, _ = policy(tmp_path, answer)

    with pytest.raises(RobotsDisallowed):
        robots.check("https://shop.test/products.json")
