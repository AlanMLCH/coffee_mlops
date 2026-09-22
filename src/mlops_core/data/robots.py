"""robots.txt, asked before every page a domain reads from a site that is not an API.

An API is an invitation with terms; a shop's pages are not. Before this project reads a
page, it asks the site's robots.txt whether this project's agent - the same identifiable
user agent every request carries - may, and it honours a `Crawl-delay`. The rules come
through `ApiClient`, so they are rate-limited and cached like any other request.

What an unreadable robots.txt means follows RFC 9309: a 4xx (there is none) allows
everything, while a server that cannot answer at all allows nothing - silence from a
failing server is not permission.
"""

import logging
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from mlops_core.data.api import ApiClient

logger = logging.getLogger(__name__)


# What a missing robots.txt and an unreachable one mean, written as robots.txt itself.
ALLOW_ALL: list[str] = []
CLOSED = ["User-agent: *", "Disallow: /"]


class RobotsDisallowed(Exception):
    """A page this project's agent may not read. Callers skip the site and say so."""


@dataclass
class RobotsPolicy:
    """The robots.txt rules of every site asked about, fetched once per site."""

    client: ApiClient
    agent: str
    _rules: dict[str, urllib.robotparser.RobotFileParser] = field(
        default_factory=dict, init=False, repr=False
    )

    def check(self, url: str) -> None:
        """Raise `RobotsDisallowed` unless this agent may read `url`."""
        if not self._rules_for(url).can_fetch(self.agent, url):
            raise RobotsDisallowed(f"robots.txt disallows {url} for {self.agent}")

    def crawl_delay(self, url: str) -> float | None:
        """The seconds the site asks between requests, if it asks."""
        delay = self._rules_for(url).crawl_delay(self.agent)
        return float(delay) if delay is not None else None

    def _rules_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._rules:
            self._rules[origin] = self._load(origin)
        return self._rules[origin]

    def _load(self, origin: str) -> urllib.robotparser.RobotFileParser:
        rules = urllib.robotparser.RobotFileParser()
        try:
            text = self.client.get_text(f"{origin}/robots.txt", cache_key=f"robots:{origin}")
        except httpx.HTTPStatusError as refused:
            if refused.response.status_code < 500:
                logger.info(
                    "%s has no robots.txt (%d): allowed", origin, refused.response.status_code
                )
                rules.parse(ALLOW_ALL)
            else:  # a server error the client does not retry: as closed as one it gave up on
                logger.warning(
                    "%s/robots.txt answered %d: closed", origin, refused.response.status_code
                )
                rules.parse(CLOSED)
            return rules
        except RuntimeError:
            # The client gave up on a failing server: nothing is allowed until it answers.
            logger.warning("%s/robots.txt could not be read: treating the site as closed", origin)
            rules.parse(CLOSED)
            return rules
        rules.parse(text.splitlines())
        return rules
