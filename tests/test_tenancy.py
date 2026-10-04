"""Domains are tenants, isolated from each other: none reads another's data. A domain can
hold subdomains - one per business - that read the tables of their parent they list
(`parent`), named one by one, read-only, and nothing of their siblings or of any other
domain; and the parent reads nothing of theirs. Whatever a query tries."""

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from mlops_core import adapter as adapters
from mlops_core.adapter import available_domains, domain_dir, load_adapter
from mlops_core.agent import dictionary
from mlops_core.agent.dictionary import domain_dictionary
from mlops_core.agent.sql import Refused, read_only, run_select, views
from mlops_core.catalog import connect
from mlops_core.config import (
    AnalysisConfig,
    DomainConfig,
    MonitoringConfig,
    ParentTables,
)
from mlops_core.storage import latest_data_version, prune_layers, table_path, write_table
from tests.test_train import toy_model

MARKET = ParentTables(domain="market", tables=["clean.prices", "analysis.trend"])


def tenant(name: str, parent: ParentTables | None = None) -> DomainConfig:
    return DomainConfig(
        name=name,
        sources={},
        models=[toy_model()],
        analysis=AnalysisConfig(min_rows=1, permutation_repeats=2, published_figures=[]),
        monitoring=MonitoringConfig(drift_share=0.5),
        parent=parent,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A market domain that keeps one table from its shops, two shops of it, and a domain
    of games that has nothing to do with either."""
    data = tmp_path / "data"
    market = data / "market"
    write_table(
        pl.DataFrame({"month": ["2026-08"], "index": [142.0]}), market / "clean" / "prices", {}
    )
    write_table(pl.DataFrame({"year": [2026], "slope": [0.4]}), market / "analysis" / "trend", {})
    write_table(pl.DataFrame({"buyer": ["secret"]}), market / "clean" / "buyers", {})
    write_table(pl.DataFrame({"sold": [3]}), shop(data, "corner") / "clean" / "sales", {})
    write_table(pl.DataFrame({"sold": [99]}), shop(data, "rival") / "clean" / "sales", {})
    write_table(pl.DataFrame({"score": [9.5]}), data / "games" / "clean" / "scores", {})
    return data


def shop(data: Path, name: str) -> Path:
    return data / "market" / "subdomains" / name


def files(table: Path) -> str:
    return f"{table.as_posix()}/*/*.parquet"


def test_a_subdomain_names_what_it_reads_of_its_parent_and_nothing_else() -> None:
    with pytest.raises(ValidationError, match=r"layer\.table"):
        ParentTables(domain="market", tables=["prices"])
    with pytest.raises(ValidationError, match="names itself as its parent"):
        tenant("market", MARKET)
    with pytest.raises(ValidationError, match="lower case"):
        tenant("Corner", MARKET)

    corner = tenant("corner", MARKET)

    assert (corner.tenant, corner.home) == ("market/corner", Path("market/subdomains/corner"))
    assert corner.parent_tables == {"market.clean.prices", "market.analysis.trend"}
    assert corner.readable("sales") == "sales"  # its own
    assert corner.readable("market.clean.prices") == "market.clean.prices"
    for kept in ("market.clean.buyers", "rival.clean.sales", "games.clean.scores"):
        with pytest.raises(ValueError, match="is not a table corner lists"):
            corner.readable(kept)


def test_a_domain_reads_no_other_domains_tables() -> None:
    games = tenant("games")

    assert (games.tenant, games.home, games.parent_tables) == ("games", Path("games"), set())
    with pytest.raises(ValueError, match="it reads no other domain"):
        games.readable("market.clean.prices")


def test_a_qualified_name_resolves_to_the_parent_only(root: Path) -> None:
    corner = shop(root, "corner")

    assert table_path(corner, "sales") == corner / "clean" / "sales"
    assert table_path(corner, "market.clean.prices") == root / "market" / "clean" / "prices"
    assert latest_data_version(corner, ["sales", "market.clean.prices"]) is not None
    for elsewhere in ("rival.clean.sales", "games.clean.scores"):
        with pytest.raises(ValueError, match="is not a table of the domain"):
            table_path(corner, elsewhere)
    with pytest.raises(ValueError, match="is not a table of the domain"):
        table_path(root / "games", "market.clean.prices")


def test_a_subdomain_queries_what_it_lists_and_nothing_else(root: Path) -> None:
    con = read_only(shop(root, "corner"), MARKET)

    assert views(con) == {"clean.sales", "market.clean.prices", "market.analysis.trend"}
    joined = run_select(con, "SELECT s.sold, p.index FROM clean.sales s, market.clean.prices p")
    assert joined.rows == [(3, 142.0)]
    for kept in (
        root / "market" / "clean" / "buyers",  # its parent's, not listed
        shop(root, "rival") / "clean" / "sales",  # a sibling's
        root / "games" / "clean" / "scores",  # another domain's
    ):
        with pytest.raises((duckdb.Error, Refused)):
            run_select(con, f"SELECT * FROM read_parquet('{files(kept)}')")


def test_a_domain_reads_none_of_its_subdomains_data(root: Path) -> None:
    con = read_only(root / "market")

    assert views(con) == {"clean.prices", "clean.buyers", "analysis.trend"}
    with pytest.raises((duckdb.Error, Refused)):
        run_select(
            con, f"SELECT * FROM read_parquet('{files(shop(root, 'corner') / 'clean' / 'sales')}')"
        )
    with pytest.raises(duckdb.Error):
        run_select(con, "SELECT * FROM games.clean.scores")


def test_a_parent_table_not_built_yet_is_not_offered(root: Path) -> None:
    later = ParentTables(domain="market", tables=["clean.prices", "predictions.next"])

    con = connect(shop(root, "corner"), later)

    assert views(con) == {"clean.sales", "market.clean.prices"}


def test_pruning_a_domain_leaves_its_subdomains_alone(root: Path) -> None:
    newer = datetime.now(UTC) + timedelta(minutes=1)
    write_table(pl.DataFrame({"month": ["2026-09"], "index": [143.0]}),
                root / "market" / "clean" / "prices", {}, newer)  # fmt: skip
    write_table(pl.DataFrame({"sold": [4]}), shop(root, "corner") / "clean" / "sales", {}, newer)

    pruned = prune_layers(root / "market", keep=1)

    assert pruned == {"clean/prices": 1}
    assert len(list((shop(root, "corner") / "clean" / "sales").glob("built_at=*"))) == 2


def test_the_agent_reads_a_parent_tables_section_from_the_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    homes = {"market/corner": tmp_path / "corner", "market": tmp_path / "market"}
    for home in homes.values():
        home.mkdir()
    (homes["market/corner"] / "data_dictionary.md").write_text(
        "# Shop\n\n## `clean.sales` — one sale\n\n| `sold` | Int | Units |\n", encoding="utf-8"
    )
    (homes["market"] / "data_dictionary.md").write_text(
        "# Market\n\n## `clean.prices` — one month\n\nJoin `clean.prices` on month.\n\n"
        "## `clean.buyers` — kept\n\nNot lent.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dictionary, "domain_dir", lambda name: homes[name])

    text = domain_dictionary(
        tenant("corner", ParentTables(domain="market", tables=["clean.prices"]))
    )

    sections = dictionary.table_sections(text)
    assert set(sections) == {"clean.sales", "market.clean.prices"}
    assert "Join `market.clean.prices` on month." in sections["market.clean.prices"]
    assert "buyers" not in text


def test_a_subdomain_is_loaded_only_from_the_domain_it_names_as_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a config says cannot move a tenant: a domain that names a parent, or a
    subdomain that names another one, is refused when it is loaded."""
    configs = {
        "corner": tenant("corner", MARKET),
        "stray": tenant("stray", ParentTables(domain="games", tables=["clean.scores"])),
    }
    packages = {
        "market": SimpleNamespace(
            adapter=lambda: SimpleNamespace(config=tenant("market")),
            subdomains=lambda: list(configs),
            subdomain=lambda name: SimpleNamespace(config=configs[name]),
            subdomain_dir=lambda name: Path("shops"),
        ),
        "games": SimpleNamespace(adapter=lambda: SimpleNamespace(config=tenant("games", MARKET))),
    }
    monkeypatch.setattr(adapters, "_package", lambda name: packages[name])
    monkeypatch.setattr(adapters, "available_domains", lambda: sorted(packages))

    assert load_adapter("market/corner").config.tenant == "market/corner"
    assert adapters.available_tenants() == ["games", "market", "market/corner", "market/stray"]
    assert domain_dir("market/corner") == Path("shops")
    with pytest.raises(ValueError, match="must be named stray and have market as its parent"):
        load_adapter("market/stray")
    with pytest.raises(ValueError, match="No subdomain 'rival' in market"):
        load_adapter("market/rival")
    with pytest.raises(ValueError, match="games is a domain: it cannot name a parent"):
        load_adapter("games")


def test_the_coffee_shops_read_coffee_and_not_each_other() -> None:
    shops = [
        load_adapter(f"coffee/{name}").config for name in adapters.available_subdomains("coffee")
    ]

    assert [config.tenant for config in shops] == ["coffee/cafe_de_barrio", "coffee/cafe_de_paso"]
    for config in shops:
        assert {name.split(".")[0] for name in config.parent_tables} == {"coffee"}
        others = [other.name for other in shops if other is not config]
        for other in others:
            with pytest.raises(ValueError, match="is not a table"):
                config.readable(f"{other}.clean.sales")
    assert len({config.models[0].training.registered_model for config in shops}) == len(shops)


def test_no_domain_imports_another_domains_code() -> None:
    """Domains share no data and no code: a subdomain's code lives in its parent's
    package, and no package imports another domain's."""
    for domain in available_domains():
        others = {f"domains.{other}" for other in available_domains() if other != domain}
        for path in domain_dir(domain).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            # Whole package names: `domains.coffee` is a prefix of `domains.coffee_extra`.
            reached = {
                name
                for name in imported
                if any(name == other or name.startswith(f"{other}.") for other in others)
            }
            assert not reached, f"{path} imports {reached}"
