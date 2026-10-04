"""Domains are tenants: each sees its own tables, and another's only as it declares them
(`uses`) - named one by one, read-only. One that declares nothing sees nothing of any
other, whatever its queries try."""

import ast
from pathlib import Path

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from mlops_core.adapter import available_domains, domain_dir
from mlops_core.agent import dictionary
from mlops_core.agent.dictionary import domain_dictionary
from mlops_core.agent.sql import Refused, read_only, run_select, views
from mlops_core.catalog import connect
from mlops_core.config import (
    AnalysisConfig,
    DomainConfig,
    DomainUse,
    MonitoringConfig,
)
from mlops_core.storage import latest_data_version, table_path, write_table
from tests.test_train import toy_model

LENT = DomainUse(domain="market", tables=["clean.prices", "analysis.trend"])


def tenant(name: str, uses: list[DomainUse] | None = None) -> DomainConfig:
    return DomainConfig(
        name=name,
        sources={},
        models=[toy_model()],
        analysis=AnalysisConfig(min_rows=1, permutation_repeats=2, published_figures=[]),
        monitoring=MonitoringConfig(drift_share=0.5),
        uses=uses or [],
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Two domains' data: a market that lends two tables and keeps one, and a shop."""
    data = tmp_path / "data"
    market = data / "market"
    write_table(
        pl.DataFrame({"month": ["2026-08"], "index": [142.0]}), market / "clean" / "prices", {}
    )
    write_table(pl.DataFrame({"year": [2026], "slope": [0.4]}), market / "analysis" / "trend", {})
    write_table(pl.DataFrame({"buyer": ["secret"]}), market / "clean" / "buyers", {})
    write_table(pl.DataFrame({"sold": [3]}), data / "shop" / "clean" / "sales", {})
    return data


def test_a_domain_names_what_it_reads_of_another_by_layer_and_table() -> None:
    with pytest.raises(ValidationError, match=r"layer\.table"):
        DomainUse(domain="market", tables=["prices"])
    with pytest.raises(ValidationError, match="lists itself"):
        tenant("market", [LENT])
    with pytest.raises(ValidationError, match="once in `uses`"):
        tenant("shop", [LENT, LENT])

    shop = tenant("shop", [LENT])

    assert shop.used_tables == {"market.clean.prices", "market.analysis.trend"}
    assert shop.readable("sales") == "sales"  # its own
    assert shop.readable("market.clean.prices") == "market.clean.prices"
    with pytest.raises(ValueError, match=r"does not declare market.clean.buyers"):
        shop.readable("market.clean.buyers")


def test_a_lent_table_lives_beside_the_domain_under_the_same_root(root: Path) -> None:
    shop = root / "shop"

    assert table_path(shop, "sales") == shop / "clean" / "sales"
    assert table_path(shop, "market.clean.prices") == root / "market" / "clean" / "prices"
    assert latest_data_version(shop, ["sales", "market.clean.prices"]) is not None


def test_a_domain_queries_what_it_declares_and_nothing_else_of_another(root: Path) -> None:
    con = read_only(root / "shop", [LENT])

    assert views(con) >= {"clean.sales", "market.clean.prices", "market.analysis.trend"}
    assert "market.clean.buyers" not in views(con)
    joined = run_select(con, "SELECT s.sold, p.index FROM clean.sales s, market.clean.prices p")
    assert joined.rows == [(3, 142.0)]
    kept = (root / "market" / "clean" / "buyers").as_posix()
    with pytest.raises((duckdb.Error, Refused)):
        run_select(con, f"SELECT * FROM read_parquet('{kept}/*/*.parquet')")


def test_a_domain_that_declares_nothing_sees_nothing_of_another(root: Path) -> None:
    con = read_only(root / "shop")
    lent = (root / "market" / "clean" / "prices").as_posix()

    assert not any(name.startswith("market.") for name in views(con))
    with pytest.raises(duckdb.Error):
        run_select(con, "SELECT * FROM market.clean.prices")
    with pytest.raises((duckdb.Error, Refused)):
        run_select(con, f"SELECT * FROM read_parquet('{lent}/*/*.parquet')")


def test_a_lent_table_not_built_yet_is_not_offered(root: Path) -> None:
    later = DomainUse(domain="market", tables=["clean.prices", "predictions.next"])

    con = connect(root / "shop", [later])

    assert views(con) == {"clean.sales", "market.clean.prices"}


def test_the_agent_reads_a_lent_tables_section_from_the_domain_that_owns_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    homes = {"shop": tmp_path / "shop", "market": tmp_path / "market"}
    for home in homes.values():
        home.mkdir()
    (homes["shop"] / "data_dictionary.md").write_text(
        "# Shop\n\n## `clean.sales` — one sale\n\n| `sold` | Int | Units |\n", encoding="utf-8"
    )
    (homes["market"] / "data_dictionary.md").write_text(
        "# Market\n\n## `clean.prices` — one month\n\nJoin `clean.prices` on month.\n\n"
        "## `clean.buyers` — kept\n\nNot lent.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dictionary, "domain_dir", lambda name: homes[name])

    text = domain_dictionary(tenant("shop", [DomainUse(domain="market", tables=["clean.prices"])]))

    sections = dictionary.table_sections(text)
    assert set(sections) == {"clean.sales", "market.clean.prices"}
    assert "Join `market.clean.prices` on month." in sections["market.clean.prices"]
    assert "buyers" not in text


def test_no_domain_imports_another_domains_code() -> None:
    """A domain reads another's tables, never its code: the contract is the data."""
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
            # Whole package names: `domains.coffee` is a prefix of `domains.coffee_shop`.
            reached = {
                name
                for name in imported
                if any(name == other or name.startswith(f"{other}.") for other in others)
            }
            assert not reached, f"{path} imports {reached}"
