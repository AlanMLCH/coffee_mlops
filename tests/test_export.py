"""A snapshot of the explorer: what goes in it, what a source's terms keep home, and that
the explorer can read it back as its data directory."""

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from mlops_core.catalog import connect
from mlops_core.config import (
    AnalysisConfig,
    DomainConfig,
    ExploreConfig,
    ExploreDataset,
    ExploreFinding,
    MapLayer,
    MapView,
    MonitoringConfig,
    ShowcaseConfig,
)
from mlops_core.explore import export
from mlops_core.explore.export import (
    SNAPSHOT_FILE,
    export_snapshot,
    named_tables,
    published_tables,
    unpack_snapshot,
)
from mlops_core.storage import write_table
from tests.test_train import toy_model

AT = datetime(2026, 10, 2, tzinfo=UTC)
OPEN = ShowcaseConfig()  # every table may go


def explore(showcase: ShowcaseConfig = OPEN) -> ExploreConfig:
    return ExploreConfig(
        title="Toy",
        view=MapView(latitude=0, longitude=0, zoom=5),
        layers=[MapLayer(name="Places", kind="points", sql="SELECT * FROM clean.places")],
        datasets=[
            ExploreDataset(
                name="Prices", table="clean.prices", measures={"n": "count(*)"}, dimensions=["k"]
            )
        ],
        findings=[ExploreFinding(title="Lots", text="Every lot.", sql="SELECT * FROM clean.lots")],
        showcase=showcase,
    )


def domain(showcase: ShowcaseConfig = OPEN) -> DomainConfig:
    return DomainConfig(
        name="toy",
        sources={},
        models=[toy_model()],
        analysis=AnalysisConfig(min_rows=1, permutation_repeats=2, published_figures=[]),
        monitoring=MonitoringConfig(drift_share=0.5),
        explore=explore(showcase),
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A domain's layers: three clean tables, a study, a figure and a monitor's verdict."""
    root = tmp_path / "data" / "toy"
    write_table(pl.DataFrame({"name": ["a", "b"]}), root / "clean" / "places", {})
    write_table(
        pl.DataFrame({"k": ["x", "y"], "daily": [True, False]}), root / "clean" / "prices", {}
    )
    write_table(pl.DataFrame({"lot": ["l1"]}), root / "clean" / "lots", {})
    write_table(pl.DataFrame({"unread": [1]}), root / "clean" / "unread", {})
    write_table(pl.DataFrame({"study": [1.0]}), root / "analysis" / "price_residuals", {})
    figures = root / "analysis" / "figures" / "built_at=20261002T000000Z"
    figures.mkdir(parents=True)
    (figures / "price_residuals.png").write_bytes(b"png")
    (figures / "manifest.json").write_text("{}", encoding="utf-8")  # complete
    verdict = root / "monitoring" / "price_drift" / "built_at=20261002T000000Z"
    verdict.mkdir(parents=True)
    (verdict / "verdict.json").write_text('{"model": "price"}', encoding="utf-8")
    (verdict / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def test_the_tables_a_page_names_are_found_in_its_queries() -> None:
    assert named_tables(explore()) == {"clean.places", "clean.prices", "clean.lots"}


def test_a_snapshot_holds_what_the_pages_read_and_what_the_terms_allow(
    data_dir: Path, tmp_path: Path
) -> None:
    terms = ShowcaseConfig(
        withheld={"clean.lots": "all rights reserved"},
        rows={"clean.prices": "NOT daily"},
        credits="Prices: someone.",
    )
    assert published_tables(explore(terms), data_dir) >= {
        "analysis.price_residuals",
        "monitoring.price_drift",
    }

    snapshot = export_snapshot(domain(terms), data_dir, tmp_path / "dist", AT)

    assert snapshot.archive.name == "toy-showcase-20261002.zip"
    assert snapshot.withheld == ["clean.lots"] and snapshot.filtered == ["clean.prices"]
    assert "clean.unread" not in snapshot.tables  # no page reads it
    names = zipfile.ZipFile(snapshot.archive).namelist()
    assert any(name.startswith("analysis/figures/") for name in names)
    assert any(name.endswith("verdict.json") for name in names)
    unpacked = tmp_path / "showcase" / "toy"
    described = unpack_snapshot(str(snapshot.archive), unpacked)
    assert described["withheld"] == {"clean.lots": "all rights reserved"}
    con = connect(unpacked)
    assert con.execute("SELECT k FROM clean.prices").fetchall() == [("y",)]  # its part only
    assert con.execute("SELECT count(*) FROM clean.places").fetchone() == (2,)
    tables = {
        row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    assert "lots" not in tables


def test_a_snapshot_is_unpacked_once_and_can_come_from_a_url(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = export_snapshot(domain(), data_dir, tmp_path / "dist", AT).archive
    asked: list[str] = []

    def served(url: str, timeout: float) -> io.BytesIO:
        asked.append(url)
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(export.urllib.request, "urlopen", served)
    target = tmp_path / "cloud" / "toy"

    first = unpack_snapshot("https://example.org/toy-showcase.zip", target)
    (target / "clean" / "places").rename(target / "clean" / "moved")  # read as it is now
    again = unpack_snapshot("https://example.org/toy-showcase.zip", target)

    assert asked == ["https://example.org/toy-showcase.zip"]
    assert first == again and first["domain"] == "toy"
    assert json.loads((target / SNAPSHOT_FILE).read_text(encoding="utf-8"))["tables"]


def test_a_table_not_built_goes_without_it_and_a_domain_without_a_map_has_nothing(
    data_dir: Path, tmp_path: Path
) -> None:
    config = domain()
    missing = config.model_copy(
        update={
            "explore": explore().model_copy(
                update={
                    "layers": [MapLayer(name="X", kind="points", sql="SELECT * FROM clean.gone")]
                }
            )
        }
    )

    assert "clean.gone" not in export_snapshot(missing, data_dir, tmp_path, AT).tables
    with pytest.raises(ValueError, match="no explorer"):
        export_snapshot(config.model_copy(update={"explore": None}), data_dir, tmp_path, AT)


def test_a_showcase_names_its_tables_with_their_layer() -> None:
    with pytest.raises(ValueError, match=r"layer\.table"):
        ShowcaseConfig(withheld={"lots": "why"})
