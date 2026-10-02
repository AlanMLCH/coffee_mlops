"""A snapshot of what the explorer shows, to publish where nothing else runs: `mlops export`.

The explorer reads its layers through DuckDB views over the newest partition of each table
(`mlops_core.catalog`). A snapshot is those partitions - only the tables its pages name,
with every study, figure and monitoring verdict - in a directory laid out as the domain's
data directory is, zipped. Opened by the explorer in showcase mode (`MLOPS_SHOWCASE`), it is
the same app without the agent: no local model, no index, no prediction API, no MLflow.

A table the domain's showcase config withholds - its source's terms keep it home - is not
copied, and a table it publishes in part is copied with its rows filtered. Nothing is
inferred: what goes is what the pages read and what the config allows.
"""

import json
import logging
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mlops_core.catalog import LAYERS, connect
from mlops_core.config import DomainConfig, ExploreConfig
from mlops_core.storage import latest_partition, write_table

logger = logging.getLogger(__name__)

SNAPSHOT_FILE = "SNAPSHOT.json"
FIGURES = "figures"
# Read by the pages without being named in the YAML: every model's studies and figures,
# and the monitor's verdicts in the Models tab.
WHOLE_LAYERS = ("analysis", "monitoring")
_TABLE = re.compile(rf"\b({'|'.join(LAYERS)})\.([A-Za-z_]\w*)\b")


@dataclass(frozen=True)
class Snapshot:
    """What an export wrote, and what it left out."""

    archive: Path
    tables: list[str] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)


def named_tables(explore: ExploreConfig) -> set[str]:
    """Every `layer.table` the explorer's pages name."""
    return {f"{layer}.{table}" for text in explore.queries for layer, table in _TABLE.findall(text)}


def published_tables(explore: ExploreConfig, data_dir: Path) -> set[str]:
    """The tables a snapshot holds before the config withholds any: the ones the pages
    name, and every built table of the layers the pages read whole."""
    whole = {
        f"{layer}.{table.name}"
        for layer in WHOLE_LAYERS
        if (data_dir / layer).is_dir()
        for table in (data_dir / layer).iterdir()
        if table.is_dir() and table.name != FIGURES
    }
    return named_tables(explore) | whole


def export_snapshot(
    config: DomainConfig, data_dir: Path, out_dir: Path, at: datetime | None = None
) -> Snapshot:
    """Copy what the explorer reads into `out_dir/<domain>-showcase/` and zip it beside it."""
    if config.explore is None:
        raise ValueError(f"{config.name} has no explorer to export")
    showcase, at = config.explore.showcase, at or datetime.now(UTC)
    staging = out_dir / f"{config.name}-showcase"
    shutil.rmtree(staging, ignore_errors=True)
    tables, withheld, filtered = [], [], []
    con = connect(data_dir)
    for name in sorted(published_tables(config.explore, data_dir)):
        layer, table = name.split(".")
        partition = latest_partition(data_dir / layer / table)
        if partition is None:
            logger.warning("%s is not built: the snapshot goes without it", name)
            continue
        if name in showcase.withheld:
            logger.info("%s withheld: %s", name, showcase.withheld[name])
            withheld.append(name)
            continue
        if name in showcase.rows:
            condition = showcase.rows[name]
            rows = con.execute(f"SELECT * FROM {name} WHERE {condition}").pl()
            write_table(rows, staging / layer / table, {table: partition.name}, at)
            filtered.append(name)
        else:
            shutil.copytree(partition, staging / layer / table / partition.name)
        tables.append(name)
    figures = latest_partition(data_dir / "analysis" / FIGURES)
    if figures is not None:
        shutil.copytree(figures, staging / "analysis" / FIGURES / figures.name)
    (staging / SNAPSHOT_FILE).write_text(
        json.dumps(
            {
                "domain": config.name,
                "exported_at": at.isoformat(),
                "tables": tables,
                "withheld": showcase.withheld,
                "filtered": showcase.rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = out_dir / f"{config.name}-showcase-{at:%Y%m%d}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(staging).as_posix())
    shutil.rmtree(staging)
    return Snapshot(archive, tables, withheld, filtered)


def unpack_snapshot(source: str, data_dir: Path) -> dict[str, object]:
    """Lay a snapshot - a URL or a path - out as `data_dir`, once: an unpacked one is read
    again as it is. Returns what the snapshot says about itself."""
    marker = data_dir / SNAPSHOT_FILE
    if not marker.is_file():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir.parent / f"{data_dir.name}-showcase.zip"
        if re.match(r"https?://", source):
            with urllib.request.urlopen(
                source, timeout=120
            ) as response:  # http(s) only, matched above
                archive.write_bytes(response.read())
        else:
            archive = Path(source)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(data_dir)  # zipfile drops absolute paths and `..`
    described: dict[str, object] = json.loads(marker.read_text(encoding="utf-8"))
    return described
