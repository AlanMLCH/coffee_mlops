"""Clean layer: the domain turns validated raw frames into its tables; the core holds
each one to its contract and writes it with lineage.

The split is the contract's: *how* two scrapes are harmonised or a register is placed
on a map is domain knowledge, while "every clean table meets a strict contract before
anyone can read it, and records which raw partitions it came from" holds for any domain.
So does cutting a corpus into chunks: a domain that lists documents gets the corpus'
two tables from the core, filed under the domain's own topics.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from mlops_core.adapter import DomainAdapter
from mlops_core.contracts import check_contract
from mlops_core.data.corpus import corpus_contracts, corpus_tables
from mlops_core.data.validate import validate_raw
from mlops_core.storage import write_table

logger = logging.getLogger(__name__)


def build_clean(
    adapter: DomainAdapter, data_dir: Path, at: datetime | None = None
) -> dict[str, Path]:
    """Validate the latest raw data, clean it, check the clean contracts, write Parquet."""
    sources = validate_raw(adapter, data_dir / "raw")
    frames = {name: source.frame for name, source in sources.items()}
    lineage = {name: source.lineage for name, source in sources.items()}

    read_at = {name: source.artifact.manifest.ingested_at for name, source in sources.items()}
    tables = dict(adapter.clean(frames, read_at))
    contracts = dict(adapter.clean_contracts())
    if tables.keys() != contracts.keys():
        # A table without a contract is a promise nobody made; a contract without a
        # table is a promise nobody keeps. Either way the domain is inconsistent.
        raise ValueError(
            f"Clean tables {sorted(tables)} do not match their contracts {sorted(contracts)}"
        )

    config = adapter.config
    if config.corpus is not None and config.documents:
        # The corpus is cut the same way for every domain, so the core builds its tables.
        taken = sorted(set(config.corpus_tables) & tables.keys())
        if taken:
            raise ValueError(f"{taken} are the corpus' tables; a domain cannot build its own")
        tables |= corpus_tables(config.documents, config.corpus, frames, read_at)
        contracts |= corpus_contracts(config.corpus)
    # Every table is checked before any is written: a half-written layer would pair a
    # new table with a stale one, and readers always take the newest of each.
    checked = {name: check_contract(contracts[name], table.frame) for name, table in tables.items()}

    built_at = at or datetime.now(UTC)
    return {
        name: write_table(
            checked[name],
            data_dir / "clean" / name,
            {source: lineage[source] for source in table.inputs if source in lineage},
            built_at,
        )
        for name, table in tables.items()
    }
