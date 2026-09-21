from pathlib import Path

from coffee_mlops.catalog import connect
from coffee_mlops.config import DomainConfig
from coffee_mlops.data.clean import build_clean
from coffee_mlops.ml.features import build_features


def test_every_built_table_is_queryable_by_layer(
    coffee_config: DomainConfig, raw_dir: Path
) -> None:
    data_dir = raw_dir.parent
    build_clean(coffee_config, data_dir)
    build_features(coffee_config, data_dir)

    con = connect(data_dir)

    tables = con.sql(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables ORDER BY 1"
    ).fetchall()
    assert [t[0] for t in tables] == [
        "clean.boroughs",
        "clean.coffee_reviews",
        "clean.coffee_shops",
        "clean.market_context",
        "features.review_features",
    ]
    assert con.sql("SELECT count(*) FROM features.review_features").fetchone() == (25,)


def test_partition_folder_is_not_exposed_as_a_column(
    coffee_config: DomainConfig, raw_dir: Path
) -> None:
    build_clean(coffee_config, raw_dir.parent)

    columns = [
        row[0] for row in connect(raw_dir.parent).sql("DESCRIBE clean.coffee_reviews").fetchall()
    ]

    assert "built_at" not in columns


def test_empty_data_dir_gives_an_empty_catalog(tmp_path: Path) -> None:
    con = connect(tmp_path)

    assert con.sql("SELECT count(*) FROM information_schema.tables").fetchone() == (0,)


def test_table_with_only_an_interrupted_build_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "features" / "review_features" / "built_at=20260919T120000Z").mkdir(parents=True)

    con = connect(tmp_path)

    assert con.sql("SELECT count(*) FROM information_schema.tables").fetchone() == (0,)
