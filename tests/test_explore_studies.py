"""What the explorer's Findings and Models tabs read: the analysis layer as it is on disk."""

import json
from pathlib import Path

from mlops_core.explore.studies import domain_figures, domain_studies, figure, lineage
from mlops_core.storage import MANIFEST_NAME


def partition(table_dir: Path, stamp: str, files: tuple[str, ...], inputs: dict[str, str]) -> Path:
    """A partition as the pipeline leaves it: its files, then the manifest that completes it."""
    path = table_dir / f"built_at={stamp}"
    path.mkdir(parents=True)
    for name in files:
        (path / name).write_bytes(b"")
    manifest = {"table": table_dir.name, "rows": 1, "built_at": stamp, "inputs": inputs}
    (path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_the_domains_studies_and_figures_are_what_the_core_did_not_compute(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    for name in ("review_residuals", "market_summary", "shop_kinds"):
        partition(analysis / name, "20260929T000000Z", (f"{name}.parquet",), {})
    (analysis / "half_written" / "built_at=20260929T000000Z").mkdir(parents=True)
    drawn = ("review_residual_bias.png", "market_share.png", "shop_kinds.png")
    partition(analysis / "figures", "20260929T000000Z", drawn, {})

    assert domain_studies(tmp_path, ["review"]) == ["market_summary", "shop_kinds"]
    assert [p.name for p in domain_figures(tmp_path, ["review"])] == [
        "market_share.png",
        "shop_kinds.png",
    ]
    assert figure(tmp_path, "review_residual_bias") is not None
    assert figure(tmp_path, "never_drawn") is None


def test_a_models_studies_are_stamped_with_what_they_read(tmp_path: Path) -> None:
    inputs = {"review_features": "built_at=20260928T000000Z"}
    partition(tmp_path / "analysis" / "review_target_distribution", "20260929T000000Z", (), inputs)

    assert lineage(tmp_path, "review") == {"built_at": "20260929T000000Z", **inputs}
    assert lineage(tmp_path, "offer") == {}


def test_nothing_built_is_nothing_to_show(tmp_path: Path) -> None:
    # A drawing without its manifest is unfinished: it is not the newest one.
    (tmp_path / "analysis" / "figures" / "built_at=20260929T000000Z").mkdir(parents=True)

    assert domain_studies(tmp_path / "elsewhere", ["review"]) == []
    assert domain_figures(tmp_path, ["review"]) == []
    assert figure(tmp_path, "market_share") is None
