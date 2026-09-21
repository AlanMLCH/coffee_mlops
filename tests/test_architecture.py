"""The pipelines stay decoupled: they exchange Parquet on disk, never imports.

Without this test the boundary erodes one convenient import at a time, and with it
goes the ability to run, deploy and install either pipeline on its own.
"""

import ast
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mlops_core"
SHARED = ["config.py", "contracts.py", "storage.py", "catalog.py"]


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def offenders(paths: list[Path], forbidden: str) -> list[str]:
    return [
        f"{path.relative_to(SRC).as_posix()} imports {module}"
        for path in paths
        for module in imported_modules(path)
        if module.startswith(forbidden)
    ]


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("ml", "mlops_core.data"),
        ("data", "mlops_core.ml"),
        ("serving", "mlops_core.data"),
        ("data", "mlops_core.analysis"),
        ("ml", "mlops_core.analysis"),
        ("ml", "mlops_core.serving"),
    ],
)
def test_packages_do_not_reach_across_the_boundary(package: str, forbidden: str) -> None:
    assert offenders(list((SRC / package).rglob("*.py")), forbidden) == []


@pytest.mark.parametrize("forbidden", ["mlops_core.data", "mlops_core.ml"])
def test_shared_modules_do_not_depend_on_a_pipeline(forbidden: str) -> None:
    # The CLI is the one place allowed to know about both.
    assert offenders([SRC / name for name in SHARED], forbidden) == []


def test_every_source_file_is_actually_in_the_repository() -> None:
    """A too-broad ignore rule (`data/` also matches src/mlops_core/data/) kept four
    modules out of the repository: everything worked locally and CI failed on an import.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "src"],
        cwd=SRC.parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    committed = {Path(path).resolve() for path in (SRC.parents[1] / p for p in tracked)}

    on_disk = {path.resolve() for path in SRC.rglob("*.py")}

    assert on_disk - committed == set()
