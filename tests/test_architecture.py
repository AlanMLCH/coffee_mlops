"""The pipelines stay decoupled: they exchange Parquet on disk, never imports.

Without this test the boundary erodes one convenient import at a time, and with it
goes the ability to run, deploy and install either pipeline on its own.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "coffee_mlops"
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
        ("ml", "coffee_mlops.data"),
        ("data", "coffee_mlops.ml"),
        ("serving", "coffee_mlops.data"),
    ],
)
def test_packages_do_not_reach_across_the_boundary(package: str, forbidden: str) -> None:
    assert offenders(list((SRC / package).rglob("*.py")), forbidden) == []


@pytest.mark.parametrize("forbidden", ["coffee_mlops.data", "coffee_mlops.ml"])
def test_shared_modules_do_not_depend_on_a_pipeline(forbidden: str) -> None:
    # The CLI is the one place allowed to know about both.
    assert offenders([SRC / name for name in SHARED], forbidden) == []
