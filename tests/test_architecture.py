"""The boundaries hold because tests hold them, not because they are documented.

Two kinds. The pipelines stay decoupled: they exchange Parquet on disk, never imports,
so each can be installed and deployed on its own. And the core stays generic: it never
imports a domain and never even names one, which is the whole claim of the framework -
a new domain is a new package under `domains/` and nothing in `mlops_core` changes.
Without these tests, both boundaries erode one convenient import at a time.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = SRC / "mlops_core"
DOMAINS = SRC / "domains"
SHARED = ["config.py", "contracts.py", "storage.py", "catalog.py", "adapter.py", "stats.py"]
# Words that belong to a domain. The core naming any of them - in code, a comment or a
# docstring - is how a domain's assumptions start leaking into what should be generic.
DOMAIN_WORDS = re.compile(
    r"coffee|caf[eé]|\bcqi\b|\bpsd\b|denue|overpass|inegi|borough|alcald|cup.?points|"
    r"grading|arabica|videogame|\bsteam\b|\brawg\b",
    re.IGNORECASE,
)


def imported_modules(path: Path) -> set[str]:
    """What a module imports at runtime. `if TYPE_CHECKING:` blocks are skipped: an import
    only a type checker ever executes couples nothing at install or run time."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
                visit(node.orelse)
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    modules.update(alias.name for alias in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module:
                    modules.add(child.module)

    visit(tree.body)
    return modules


def offenders(paths: list[Path], forbidden: str) -> list[str]:
    return [
        f"{path.relative_to(SRC).as_posix()} imports {module}"
        for path in paths
        for module in imported_modules(path)
        if module == forbidden or module.startswith(f"{forbidden}.")
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
        # Retrieval reads the clean layer from disk, as the model pipeline does.
        ("rag", "mlops_core.data"),
        ("rag", "mlops_core.ml"),
        ("data", "mlops_core.rag"),
        ("ml", "mlops_core.rag"),
        ("serving", "mlops_core.rag"),
    ],
)
def test_packages_do_not_reach_across_the_boundary(package: str, forbidden: str) -> None:
    assert offenders(list((CORE / package).rglob("*.py")), forbidden) == []


@pytest.mark.parametrize("forbidden", ["mlops_core.data", "mlops_core.ml"])
def test_shared_modules_do_not_depend_on_a_pipeline(forbidden: str) -> None:
    # The CLI and the orchestrator are the places allowed to know about both.
    assert offenders([CORE / name for name in SHARED], forbidden) == []


def test_the_core_never_imports_a_domain() -> None:
    """Domains are found by name at runtime (`load_adapter`); a static import of one
    would make the core depend on the very thing it must stay free of."""
    assert offenders(list(CORE.rglob("*.py")), "domains") == []


def test_the_core_never_names_a_domain() -> None:
    mentions = [
        f"{path.relative_to(SRC).as_posix()}:{number}: {line.strip()}"
        for path in sorted(CORE.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if DOMAIN_WORDS.search(line)
    ]

    assert mentions == []


def test_every_domain_exposes_an_adapter() -> None:
    """The one thing the core asks a domain package for."""
    for package in sorted(p for p in DOMAINS.iterdir() if (p / "__init__.py").is_file()):
        tree = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert "adapter" in functions, f"domains/{package.name} has no adapter()"


def test_every_domain_is_named_after_its_package() -> None:
    """The core finds a domain's files (its config, its question set) by the name in its
    config; a config named otherwise would send it to another package's directory."""
    from mlops_core.adapter import available_domains, domain_dir, load_adapter

    for domain in available_domains():
        assert load_adapter(domain).config.name == domain
        assert (domain_dir(domain) / "config.yaml").is_file()


def test_every_source_file_is_actually_in_the_repository() -> None:
    """A too-broad ignore rule (`data/` also matched src/mlops_core/data/) kept four
    modules out of the repository: everything worked locally and CI failed on an import.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "src"],
        cwd=SRC.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    committed = {(SRC.parent / path).resolve() for path in tracked}

    on_disk = {path.resolve() for path in SRC.rglob("*.py")}

    assert on_disk - committed == set()
