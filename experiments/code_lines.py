"""How many lines of code the core and each domain take: the framework's cost, measured.

The criterion the framework is judged by is what a second domain costs: an adapter and a
YAML, never an edit to the core. This counts code lines - no blank lines, comments or
docstrings - per package, the same way every time, so the second domain's number can be
set beside the first's. Run with:

    uv run python experiments/code_lines.py
"""

import ast
import io
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
# A domain's glue: what it writes only because the core asks, as opposed to what it knows.
GLUE = ("adapter.py", "__init__.py", "request.py", "features.py")
LAYOUT = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}


def docstring_lines(tree: ast.Module) -> set[int]:
    """The lines of every module, class and function docstring."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def code_lines(path: Path) -> int:
    """Lines holding at least one token of code."""
    source = path.read_text(encoding="utf-8")
    skipped = docstring_lines(ast.parse(source))
    lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in LAYOUT or token.type == tokenize.ENDMARKER:
            continue
        lines.update(n for n in range(token.start[0], token.end[0] + 1) if n not in skipped)
    return len(lines)


def main() -> None:
    packages = {"mlops_core": sorted((SRC / "mlops_core").rglob("*.py"))}
    for domain in sorted(p for p in (SRC / "domains").iterdir() if (p / "__init__.py").is_file()):
        packages[f"domains/{domain.name}"] = sorted(domain.rglob("*.py"))
        packages[f"  its glue ({', '.join(GLUE)})"] = [domain / name for name in GLUE]
    for name, files in packages.items():
        present = [f for f in files if f.is_file()]
        print(f"{name}: {len(present)} files, {sum(map(code_lines, present)):,} code lines")


if __name__ == "__main__":
    main()
