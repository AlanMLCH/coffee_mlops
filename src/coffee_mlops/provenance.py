"""Which code produced a result.

MLflow records the entry point but not the revision when the pipeline runs from a CLI,
so without this a run cannot be traced back to the code that made it. `dirty` matters as
much as the hash: a run made from uncommitted changes is not reproducible, and saying so
is better than implying otherwise.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CodeVersion:
    commit: str
    dirty: bool

    def as_tags(self) -> dict[str, str]:
        return {"git_commit": self.commit, "git_dirty": str(self.dirty)}


def _git(repo: Path, *args: str) -> str | None:
    """Run a git command, or return None where git cannot answer (no git, no checkout,
    a container built from a tarball). Provenance is nice to have, never a hard failure."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version(repo: Path = REPO_ROOT) -> CodeVersion | None:
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None
    return CodeVersion(commit=commit, dirty=bool(_git(repo, "status", "--porcelain")))
