import re
import subprocess
from pathlib import Path

import pytest

from mlops_core.provenance import CodeVersion, code_version


def test_a_checkout_reports_its_revision() -> None:
    version = code_version()

    assert version is not None
    assert re.fullmatch(r"[0-9a-f]{40}", version.commit)


def test_outside_a_checkout_provenance_is_absent_not_fatal(tmp_path: Path) -> None:
    """Inside the API image there is no git history; the pipeline must still run."""
    assert code_version(tmp_path) is None


def test_tags_say_whether_the_tree_was_dirty() -> None:
    tags = CodeVersion(commit="abc123", dirty=True).as_tags()

    assert tags == {"git_commit": "abc123", "git_dirty": "True"}


def test_a_machine_without_git_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", no_git)

    assert code_version() is None
