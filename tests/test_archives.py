"""Archives a source comes as: a ZIP read by Python, a RAR read by bsdtar. No fixture can be
written as a RAR (only RAR's own program writes one), so the RAR path is tested against a
stood-in bsdtar: the commands it is given, and what it says when it cannot."""

import io
import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mlops_core.data import archives


@pytest.fixture(autouse=True)
def forget_bsdtar() -> Iterator[None]:
    found = archives.bsdtar  # a test may stand another in; the cache is this one's
    found.cache_clear()
    yield
    found.cache_clear()


def test_a_zip_is_read_by_python_whatever_it_is_called(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("QQP/01-2025_01.csv", "a,b\n1,2\n")
    called_rar = tmp_path / "QQP_2025.rar"  # the name says RAR; the bytes say ZIP
    called_rar.write_bytes(buffer.getvalue())

    assert not archives.is_rar(called_rar)
    assert archives.names(called_rar) == ["QQP/01-2025_01.csv"]
    assert archives.read(called_rar, "QQP/01-2025_01.csv") == b"a,b\n1,2\n"


def test_a_rar_is_read_by_bsdtar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rar = tmp_path / "QQP_2025.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x01\x00 and then the archive")
    ran: list[list[str]] = []

    def bsdtar(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        ran.append(command)
        if command[1] == "-tf":
            return subprocess.CompletedProcess(command, 0, b"QQP_2025/\nQQP_2025/01.csv\n", b"")
        return subprocess.CompletedProcess(command, 0, b"producto,precio\n", b"")

    monkeypatch.setattr(archives, "bsdtar", lambda: "bsdtar")
    monkeypatch.setattr(subprocess, "run", bsdtar)

    assert archives.is_rar(rar)
    assert archives.names(rar) == ["QQP_2025/", "QQP_2025/01.csv"]
    assert archives.read(rar, "QQP_2025/01.csv") == b"producto,precio\n"
    assert ran == [["bsdtar", "-tf", str(rar)], ["bsdtar", "-xOf", str(rar), "QQP_2025/01.csv"]]


def test_a_rar_bsdtar_cannot_read_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rar = tmp_path / "broken.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x00 truncated")

    def fails(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command, b"", b"x\nTruncated RAR file data\n")

    monkeypatch.setattr(archives, "bsdtar", lambda: "bsdtar")
    monkeypatch.setattr(subprocess, "run", fails)

    with pytest.raises(ValueError, match=r"broken.rar: bsdtar could not read it \(Truncated"):
        archives.names(rar)

    def silent(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command, b"", b"")

    monkeypatch.setattr(subprocess, "run", silent)
    with pytest.raises(ValueError, match="no reason given"):
        archives.read(rar, "x.csv")


def which_in(found: dict[str, str]) -> Any:
    """`shutil.which`, over one folder holding what `found` names."""
    return lambda name, path=None: found.get(name)


def answering(versions: dict[str, bytes]) -> Any:
    return lambda command, **o: subprocess.CompletedProcess(command, 0, versions[command[0]], b"")


@pytest.mark.parametrize(
    ("found", "versions", "chosen"),
    [
        ({"bsdtar": "/usr/bin/bsdtar"}, {"/usr/bin/bsdtar": b"bsdtar 3.7.2"}, "/usr/bin/bsdtar"),
        ({"tar": "/usr/bin/tar"}, {"/usr/bin/tar": b"bsdtar 3.8.4 - libarchive 3.8.4"},
         "/usr/bin/tar"),
    ],
)  # fmt: skip
def test_bsdtar_is_found_as_itself_or_as_a_tar_that_is_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    found: dict[str, str],
    versions: dict[str, bytes],
    chosen: str,
) -> None:
    monkeypatch.setenv("PATH", "one-folder")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))  # no Windows here
    monkeypatch.setattr(archives.shutil, "which", which_in(found))
    monkeypatch.setattr(subprocess, "run", answering(versions))

    assert archives.bsdtar() == chosen


def test_a_gnu_tar_first_on_the_path_is_passed_over_for_windows_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git for Windows puts its GNU tar before Windows' bsdtar."""
    system = tmp_path / "System32" / "tar.exe"
    system.parent.mkdir()
    system.write_bytes(b"")
    monkeypatch.setenv("PATH", "git-usr-bin")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(archives.shutil, "which", which_in({"tar": "git-tar"}))
    monkeypatch.setattr(
        subprocess, "run", answering({"git-tar": b"tar (GNU tar) 1.35", str(system): b"bsdtar 3.8"})
    )

    assert archives.bsdtar() == str(system)


def test_gnu_tar_cannot_read_a_rar_and_says_what_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "one-folder" + archives.os.pathsep)  # an empty entry too
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(archives.shutil, "which", which_in({"tar": "/usr/bin/tar"}))
    monkeypatch.setattr(subprocess, "run", answering({"/usr/bin/tar": b"tar (GNU tar) 1.35"}))

    with pytest.raises(FileNotFoundError, match="install libarchive-tools"):
        archives.bsdtar()
