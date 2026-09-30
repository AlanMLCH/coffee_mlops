"""Archives a source is downloaded as: ZIP, which Python reads, and RAR, which it does not.

A publisher picks the format; a raw download is kept as it came, so the reading has to
follow. Python's standard library has no RAR reader, and the packages that offer one call
an outside program anyway. libarchive's `bsdtar` reads RAR 5 and ships with Windows 10
and later (`tar.exe`) and with macOS (`tar`); on Linux it is the `libarchive-tools`
package. Verified 2026-09-29 with Windows' bsdtar 3.8.4 on PROFECO's 2025 archive (RAR 5,
85 MB, 24 files): a 173 MB member streams out in under half a second.

The format is told by the file's first bytes, not its name: a download is what it is.
"""

import os
import shutil
import subprocess
import zipfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path

RAR_MAGIC = (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")  # RAR 4, RAR 5


def is_rar(path: Path) -> bool:
    with path.open("rb") as file:
        head = file.read(8)
    return any(head.startswith(magic) for magic in RAR_MAGIC)


def names(path: Path) -> list[str]:
    """Every file's name in the archive, as the archive spells it."""
    if is_rar(path):
        listed = _bsdtar_run("-tf", path)
        return [line for line in listed.decode("utf-8").splitlines() if line]
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def read(path: Path, name: str) -> bytes:
    """One file of the archive, whole."""
    if is_rar(path):
        return _bsdtar_run("-xOf", path, name)
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def _bsdtar_run(flag: str, path: Path, *members: str) -> bytes:
    try:
        done = subprocess.run(
            [bsdtar(), flag, str(path), *members], capture_output=True, check=True
        )
    except subprocess.CalledProcessError as failed:
        said = failed.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = said[-1] if said else "no reason given"
        raise ValueError(f"{path.name}: bsdtar could not read it ({reason})") from failed
    return done.stdout


@cache
def bsdtar() -> str:
    """The program that reads a RAR: `bsdtar`, or a `tar` that is bsdtar (Windows, macOS).
    GNU tar cannot, and it can come first on the PATH - Git for Windows puts its own
    there - so every `tar` found is asked what it is, and Windows' own is tried last."""
    for found in dict.fromkeys(_candidates()):
        version = subprocess.run([found, "--version"], capture_output=True, check=False)
        if version.stdout.startswith(b"bsdtar"):
            return found
    raise FileNotFoundError(
        "Reading a RAR archive needs libarchive's bsdtar: Windows 10+ and macOS have it as "
        "`tar`; on Linux, install libarchive-tools"
    )


def _candidates() -> Iterator[str]:
    """Every `bsdtar` and `tar` on the PATH, folder by folder, then Windows' own."""
    for name in ("bsdtar", "tar"):
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            found = shutil.which(name, path=folder) if folder else None
            if found is not None:
                yield found
    system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "tar.exe"
    if system.is_file():
        yield str(system)
