# Copyright (c) Microsoft. All rights reserved.

"""The file sample must stay within one hosted session's upload directory."""

from __future__ import annotations

import os
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"),
    reason="The hosted file sample requires POSIX no-follow directory opens.",
)

_sample = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="sample_file_tools")
read_file = cast(Callable[[str], str], _sample["read_file"])
list_files = cast(Callable[[], list[str]], _sample["list_files"])


@pytest.fixture
def files_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "sample_files"
    root.mkdir()
    return root


def test_file_tools_only_access_regular_files_in_session_home(files_root: Path) -> None:
    (files_root / "report.txt").write_text("Safe report", encoding="utf-8")
    outside = files_root.parent / "private.txt"
    outside.write_text("Not for the agent", encoding="utf-8")
    (files_root / "shortcut.txt").symlink_to(outside)

    assert list_files() == ["report.txt"]
    assert read_file("report.txt") == "Safe report"
    for filename in ("../private.txt", str(outside), "shortcut.txt"):
        with pytest.raises(ValueError):
            read_file(filename)


def test_file_tools_refuse_a_symlinked_session_directory(files_root: Path) -> None:
    outside = files_root.parent / "private"
    outside.mkdir()
    (outside / "secret.txt").write_text("Not for the agent", encoding="utf-8")
    files_root.rmdir()
    files_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        list_files()
    with pytest.raises(OSError):
        read_file("secret.txt")


def test_file_reader_enforces_byte_limit(files_root: Path) -> None:
    (files_root / "large.txt").write_bytes(b"x" * 1_000_001)
    with pytest.raises(ValueError, match="at most 1 MB"):
        read_file("large.txt")


def test_file_reader_fails_closed_without_no_follow(files_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (files_root / "report.txt").write_text("Safe report", encoding="utf-8")
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(RuntimeError, match="no-follow"):
        read_file("report.txt")
