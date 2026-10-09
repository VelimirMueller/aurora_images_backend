"""Rename guard (S-E0-02): the old brand must not come back.

The natural phenomenon ("aurora borealis", "aurora corona", ...) is real content and stays,
so the pattern matches only brand spellings, never a bare "aurora".
The guard is git-only by design: it checks tracked files and skips outside a checkout.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BRAND = r"aurorae|aurora_images|aurora-uploads|AURORA_[A-Z]"
# The README must name the old env prefixes to say that they are ignored.
ALLOWED = {("README.md", "are ignored")}


def _brand_hits() -> list[tuple[str, str]]:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(  # noqa: S603 - fixed arguments, no user input
        [git, "grep", "-n", "-i", "-E", BRAND, "--", ".", ":!uv.lock", ":!tests/test_naming.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr  # 1 = no match
    hits = []
    for line in result.stdout.splitlines():
        path, _, text = line.split(":", 2)
        hits.append((path, text))
    return hits


def test_old_brand_name_is_gone() -> None:
    unexpected = [
        f"{path}: {text.strip()}"
        for path, text in _brand_hits()
        if not any(path == file and marker in text for file, marker in ALLOWED)
    ]
    assert unexpected == []
