"""
paths.py -- repository-relative data paths.

All runtime data lives in ``<repo>/data`` regardless of the current
working directory, so the CLI, the web app and the scripts work when
launched from anywhere.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def data_path(name: str) -> str:
    """Absolute path to a file inside the repository's ``data/`` folder."""
    return str(DATA_DIR / name)
