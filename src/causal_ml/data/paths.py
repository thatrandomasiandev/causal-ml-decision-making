"""Project-relative path resolution."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the root of the 01-causal-ml-decision-making project."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Return the local data directory (data/raw/)."""
    return project_root() / "data" / "raw"


def bundled_dir() -> Path:
    """Return bundled fallback datasets shipped with the project."""
    return project_root() / "data" / "bundled"


def ensure_data_dir() -> Path:
    """Create data/raw if missing and return its path."""
    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
