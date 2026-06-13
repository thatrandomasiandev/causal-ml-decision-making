"""Reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator."""
    return np.random.default_rng(seed)


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic short hash for a config dict."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
