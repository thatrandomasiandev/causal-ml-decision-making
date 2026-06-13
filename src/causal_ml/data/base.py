"""Dataset protocol and metadata for causal ML benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CausalDataset:
    """Container for observational data with optional ground-truth accessors."""

    X: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1


@dataclass
class UpliftDataset(CausalDataset):
    """Observational treatment-outcome dataset."""

    T: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if self.T.size == 0 or self.Y.size == 0:
            raise ValueError("UpliftDataset requires T and Y arrays.")


@dataclass
class BanditDataset(CausalDataset):
    """Logged bandit interaction dataset."""

    actions: np.ndarray = field(default_factory=lambda: np.array([]))
    rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    propensities: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if self.actions.size == 0 or self.rewards.size == 0 or self.propensities.size == 0:
            raise ValueError("BanditDataset requires actions, rewards, and propensities.")


@dataclass
class DAGDataset(CausalDataset):
    """Static or time-series data generated from a known DAG."""

    adjacency: np.ndarray = field(default_factory=lambda: np.array([]))
    is_time_series: bool = False

    def __post_init__(self) -> None:
        if self.adjacency.size == 0:
            raise ValueError("DAGDataset requires an adjacency matrix.")
