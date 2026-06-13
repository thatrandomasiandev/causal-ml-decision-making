"""Synthetic DAG and time-series DGPs with known structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_ml.data.base import DAGDataset
from causal_ml.utils.seed import set_seed


@dataclass
class DAGDGPConfig:
    n_samples: int = 5000
    n_nodes: int = 5
    edge_prob: float = 0.3
    noise_std: float = 0.5
    seed: int = 42
    time_series: bool = False
    n_timesteps: int = 500
    max_lag: int = 2


def _random_dag(n_nodes: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a random DAG via upper-triangular adjacency (topological order fixed)."""
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = rng.uniform(0.3, 0.9) * rng.choice([-1.0, 1.0])
    return adj


def generate_dag_data(config: DAGDGPConfig | None = None) -> DAGDataset:
    """
    Generate data from a linear SEM: X = X @ B^T + noise.

    For time series, builds lagged adjacency of shape (d, d, max_lag+1).
    """
    cfg = config or DAGDGPConfig()
    rng = set_seed(cfg.seed)

    if cfg.time_series:
        return _generate_time_series(cfg, rng)

    adj = _random_dag(cfg.n_nodes, cfg.edge_prob, rng)
    X = rng.standard_normal((cfg.n_samples, cfg.n_nodes))
    for _ in range(5):
        X = X @ adj.T + rng.normal(0, cfg.noise_std, size=X.shape)

    return DAGDataset(
        X=X,
        adjacency=adj,
        is_time_series=False,
        metadata={
            "dgp": "dag_static",
            "n_samples": cfg.n_samples,
            "n_nodes": cfg.n_nodes,
            "edge_prob": cfg.edge_prob,
            "seed": cfg.seed,
        },
        ground_truth={"adjacency": adj, "edge_list": _adj_to_edges(adj)},
    )


def _generate_time_series(cfg: DAGDGPConfig, rng: np.random.Generator) -> DAGDataset:
    """Generate multivariate time series with known lagged causal structure."""
    d = cfg.n_nodes
    max_lag = cfg.max_lag
    T = cfg.n_timesteps

    # Lagged adjacency: lag 0 = contemporaneous (upper triangular), lag>=1 = lagged edges
    adj_lagged = np.zeros((d, d, max_lag + 1))
    adj_lagged[:, :, 0] = _random_dag(d, cfg.edge_prob * 0.5, rng)

    for lag in range(1, max_lag + 1):
        for i in range(d):
            for j in range(d):
                if i != j and rng.random() < cfg.edge_prob * 0.4:
                    adj_lagged[i, j, lag] = rng.uniform(0.2, 0.6) * rng.choice([-1.0, 1.0])

    series = np.zeros((T, d))
    burn_in = max_lag + 10
    total = T + burn_in
    full = np.zeros((total, d))

    for t in range(total):
        noise = rng.normal(0, cfg.noise_std, size=d)
        val = noise.copy()
        for lag in range(max_lag + 1):
            if t - lag < 0:
                continue
            val += full[t - lag] @ adj_lagged[:, :, lag].T
        full[t] = val

    series = full[burn_in:]

    return DAGDataset(
        X=series,
        adjacency=adj_lagged,
        is_time_series=True,
        metadata={
            "dgp": "dag_time_series",
            "n_timesteps": T,
            "n_nodes": d,
            "max_lag": max_lag,
            "seed": cfg.seed,
        },
        ground_truth={"adjacency_lagged": adj_lagged, "adjacency": adj_lagged[:, :, 0]},
    )


def _adj_to_edges(adj: np.ndarray, threshold: float = 1e-8) -> list[tuple[int, int, float]]:
    edges = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if abs(adj[i, j]) > threshold:
                edges.append((i, j, float(adj[i, j])))
    return edges
