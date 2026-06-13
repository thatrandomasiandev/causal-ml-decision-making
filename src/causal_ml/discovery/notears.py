"""NOTEARS: continuous optimization for DAG structure learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class NOTEARSResult:
    adjacency: np.ndarray
    loss: float
    success: bool


def _adj_to_W(adj: np.ndarray) -> np.ndarray:
    return adj.copy()


def _h(W: np.ndarray) -> float:
    """Acyclicity constraint h(W) = tr(exp(W * W)) - d."""
    d = W.shape[0]
    M = W * W
    return float(np.trace(np.linalg.matrix_power(M + np.eye(d) * 1e-8, d)) - d)


def notears(
    X: np.ndarray,
    lambda1: float = 0.1,
    max_iter: int = 100,
    threshold: float = 0.1,
) -> NOTEARSResult:
    """
    Linear NOTEARS via augmented Lagrangian.

    Minimize 0.5/n * ||X - XW||^2 + lambda1 * ||W||_1 subject to h(W) = 0.
    """
    n, d = X.shape
    X = X - X.mean(axis=0)
    W_init = np.zeros((d, d))

    def loss(W_flat):
        W = W_flat.reshape(d, d)
        residual = X - X @ W
        return 0.5 / n * np.sum(residual**2) + lambda1 * np.sum(np.abs(W))

    def acyclicity(W_flat):
        W = W_flat.reshape(d, d)
        return _h(W)

    rho = 1.0
    alpha = 0.0
    W = W_init.copy()

    for _ in range(max_iter):
        w_flat = W.flatten()

        def augmented(w_flat):
            return loss(w_flat) + 0.5 * rho * acyclicity(w_flat) ** 2 + alpha * acyclicity(w_flat)

        result = minimize(augmented, w_flat, method="L-BFGS-B")
        W = result.x.reshape(d, d)
        np.fill_diagonal(W, 0)
        h_val = acyclicity(W.flatten())
        if h_val < 1e-8:
            break
        alpha += rho * h_val
        rho = min(rho * 10, 1e6)

    adj = W.copy()
    adj[np.abs(adj) < threshold] = 0

    return NOTEARSResult(
        adjacency=adj,
        loss=float(loss(W.flatten())),
        success=h_val < 1e-4,
    )
