"""NOTEARS: continuous optimization for DAG structure learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class NOTEARSResult:
    """Container for NOTEARS structure learning results.

    Attributes:
        adjacency: Estimated weighted adjacency matrix of shape (d, d).
        loss: Final objective value (data fidelity + sparsity penalty).
        success: Whether the acyclicity constraint was satisfied.
    """

    adjacency: np.ndarray
    loss: float
    success: bool


def _adj_to_W(adj: np.ndarray) -> np.ndarray:
    return adj.copy()


def _h(W: np.ndarray) -> float:
    """Acyclicity constraint h(W) = tr(exp(W ⊙ W)) - d.

    Args:
        W: Weight matrix of shape (d, d).

    Returns:
        Scalar acyclicity violation; zero iff the graph is a DAG.
    """
    d = W.shape[0]
    M = W * W
    return float(np.trace(np.linalg.matrix_power(M + np.eye(d) * 1e-8, d)) - d)


def _compute_gradient_h(W: np.ndarray) -> np.ndarray:
    """Analytic gradient of h(W) = tr(e^{W⊙W}) - d via matrix exponential.

    Uses the identity ∇_W h = 2W ⊙ e^{W⊙W}.

    Args:
        W: Weight matrix of shape (d, d).

    Returns:
        Gradient matrix of shape (d, d).
    """
    M = W * W
    E = expm(M)
    return 2.0 * W * E


def notears(
    X: np.ndarray,
    lambda1: float = 0.1,
    max_iter: int = 100,
    threshold: float = 0.1,
) -> NOTEARSResult:
    """Linear NOTEARS via augmented Lagrangian.

    Minimizes 0.5/n ||X - XW||² + λ₁||W||₁ subject to h(W) = 0.

    Args:
        X: Data matrix of shape (n, d).
        lambda1: L1 regularization strength.
        max_iter: Maximum augmented Lagrangian iterations.
        threshold: Entries with |w_ij| below this are zeroed.

    Returns:
        NOTEARSResult with the estimated adjacency, loss, and convergence flag.
    """
    n, d = X.shape
    X = X - X.mean(axis=0)
    W_init = np.zeros((d, d))

    def loss(W_flat: np.ndarray) -> float:
        W = W_flat.reshape(d, d)
        residual = X - X @ W
        return 0.5 / n * np.sum(residual**2) + lambda1 * np.sum(np.abs(W))

    def acyclicity(W_flat: np.ndarray) -> float:
        W = W_flat.reshape(d, d)
        return _h(W)

    rho = 1.0
    alpha = 0.0
    h_val = np.inf
    W = W_init.copy()

    for _ in range(max_iter):
        w_flat = W.flatten()

        def augmented(w_flat: np.ndarray) -> float:
            return (
                loss(w_flat)
                + 0.5 * rho * acyclicity(w_flat) ** 2
                + alpha * acyclicity(w_flat)
            )

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


def notears_mlp(
    X: np.ndarray,
    *,
    hidden_dim: int = 16,
    lambda1: float = 0.01,
    max_iter: int = 100,
    lr: float = 1e-3,
    rho_max: float = 1e16,
    threshold: float = 0.3,
    seed: int = 42,
) -> NOTEARSResult:
    """Nonlinear NOTEARS using a 2-layer MLP for each structural equation.

    Each variable j is modelled as x_j = f_j(x_{\\j}) + ε_j where
    f_j : ℝ^d → ℝ is a 2-layer MLP (nn.Sequential: Linear → ReLU → Linear).
    The acyclicity constraint is identical to the linear version:
    h(W) = tr(e^{W⊙W}) - d, where W_ij = ||w^{(1)}_{j,i}||₂ aggregates
    the first-layer weights connecting variable i to MLP j.

    Args:
        X: Data matrix of shape (n, d).
        hidden_dim: Hidden layer width for each MLP.
        lambda1: L1 penalty on the adjacency weights.
        max_iter: Augmented Lagrangian outer iterations.
        lr: Adam learning rate.
        rho_max: Maximum penalty coefficient.
        threshold: Edge weight threshold for the final adjacency.
        seed: Random seed for reproducibility.

    Returns:
        NOTEARSResult with the estimated adjacency, loss, and convergence flag.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for notears_mlp. "
            "Install with: pip install torch"
        ) from exc

    torch.manual_seed(seed)
    n, d = X.shape
    X_centered = X - X.mean(axis=0)
    X_torch = torch.tensor(X_centered, dtype=torch.float32)

    class _MLPModel(nn.Module):
        """Collection of d independent MLPs, one per variable."""

        def __init__(self, d: int, hidden: int) -> None:
            super().__init__()
            self.d = d
            self.hidden = hidden
            self.fc1 = nn.Linear(d, d * hidden, bias=True)
            self.fc2 = nn.Linear(d * hidden, d, bias=True)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: f(X) of shape (n, d)."""
            h = self.fc1(x)
            h = torch.relu(h)
            return self.fc2(h)

        def adjacency_matrix(self) -> torch.Tensor:
            """Extract W where W_ij = ||first-layer weights from i to MLP j||₂."""
            w1 = self.fc1.weight.view(self.d, self.hidden, self.d)
            return torch.sqrt(torch.sum(w1 ** 2, dim=1)).T

    model = _MLPModel(d, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rho = 1.0
    alpha = 0.0
    h_val = np.inf

    for iteration in range(max_iter):
        for _ in range(200):
            optimizer.zero_grad()
            X_hat = model(X_torch)
            mse_loss = 0.5 / n * torch.sum((X_torch - X_hat) ** 2)

            W = model.adjacency_matrix()
            l1_penalty = lambda1 * torch.sum(torch.abs(W))
            M = W * W
            h = torch.trace(torch.matrix_exp(M)) - d

            objective = mse_loss + l1_penalty + 0.5 * rho * h * h + alpha * h
            objective.backward()
            optimizer.step()

        with torch.no_grad():
            W_np = model.adjacency_matrix().cpu().numpy()
            h_val = float(np.trace(expm(W_np * W_np)) - d)

        logger.debug(
            "notears_mlp iter=%d  h=%.6e  rho=%.2e",
            iteration, h_val, rho,
        )

        if h_val < 1e-8:
            break

        alpha += rho * h_val
        rho = min(rho * 10, rho_max)

    with torch.no_grad():
        W_final = model.adjacency_matrix().cpu().numpy()
        X_hat_np = model(X_torch).cpu().numpy()
        data_loss = 0.5 / n * np.sum((X_centered - X_hat_np) ** 2)

    adj = W_final.copy()
    adj[np.abs(adj) < threshold] = 0
    np.fill_diagonal(adj, 0)

    return NOTEARSResult(
        adjacency=adj,
        loss=float(data_loss + lambda1 * np.sum(np.abs(W_final))),
        success=h_val < 1e-4,
    )
