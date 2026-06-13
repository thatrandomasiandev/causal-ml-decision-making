"""Policy learning from CATE estimates."""

from __future__ import annotations

import numpy as np


def threshold_policy(tau_hat: np.ndarray, cost: float = 0.0) -> np.ndarray:
    """Treat when predicted CATE exceeds cost threshold."""
    return (tau_hat > cost).astype(float)


def cost_sensitive_policy(tau_hat: np.ndarray, treatment_cost: float = 0.0) -> np.ndarray:
    """Alias for threshold policy with explicit treatment cost."""
    return threshold_policy(tau_hat, cost=treatment_cost)
