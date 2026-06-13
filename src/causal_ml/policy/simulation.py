"""Policy rollout simulation and regret computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_ml.data.bandit_dgp import policy_value


@dataclass
class SimulationResult:
    policy_value: float
    oracle_value: float
    regret: float
    n_samples: int


def simulate_policy(
    actions: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    oracle_value: float,
) -> SimulationResult:
    """Roll out a policy on fresh counterfactual outcomes."""
    value = policy_value(actions, y0, y1)
    regret = oracle_value - value
    return SimulationResult(
        policy_value=value,
        oracle_value=oracle_value,
        regret=float(regret),
        n_samples=len(actions),
    )


def simulate_from_tau(
    tau_hat: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    oracle_value: float,
    cost: float = 0.0,
) -> SimulationResult:
    """Simulate threshold policy derived from CATE estimates."""
    from causal_ml.policy.learning import threshold_policy

    actions = threshold_policy(tau_hat, cost=cost)
    return simulate_policy(actions, y0, y1, oracle_value)
