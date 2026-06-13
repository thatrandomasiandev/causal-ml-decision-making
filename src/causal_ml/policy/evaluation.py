"""Off-policy policy value estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyEvalResult:
    value: float
    estimator: str
    std_error: float | None = None


def ips(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
) -> PolicyEvalResult:
    """Inverse Propensity Scoring for deterministic target policies."""
    rho = np.where(actions == target_actions, 1.0 / np.clip(propensities, 1e-6, None), 0.0)
    values = rho * rewards
    se = float(np.std(values) / np.sqrt(len(values)))
    return PolicyEvalResult(value=float(np.mean(values)), estimator="IPS", std_error=se)


def snips(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
) -> PolicyEvalResult:
    """Self-Normalized IPS."""
    rho = np.where(actions == target_actions, 1.0 / np.clip(propensities, 1e-6, None), 0.0)
    numerator = np.sum(rho * rewards)
    denominator = np.sum(rho)
    if denominator < 1e-12:
        return PolicyEvalResult(value=0.0, estimator="SNIPS", std_error=None)
    value = numerator / denominator
    return PolicyEvalResult(value=float(value), estimator="SNIPS", std_error=None)


def dr_policy_value(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
    mu_hat: np.ndarray,
) -> PolicyEvalResult:
    """
    Doubly robust policy value estimator.

    V_DR = mean[ mu(pi_e) + rho * (r - mu(a)) ]
    where mu(a) is the predicted reward under action a.
    """
    rho = np.where(actions == target_actions, 1.0 / np.clip(propensities, 1e-6, None), 0.0)
    mu_e = np.where(target_actions == 1, mu_hat[:, 1], mu_hat[:, 0])
    mu_a = np.where(actions == 1, mu_hat[:, 1], mu_hat[:, 0])
    dr = mu_e + rho * (rewards - mu_a)
    se = float(np.std(dr) / np.sqrt(len(dr)))
    return PolicyEvalResult(value=float(np.mean(dr)), estimator="DR", std_error=se)
