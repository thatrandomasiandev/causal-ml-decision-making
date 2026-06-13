"""Off-policy policy value estimators.

Provides both deterministic-policy convenience wrappers (``ips``, ``snips``,
``dr_policy_value``) and general-purpose estimators that accept pre-computed
importance-sampling weights (``ips_estimator``, ``snips_estimator``,
``dr_estimator``), plus diagnostics for weight clipping and effective sample
size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicyEvalResult:
    """Container for off-policy evaluation results.

    Attributes:
        value: Estimated policy value.
        estimator: Name of the estimator used.
        std_error: Standard error of the estimate, if available.
    """

    value: float
    estimator: str
    std_error: float | None = None


# ---------------------------------------------------------------------------
# Deterministic-policy convenience estimators (original API)
# ---------------------------------------------------------------------------


def ips(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
) -> PolicyEvalResult:
    """Inverse Propensity Scoring for deterministic target policies.

    V_IPS = (1/n) Σ 𝟙[a_i = π_e(x_i)] · r_i / π_b(a_i|x_i).

    Args:
        actions: Observed actions of shape (n,).
        rewards: Observed rewards of shape (n,).
        propensities: Behaviour-policy probabilities π_b(a|x) of shape (n,).
        target_actions: Actions selected by the evaluation policy, shape (n,).

    Returns:
        PolicyEvalResult with estimated value and standard error.
    """
    rho = np.where(
        actions == target_actions,
        1.0 / np.clip(propensities, 1e-6, None),
        0.0,
    )
    values = rho * rewards
    se = float(np.std(values) / np.sqrt(len(values)))
    return PolicyEvalResult(value=float(np.mean(values)), estimator="IPS", std_error=se)


def snips(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
) -> PolicyEvalResult:
    """Self-Normalized Inverse Propensity Scoring.

    V_SNIPS = Σ w_i r_i / Σ w_i.

    Args:
        actions: Observed actions of shape (n,).
        rewards: Observed rewards of shape (n,).
        propensities: Behaviour-policy probabilities of shape (n,).
        target_actions: Evaluation-policy actions of shape (n,).

    Returns:
        PolicyEvalResult with estimated value.
    """
    rho = np.where(
        actions == target_actions,
        1.0 / np.clip(propensities, 1e-6, None),
        0.0,
    )
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
    """Doubly robust policy value estimator for deterministic policies.

    V_DR = mean[μ̂(π_e) + ρ · (r − μ̂(a))].

    Args:
        actions: Observed actions of shape (n,).
        rewards: Observed rewards of shape (n,).
        propensities: Behaviour-policy probabilities of shape (n,).
        target_actions: Evaluation-policy actions of shape (n,).
        mu_hat: Predicted rewards of shape (n, n_actions).

    Returns:
        PolicyEvalResult with estimated value and standard error.
    """
    rho = np.where(
        actions == target_actions,
        1.0 / np.clip(propensities, 1e-6, None),
        0.0,
    )
    mu_e = np.where(target_actions == 1, mu_hat[:, 1], mu_hat[:, 0])
    mu_a = np.where(actions == 1, mu_hat[:, 1], mu_hat[:, 0])
    dr = mu_e + rho * (rewards - mu_a)
    se = float(np.std(dr) / np.sqrt(len(dr)))
    return PolicyEvalResult(value=float(np.mean(dr)), estimator="DR", std_error=se)


# ---------------------------------------------------------------------------
# General-purpose estimators (accept pre-computed importance weights)
# ---------------------------------------------------------------------------


def clip_weights(
    weights: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    """Clip importance-sampling weights to reduce variance.

    w_clipped = min(w, max_weight).

    Args:
        weights: Raw importance weights of shape (n,).
        max_weight: Upper bound for clipping.

    Returns:
        Clipped weight array of shape (n,).
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    clipped = np.clip(weights, None, max_weight)
    n_clipped = int(np.sum(weights > max_weight))
    if n_clipped > 0:
        logger.debug("Clipped %d / %d importance weights at %.2f", n_clipped, len(weights), max_weight)
    return clipped


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size for importance-sampling weights.

    ESS = (Σ w_i)² / Σ w_i².

    Args:
        weights: Importance weights of shape (n,).

    Returns:
        Effective sample size (scalar).
    """
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    if sum_w2 < 1e-30:
        return 0.0
    return float(sum_w ** 2 / sum_w2)


def ips_estimator(
    weights: np.ndarray,
    rewards: np.ndarray,
) -> PolicyEvalResult:
    """General Inverse Propensity Scoring with pre-computed weights.

    V_IPS = (1/n) Σ w_i · r_i where w_i = π_e(a_i|x_i) / π_b(a_i|x_i).

    Args:
        weights: Pre-computed importance-sampling ratios of shape (n,).
        rewards: Observed rewards of shape (n,).

    Returns:
        PolicyEvalResult with estimated value, standard error, and ESS logged.
    """
    n = len(rewards)
    values = weights * rewards
    mean_val = float(np.mean(values))
    se = float(np.std(values) / np.sqrt(n))
    logger.debug("IPS estimator: value=%.4f  ESS=%.1f / %d", mean_val, effective_sample_size(weights), n)
    return PolicyEvalResult(value=mean_val, estimator="IPS", std_error=se)


def snips_estimator(
    weights: np.ndarray,
    rewards: np.ndarray,
) -> PolicyEvalResult:
    """General Self-Normalized IPS with pre-computed weights.

    V_SNIPS = Σ w_i r_i / Σ w_i.

    Args:
        weights: Pre-computed importance-sampling ratios of shape (n,).
        rewards: Observed rewards of shape (n,).

    Returns:
        PolicyEvalResult with estimated value.
    """
    sum_w = np.sum(weights)
    if sum_w < 1e-12:
        return PolicyEvalResult(value=0.0, estimator="SNIPS", std_error=None)
    value = float(np.sum(weights * rewards) / sum_w)
    return PolicyEvalResult(value=value, estimator="SNIPS", std_error=None)


def dr_estimator(
    weights: np.ndarray,
    rewards: np.ndarray,
    q_hat: np.ndarray,
    dm_value: float,
) -> PolicyEvalResult:
    """General Doubly Robust estimator with pre-computed weights.

    V_DR = V_DM + (1/n) Σ w_i (r_i − Q̂(x_i, a_i)) where V_DM is the
    direct-method (fitted reward model) estimate.

    Args:
        weights: Pre-computed importance-sampling ratios of shape (n,).
        rewards: Observed rewards of shape (n,).
        q_hat: Fitted reward-model predictions Q̂(x_i, a_i) of shape (n,).
        dm_value: Direct-method policy value estimate (scalar).

    Returns:
        PolicyEvalResult with estimated value and standard error.
    """
    n = len(rewards)
    correction = weights * (rewards - q_hat)
    dr_values = dm_value + correction
    value = float(dm_value + np.mean(correction))
    se = float(np.std(dr_values) / np.sqrt(n))
    return PolicyEvalResult(value=value, estimator="DR", std_error=se)
