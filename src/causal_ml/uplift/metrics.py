"""Uplift evaluation metrics."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "pehe",
    "ate_error",
    "auuc",
    "qini",
    "policy_value",
    "qini_coefficient",
]


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Precision in Estimation of Heterogeneous Effects."""
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_error(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Absolute error in average treatment effect."""
    return float(abs(np.mean(tau_hat) - np.mean(tau_true)))


def _cumulative_gain(tau: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Cumulative uplift gain when targeting top-k by predicted tau."""
    order = np.argsort(-tau)
    sorted_outcomes = outcomes[order]
    n = len(tau)
    cumulative = np.cumsum(sorted_outcomes) / np.arange(1, n + 1)
    return cumulative


def auuc(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """
    Area under the uplift curve (trapezoid approximation).

    Uses true tau as outcome proxy for ranking evaluation.
    """
    n = len(tau_hat)
    if n < 2:
        return 0.0
    gain_hat = _cumulative_gain(tau_hat, tau_true)
    gain_oracle = _cumulative_gain(tau_true, tau_true)
    x = np.linspace(0, 1, n)
    # Normalized AUUC: area between model curve and random baseline
    random_baseline = np.full(n, gain_oracle[-1])
    model_area = np.trapz(gain_hat - random_baseline, x)
    oracle_area = np.trapz(gain_oracle - random_baseline, x)
    if abs(oracle_area) < 1e-12:
        return 0.0
    return float(model_area / oracle_area)


def qini(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Qini coefficient approximation using true tau."""
    n = len(tau_hat)
    if n < 2:
        return 0.0
    order = np.argsort(-tau_hat)
    sorted_tau = tau_true[order]
    cum_treat = np.cumsum(sorted_tau)
    x = np.arange(1, n + 1) / n
    return float(np.trapz(cum_treat / n, x))


def policy_value(
    tau_hat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """IPW estimate of the policy value for a treat-above-threshold rule.

    V(pi) = E[ Y * I(tau_hat > threshold) * T / e
              + Y * I(tau_hat <= threshold) * (1 - T) / (1 - e) ]

    Uses inverse-propensity weighting so the metric is valid even under
    observational data, provided ``e_hat`` is well-specified.

    Args:
        tau_hat: Predicted CATE of shape ``(n,)``.
        Y: Observed outcomes of shape ``(n,)``.
        T: Binary treatment indicator of shape ``(n,)``.
        e_hat: Propensity scores of shape ``(n,)``, clipped away from 0/1.
        threshold: CATE threshold; units with ``tau_hat > threshold`` are
            assigned treatment under the policy.

    Returns:
        Estimated expected outcome under the targeting policy.
    """
    e_safe = np.clip(e_hat, 1e-6, 1.0 - 1e-6)
    treat_rule = (tau_hat > threshold).astype(float)
    control_rule = 1.0 - treat_rule

    ipw_treat = Y * treat_rule * T / e_safe
    ipw_control = Y * control_rule * (1.0 - T) / (1.0 - e_safe)

    return float(np.mean(ipw_treat + ipw_control))


def qini_coefficient(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
) -> float:
    """Normalised Qini coefficient.

    Qini_coeff = (AUUC_model - AUUC_random) / (AUUC_perfect - AUUC_random)

    Unlike the simpler ``qini`` function this computes the *normalized*
    coefficient that lies in [-inf, 1], where 1 indicates a perfect ranker
    and 0 corresponds to random targeting.

    Args:
        tau_hat: Predicted CATE of shape ``(n,)``.
        tau_true: Ground-truth CATE of shape ``(n,)``.

    Returns:
        Normalised Qini coefficient (float).  Returns 0.0 if the perfect
        curve is degenerate.
    """
    n = len(tau_hat)
    if n < 2:
        return 0.0

    fractions = np.arange(1, n + 1) / n

    def _cum_gain(ordering: np.ndarray) -> np.ndarray:
        sorted_tau = tau_true[ordering]
        return np.cumsum(sorted_tau) / n

    model_order = np.argsort(-tau_hat)
    perfect_order = np.argsort(-tau_true)
    random_gain = np.mean(tau_true) * fractions

    model_gain = _cum_gain(model_order)
    perfect_gain = _cum_gain(perfect_order)

    model_area = float(np.trapz(model_gain - random_gain, fractions))
    perfect_area = float(np.trapz(perfect_gain - random_gain, fractions))

    if abs(perfect_area) < 1e-12:
        return 0.0

    return model_area / perfect_area
