"""Uplift evaluation metrics."""

from __future__ import annotations

import numpy as np


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
