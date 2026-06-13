"""Tests for sensitivity analysis module."""

import numpy as np
import pytest

from causal_ml.sensitivity.evalue import evalue
from causal_ml.sensitivity.placebo import placebo_test
from causal_ml.sensitivity.rosenbaum import rosenbaum_bounds


def test_rosenbaum_bounds_widen_with_gamma():
    rng = np.random.default_rng(42)
    n = 100
    T = rng.binomial(1, 0.5, n)
    Y = T * 2.0 + rng.normal(0, 1, n)
    gammas = [1.0, 1.5, 2.0, 3.0]
    df = rosenbaum_bounds(Y, T, gammas)
    assert list(df["gamma"]) == gammas
    # Upper p-value bound should be non-decreasing in gamma
    assert df["p_upper"].is_monotonic_increasing


def test_evalue_at_least_one():
    result = evalue(ate=0.5, ate_se=0.1)
    assert result["evalue"] >= 1.0
    assert result["evalue_lower"] >= 1.0


def test_evalue_raises_on_zero_se():
    with pytest.raises(ValueError, match="ate_se must be positive"):
        evalue(0.5, 0.0)


def _naive_ate(X, T, Y):
    return float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))


def test_placebo_p_value_under_null():
    rng = np.random.default_rng(99)
    n = 200
    X = rng.standard_normal((n, 3))
    T = rng.binomial(1, 0.5, n).astype(float)
    Y = rng.normal(0, 1, n)  # outcome independent of T
    result = placebo_test(_naive_ate, X, T, Y, n_permutations=100, seed=99)
    assert 0 <= result["p_value"] <= 1.0
    assert len(result["permutation_distribution"]) == 100
