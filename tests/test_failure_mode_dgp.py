"""Tests for FailureModeDGP."""

import numpy as np

from causal_ml.data.synthetic import FailureModeDGP, generate_failure_mode_data, overlap_to_alpha
from causal_ml.uplift.meta_learners import fit_uplift
from causal_ml.uplift.metrics import pehe


def test_reproducibility_same_seed():
    d1 = FailureModeDGP(n=100, seed=42).generate()
    d2 = FailureModeDGP(n=100, seed=42).generate()
    np.testing.assert_array_equal(d1.X, d2.X)
    np.testing.assert_array_equal(d1.T, d2.T)
    np.testing.assert_array_equal(d1.Y, d2.Y)


def test_shape_correctness():
    data = generate_failure_mode_data(n=300, p=15, seed=7)
    assert data.X.shape == (300, 15)
    assert data.T.shape == (300,)
    assert data.Y.shape == (300,)
    assert len(data.ground_truth["tau"]) == 300


def test_ate_recovery_low_confounding():
    data = FailureModeDGP(
        n=3000, overlap=1.0, confounding=0.0, heterogeneity=0.0, seed=10
    ).generate()
    tau = data.ground_truth["tau"]
    assert abs(np.mean(tau) - 1.0) < 0.01


def test_pehe_nonzero_with_heterogeneity():
    data = FailureModeDGP(n=500, heterogeneity=1.0, confounding=0.5, seed=11).generate()
    result = fit_uplift(data.X, data.T, data.Y, estimator="T", n_splits=2, seed=11)
    assert pehe(result.tau_hat, data.ground_truth["tau"]) > 0


def test_overlap_alpha_monotone():
    assert overlap_to_alpha(1.0) < overlap_to_alpha(0.5)
    assert overlap_to_alpha(0.5) < overlap_to_alpha(0.1)
