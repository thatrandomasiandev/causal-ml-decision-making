"""Tests for uplift metrics and estimators."""

import numpy as np

from causal_ml.data.uplift_dgp import UpliftDGPConfig, generate_uplift_data
from causal_ml.uplift.doubly_robust import dr_learner
from causal_ml.uplift.meta_learners import fit_uplift
from causal_ml.uplift.metrics import ate_error, pehe


def test_pehe_zero_for_perfect_estimate():
    tau = np.array([1.0, 2.0, 3.0])
    assert pehe(tau, tau) == 0.0


def test_t_learner_runs():
    data = generate_uplift_data(UpliftDGPConfig(n_samples=300, seed=5))
    result = fit_uplift(data.X, data.T, data.Y, estimator="T", n_splits=2, seed=5)
    assert result.tau_hat.shape == (300,)
    assert pehe(result.tau_hat, data.ground_truth["tau"]) < 2.0


def test_dr_learner_runs():
    data = generate_uplift_data(UpliftDGPConfig(n_samples=300, seed=6))
    result = dr_learner(data.X, data.T, data.Y, n_splits=2, seed=6)
    assert result.estimator == "DR"
    assert ate_error(result.tau_hat, data.ground_truth["tau"]) < 1.0
