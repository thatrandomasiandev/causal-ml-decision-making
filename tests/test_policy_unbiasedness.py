"""Tests for off-policy evaluation."""

import numpy as np

from causal_ml.data.bandit_dgp import BanditDGPConfig, generate_bandit_data
from causal_ml.policy.evaluation import dr_policy_value, ips, snips
from causal_ml.policy.learning import threshold_policy
from causal_ml.policy.simulation import simulate_from_tau
from causal_ml.uplift.meta_learners import fit_uplift


def test_ips_runs_on_bandit_data():
    data = generate_bandit_data(BanditDGPConfig(n_samples=500, epsilon=0.5, seed=7))
    target = data.ground_truth["oracle_actions"]
    result = ips(data.actions, data.rewards, data.propensities, target)
    assert np.isfinite(result.value)


def test_snips_runs_on_bandit_data():
    data = generate_bandit_data(BanditDGPConfig(n_samples=500, seed=8))
    target = data.ground_truth["oracle_actions"]
    result = snips(data.actions, data.rewards, data.propensities, target)
    assert np.isfinite(result.value)


def test_regret_nonnegative_for_oracle():
    data = generate_bandit_data(BanditDGPConfig(n_samples=500, seed=9))
    uplift = fit_uplift(data.X, data.actions, data.rewards, estimator="T", n_splits=2, seed=9)
    sim = simulate_from_tau(
        uplift.tau_hat,
        data.ground_truth["y0"],
        data.ground_truth["y1"],
        data.ground_truth["oracle_value"],
    )
    assert sim.regret >= -1e-6


def test_dr_policy_value_runs():
    data = generate_bandit_data(BanditDGPConfig(n_samples=300, seed=10))
    target = threshold_policy(data.ground_truth["tau"])
    y0 = data.ground_truth["y0"]
    y1 = data.ground_truth["y1"]
    mu_hat = np.column_stack([y0, y1])
    result = dr_policy_value(data.actions, data.rewards, data.propensities, target, mu_hat)
    assert np.isfinite(result.value)
