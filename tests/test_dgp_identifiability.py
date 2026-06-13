"""Tests for synthetic DGP invariants."""

import numpy as np

from causal_ml.data.bandit_dgp import BanditDGPConfig, generate_bandit_data, policy_value
from causal_ml.data.dag_dgp import DAGDGPConfig, generate_dag_data
from causal_ml.data.uplift_dgp import UpliftDGPConfig, generate_uplift_data


def test_uplift_ground_truth_shapes():
    data = generate_uplift_data(UpliftDGPConfig(n_samples=500, seed=0))
    assert data.X.shape == (500, 10)
    assert len(data.ground_truth["tau"]) == 500
    assert 0.0 <= np.mean(data.ground_truth["propensity"]) <= 1.0


def test_uplift_observed_outcome_consistency():
    data = generate_uplift_data(UpliftDGPConfig(n_samples=200, seed=1))
    y0 = data.ground_truth["y0"]
    y1 = data.ground_truth["y1"]
    expected = data.T * y1 + (1 - data.T) * y0
    np.testing.assert_allclose(data.Y, expected)


def test_bandit_oracle_value_exceeds_random():
    data = generate_bandit_data(BanditDGPConfig(n_samples=1000, seed=2))
    y0 = data.ground_truth["y0"]
    y1 = data.ground_truth["y1"]
    random_actions = np.random.default_rng(2).binomial(1, 0.5, len(y0))
    random_value = policy_value(random_actions, y0, y1)
    assert data.ground_truth["oracle_value"] >= random_value - 0.5


def test_dag_acyclicity():
    data = generate_dag_data(DAGDGPConfig(n_samples=100, n_nodes=4, seed=3))
    adj = data.ground_truth["adjacency"]
    # Upper triangular => acyclic
    assert np.allclose(np.tril(adj, k=0), 0)


def test_time_series_shape():
    data = generate_dag_data(DAGDGPConfig(n_timesteps=100, time_series=True, seed=4))
    assert data.is_time_series
    assert data.X.ndim == 2
    assert data.adjacency.ndim == 3
