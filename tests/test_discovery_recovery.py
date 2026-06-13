"""Tests for causal discovery."""

import numpy as np

from causal_ml.data.dag_dgp import DAGDGPConfig, generate_dag_data
from causal_ml.discovery.metrics import edge_precision_recall, structural_hamming_distance
from causal_ml.discovery.notears import notears


def test_notears_runs():
    data = generate_dag_data(DAGDGPConfig(n_samples=500, n_nodes=4, seed=11))
    result = notears(data.X, lambda1=0.1)
    assert result.adjacency.shape == (4, 4)


def test_shd_zero_for_identical_graphs():
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    assert structural_hamming_distance(adj, adj) == 0


def test_precision_recall_perfect():
    adj = np.array([[0, 1], [0, 0]], dtype=float)
    pr = edge_precision_recall(adj, adj)
    assert pr["precision"] == 1.0
    assert pr["recall"] == 1.0
