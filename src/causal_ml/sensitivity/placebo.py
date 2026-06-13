"""Placebo permutation test for causal estimators."""

from __future__ import annotations

from typing import Callable

import numpy as np

from causal_ml.utils.seed import set_seed


def placebo_test(
    estimator: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """
    Permutation test: shuffle treatment, re-run estimator, build null distribution.

    Parameters
    ----------
    estimator : callable
        Function (X, T, Y) -> scalar ATE estimate.
    X, T, Y : arrays
        Observational data.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        observed_ate, p_value, permutation_distribution
    """
    rng = set_seed(seed)
    T = T.astype(float).ravel()
    Y = Y.astype(float).ravel()

    observed = float(estimator(X, T, Y))
    null_ates = np.zeros(n_permutations)

    for i in range(n_permutations):
        T_perm = rng.permutation(T)
        null_ates[i] = estimator(X, T_perm, Y)

    p_value = float(np.mean(np.abs(null_ates) >= abs(observed)))

    return {
        "observed_ate": observed,
        "p_value": p_value,
        "permutation_distribution": null_ates,
    }
