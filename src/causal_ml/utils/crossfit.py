"""K-fold cross-fitting for nuisance model estimation."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import KFold


def crossfit_predict(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Out-of-fold predictions using cross-fitting.

    fit_predict_fn(train_X, train_y, test_X) -> predictions on test_X
    """
    n = len(y)
    preds = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        preds[test_idx] = fit_predict_fn(X[train_idx], y[train_idx], X[test_idx])

    return preds


def crossfit_multi(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    fit_predict_fns: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Cross-fit multiple nuisance models in parallel folds."""
    return {
        name: crossfit_predict(X, targets[name], fn, n_splits=n_splits, seed=seed)
        for name, fn in fit_predict_fns.items()
    }
