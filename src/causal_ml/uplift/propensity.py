"""Propensity and outcome model helpers."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_propensity_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


def make_outcome_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )


def fit_predict_propensity(train_X: np.ndarray, train_T: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    model = make_propensity_model()
    model.fit(train_X, train_T)
    return np.clip(model.predict_proba(test_X)[:, 1], 0.01, 0.99)


def fit_predict_outcome(
    train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray
) -> np.ndarray:
    model = make_outcome_model()
    model.fit(train_X, train_y)
    return model.predict(test_X)
