"""Propensity and outcome model helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = [
    "PropensityModel",
    "make_propensity_model",
    "make_outcome_model",
    "fit_predict_propensity",
    "fit_predict_outcome",
    "overlap_weights",
    "trim_propensity",
    "calibrate_propensity",
]


def make_propensity_model() -> Pipeline:
    """Create a default logistic-regression propensity pipeline.

    Returns:
        Scikit-learn ``Pipeline`` with standard scaling and logistic
        regression.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


def make_outcome_model() -> GradientBoostingRegressor:
    """Create a default gradient-boosted outcome model.

    Returns:
        ``GradientBoostingRegressor`` with moderate capacity defaults.
    """
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )


def fit_predict_propensity(
    train_X: np.ndarray, train_T: np.ndarray, test_X: np.ndarray,
) -> np.ndarray:
    """Fit propensity model on train data and predict on test data.

    Args:
        train_X: Training covariates of shape ``(n_train, p)``.
        train_T: Training treatment indicators of shape ``(n_train,)``.
        test_X: Test covariates of shape ``(n_test, p)``.

    Returns:
        Clipped propensity scores of shape ``(n_test,)`` in ``[0.01, 0.99]``.
    """
    model = make_propensity_model()
    model.fit(train_X, train_T)
    return np.clip(model.predict_proba(test_X)[:, 1], 0.01, 0.99)


def fit_predict_outcome(
    train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray,
) -> np.ndarray:
    """Fit outcome model on train data and predict on test data.

    Args:
        train_X: Training covariates of shape ``(n_train, p)``.
        train_y: Training outcomes of shape ``(n_train,)``.
        test_X: Test covariates of shape ``(n_test, p)``.

    Returns:
        Predicted outcomes of shape ``(n_test,)``.
    """
    model = make_outcome_model()
    model.fit(train_X, train_y)
    return model.predict(test_X)


# ---------------------------------------------------------------------------
# PropensityModel: auto-selecting wrapper
# ---------------------------------------------------------------------------


@dataclass
class PropensityModel:
    """Propensity-score estimator with automatic model selection.

    Trains both a ``LogisticRegression`` (with standard scaling) and a
    ``GradientBoostingClassifier``, evaluates each via 5-fold CV log-loss,
    and retains the better performer.

    Args:
        n_folds: Number of CV folds used for model selection.
        seed: Random seed for reproducibility.

    Attributes:
        best_model_: The fitted scikit-learn estimator chosen by CV.
        best_name_: Human-readable name of the winning model.
        cv_scores_: Dict mapping model name to mean CV negative log-loss.
    """

    n_folds: int = 5
    seed: int = 42
    best_model_: object = field(default=None, init=False, repr=False)
    best_name_: str = field(default="", init=False)
    cv_scores_: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, X: np.ndarray, T: np.ndarray) -> PropensityModel:
        """Fit and select the best propensity model via CV log-loss.

        Args:
            X: Covariate matrix of shape ``(n, p)``.
            T: Binary treatment indicator of shape ``(n,)``.

        Returns:
            ``self`` (fitted instance).
        """
        candidates: dict[str, object] = {
            "logistic": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=self.seed)),
            ]),
            "gbc": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.seed,
                )),
            ]),
        }

        best_score = -np.inf
        for name, model in candidates.items():
            scores = cross_val_score(
                model, X, T,
                cv=self.n_folds,
                scoring="neg_log_loss",
                error_score=-np.inf,
            )
            mean_score = float(np.mean(scores))
            self.cv_scores_[name] = mean_score
            logger.info(
                "PropensityModel CV neg_log_loss for %s: %.4f", name, mean_score,
            )
            if mean_score > best_score:
                best_score = mean_score
                self.best_name_ = name
                self.best_model_ = model

        self.best_model_.fit(X, T)  # type: ignore[union-attr]
        logger.info("PropensityModel selected: %s", self.best_name_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(T=1 | X), clipped to [0.01, 0.99].

        Args:
            X: Covariate matrix of shape ``(n, p)``.

        Returns:
            Propensity scores of shape ``(n,)``.
        """
        if self.best_model_ is None:
            raise RuntimeError("PropensityModel has not been fitted yet.")
        raw = self.best_model_.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        return np.clip(raw, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Propensity-score post-processing utilities
# ---------------------------------------------------------------------------


def overlap_weights(
    e_hat: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Compute overlap (tilting) weights for improved covariate balance.

    w_i = T_i * (1 - e(X_i)) + (1 - T_i) * e(X_i)

    Overlap weights emphasise the subpopulation with the most clinical
    equipoise (Li, Morgan & Zaslavsky, 2018).

    Args:
        e_hat: Propensity scores of shape ``(n,)``.
        T: Binary treatment indicator of shape ``(n,)``.

    Returns:
        Overlap weights of shape ``(n,)``.
    """
    e_safe = np.clip(e_hat, 1e-8, 1.0 - 1e-8)
    return T * (1.0 - e_safe) + (1.0 - T) * e_safe


def trim_propensity(
    e_hat: np.ndarray,
    trim_lo: float = 0.1,
    trim_hi: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove units with extreme propensity scores.

    Units outside ``[trim_lo, trim_hi]`` are considered positivity
    violators and should be excluded from downstream CATE estimation.

    Args:
        e_hat: Propensity scores of shape ``(n,)``.
        trim_lo: Lower bound (inclusive).
        trim_hi: Upper bound (inclusive).

    Returns:
        Tuple of ``(trimmed_scores, keep_mask)`` where ``keep_mask`` is a
        boolean array indicating retained units.
    """
    if not 0.0 < trim_lo < trim_hi < 1.0:
        raise ValueError(
            f"Trim bounds must satisfy 0 < trim_lo < trim_hi < 1, "
            f"got ({trim_lo}, {trim_hi})"
        )
    keep = (e_hat >= trim_lo) & (e_hat <= trim_hi)
    n_removed = int((~keep).sum())
    if n_removed > 0:
        logger.info(
            "trim_propensity removed %d / %d units (%.1f%%)",
            n_removed, len(e_hat), 100.0 * n_removed / len(e_hat),
        )
    return e_hat[keep], keep


def calibrate_propensity(
    X: np.ndarray,
    T: np.ndarray,
    seed: int = 42,
    n_folds: int = 5,
) -> np.ndarray:
    """Platt-scale propensity scores via CalibratedClassifierCV.

    Wraps the default logistic-regression propensity model with
    sigmoid (Platt) calibration using cross-validation so that
    returned probabilities are well-calibrated in the reliability-
    diagram sense.

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        seed: Random seed for reproducibility.
        n_folds: Number of cross-validation folds for calibration.

    Returns:
        Calibrated propensity scores of shape ``(n,)`` clipped to
        ``[0.01, 0.99]``.
    """
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=seed)),
    ])
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=n_folds,
    )
    calibrated.fit(X, T)
    probs = calibrated.predict_proba(X)[:, 1]
    return np.clip(probs, 0.01, 0.99)
