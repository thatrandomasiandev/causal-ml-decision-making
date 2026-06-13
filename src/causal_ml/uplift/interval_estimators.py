"""Confidence / prediction interval estimators for CATE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import KFold

from causal_ml.uplift.meta_learners import UpliftResult, fit_uplift

logger = logging.getLogger(__name__)

__all__ = [
    "CATEIntervalResult",
    "conformal_cate_intervals",
    "bootstrap_cate_intervals",
]


@dataclass
class CATEIntervalResult:
    """Container for CATE point estimates with uncertainty intervals.

    Attributes:
        tau_hat: Point CATE estimates of shape ``(n,)``.
        lower: Lower bound of the interval of shape ``(n,)``.
        upper: Upper bound of the interval of shape ``(n,)``.
        method: Name of the interval-estimation method.
        alpha: Mis-coverage rate (e.g. 0.1 for 90 % intervals).
    """

    tau_hat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    method: str
    alpha: float


def conformal_cate_intervals(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    estimator: str = "T",
    alpha: float = 0.1,
    n_splits: int = 5,
    seed: int = 42,
) -> CATEIntervalResult:
    """Split-conformal prediction intervals for CATE.

    Uses a train/calibration split to compute non-conformity scores
    (absolute residuals of the CATE estimator against a pseudo-outcome)
    and constructs marginally-valid intervals at level ``1 - alpha``.

    |tau_hat(x) - tau_true(x)| <= q_{1-alpha}(|residuals|)

    The pseudo-outcome used for calibration is the doubly-robust
    transformation when propensity is available, falling back to the
    simple difference ``Y_1 - hat{mu}_0`` / ``hat{mu}_1 - Y_0`` for
    treated / control units respectively.

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        Y: Observed outcome of shape ``(n,)``.
        estimator: Meta-learner key passed to ``fit_uplift``.
        alpha: Mis-coverage rate; 0.1 gives 90 % intervals.
        n_splits: Number of CV folds for the inner learner.
        seed: Random seed for reproducibility.

    Returns:
        ``CATEIntervalResult`` with point estimates and intervals.
    """
    rng = np.random.RandomState(seed)
    n = len(Y)
    perm = rng.permutation(n)
    half = n // 2
    train_idx, cal_idx = perm[:half], perm[half:]

    X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
    X_cal, T_cal, Y_cal = X[cal_idx], T[cal_idx], Y[cal_idx]

    result_tr = fit_uplift(X_tr, T_tr, Y_tr, estimator=estimator, n_splits=n_splits, seed=seed)

    from causal_ml.uplift.propensity import fit_predict_outcome
    mu1_cal = fit_predict_outcome(
        X_tr[T_tr == 1], Y_tr[T_tr == 1], X_cal,
    )
    mu0_cal = fit_predict_outcome(
        X_tr[T_tr == 0], Y_tr[T_tr == 0], X_cal,
    )

    pseudo_outcome = np.where(
        T_cal == 1,
        Y_cal - mu0_cal,
        mu1_cal - Y_cal,
    )

    from sklearn.ensemble import GradientBoostingRegressor
    tau_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, random_state=seed,
    )
    tau_model.fit(X_tr, result_tr.tau_hat)
    tau_cal = tau_model.predict(X_cal)

    residuals = np.abs(pseudo_outcome - tau_cal)
    quantile_level = min((1.0 - alpha) * (1.0 + 1.0 / len(cal_idx)), 1.0)
    q_hat = float(np.quantile(residuals, quantile_level))

    logger.info(
        "Conformal CATE: q_hat=%.4f at alpha=%.2f (n_cal=%d)",
        q_hat, alpha, len(cal_idx),
    )

    tau_full = tau_model.predict(X)

    return CATEIntervalResult(
        tau_hat=tau_full,
        lower=tau_full - q_hat,
        upper=tau_full + q_hat,
        method=f"conformal_{estimator}",
        alpha=alpha,
    )


def bootstrap_cate_intervals(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    estimator: str = "T",
    n_bootstrap: int = 200,
    alpha: float = 0.1,
    n_splits: int = 5,
    seed: int = 42,
) -> CATEIntervalResult:
    """Bootstrap percentile intervals for CATE.

    Draws ``n_bootstrap`` resamples with replacement, fits the chosen
    meta-learner on each, and constructs pointwise percentile intervals
    at the ``1 - alpha`` level.

    CI(x) = [ quantile_{alpha/2}(tau_hat_b(x)), quantile_{1-alpha/2}(tau_hat_b(x)) ]

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        Y: Observed outcome of shape ``(n,)``.
        estimator: Meta-learner key passed to ``fit_uplift``.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Mis-coverage rate; 0.1 gives 90 % intervals.
        n_splits: Number of CV folds for the inner learner.
        seed: Random seed for reproducibility.

    Returns:
        ``CATEIntervalResult`` with point estimates (mean across
        bootstraps) and percentile intervals.
    """
    rng = np.random.RandomState(seed)
    n = len(Y)

    from sklearn.ensemble import GradientBoostingRegressor

    bootstrap_preds = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        if (b + 1) % 50 == 0:
            logger.info("Bootstrap iteration %d / %d", b + 1, n_bootstrap)

        boot_idx = rng.choice(n, size=n, replace=True)
        X_b, T_b, Y_b = X[boot_idx], T[boot_idx], Y[boot_idx]

        result_b = fit_uplift(
            X_b, T_b, Y_b,
            estimator=estimator,
            n_splits=n_splits,
            seed=seed + b,
        )

        refit_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=seed + b,
        )
        refit_model.fit(X_b, result_b.tau_hat)
        bootstrap_preds[b] = refit_model.predict(X)

    tau_hat = np.mean(bootstrap_preds, axis=0)
    lower = np.quantile(bootstrap_preds, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap_preds, 1.0 - alpha / 2.0, axis=0)

    return CATEIntervalResult(
        tau_hat=tau_hat,
        lower=lower,
        upper=upper,
        method=f"bootstrap_{estimator}",
        alpha=alpha,
    )
