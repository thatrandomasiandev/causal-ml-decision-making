"""Doubly robust CATE estimation."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from causal_ml.uplift.meta_learners import UpliftResult
from causal_ml.uplift.propensity import fit_predict_outcome, fit_predict_propensity
from causal_ml.utils.crossfit import crossfit_predict

logger = logging.getLogger(__name__)

__all__ = [
    "dr_learner",
    "targeted_learning",
]


def dr_learner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """
    Doubly robust learner (AIPW pseudo-outcome regression).

    psi = mu1 - mu0 + T*(Y - mu1)/e - (1-T)*(Y - mu0)/(1-e)
    """
    e_hat = crossfit_predict(
        X, T, lambda tr_X, tr_T, te_X: fit_predict_propensity(tr_X, tr_T, te_X),
        n_splits=n_splits,
        seed=seed,
    )

    n = len(Y)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        tr_X, tr_T, tr_Y = X[train_idx], T[train_idx], Y[train_idx]
        te_X = X[test_idx]
        if (tr_T == 1).sum() > 0:
            mu1 = fit_predict_outcome(tr_X[tr_T == 1], tr_Y[tr_T == 1], te_X)
        else:
            mu1 = np.zeros(len(te_X))
        if (tr_T == 0).sum() > 0:
            mu0 = fit_predict_outcome(tr_X[tr_T == 0], tr_Y[tr_T == 0], te_X)
        else:
            mu0 = np.zeros(len(te_X))
        mu1_hat[test_idx] = mu1
        mu0_hat[test_idx] = mu0

    psi = (
        mu1_hat
        - mu0_hat
        + T * (Y - mu1_hat) / e_hat
        - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
    )

    tau_hat = crossfit_predict(
        X,
        psi,
        lambda tr_X, tr_y, te_X: fit_predict_outcome(tr_X, tr_y, te_X),
        n_splits=n_splits,
        seed=seed + 1,
    )

    return UpliftResult(tau_hat=tau_hat, estimator="DR")


def _compute_clever_covariate(
    T: np.ndarray,
    e_hat: np.ndarray,
) -> np.ndarray:
    """Compute the clever covariate H(A, W) used in TMLE targeting.

    H(A, W) = T / e(W) - (1 - T) / (1 - e(W))

    This is the efficient influence-function component that makes the
    initial estimator *targeted* toward the ATE functional.

    Args:
        T: Binary treatment indicator of shape ``(n,)``.
        e_hat: Estimated propensity scores of shape ``(n,)``, clipped
            away from 0 and 1.

    Returns:
        Clever covariate array of shape ``(n,)``.
    """
    e_safe = np.clip(e_hat, 1e-6, 1.0 - 1e-6)
    return T / e_safe - (1 - T) / (1.0 - e_safe)


def targeted_learning(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """One-step targeted minimum loss-based estimation (TMLE) for CATE.

    Performs the TMLE epsilon-update on initial outcome predictions to
    reduce finite-sample bias in the doubly-robust pseudo-outcome.

    1. Fit cross-validated outcome models mu0, mu1 and propensity e.
    2. Compute clever covariate H(A,W).
    3. Regress residual Y - mu_initial onto H(A,W) to obtain epsilon.
    4. Update: mu*_a = mu_a + epsilon * H_a  for a in {0, 1}.
    5. Construct DR pseudo-outcome from updated predictions.

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        Y: Observed outcome of shape ``(n,)``.
        n_splits: Number of cross-validation folds.
        seed: Random seed for reproducibility.

    Returns:
        ``UpliftResult`` with TMLE-corrected CATE predictions and
        estimator label ``"TMLE"``.
    """
    e_hat = crossfit_predict(
        X, T, lambda tr_X, tr_T, te_X: fit_predict_propensity(tr_X, tr_T, te_X),
        n_splits=n_splits,
        seed=seed,
    )
    e_hat = np.clip(e_hat, 1e-6, 1.0 - 1e-6)

    n = len(Y)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        tr_X, tr_T, tr_Y = X[train_idx], T[train_idx], Y[train_idx]
        te_X = X[test_idx]
        if (tr_T == 1).sum() > 0:
            mu1_hat[test_idx] = fit_predict_outcome(
                tr_X[tr_T == 1], tr_Y[tr_T == 1], te_X,
            )
        if (tr_T == 0).sum() > 0:
            mu0_hat[test_idx] = fit_predict_outcome(
                tr_X[tr_T == 0], tr_Y[tr_T == 0], te_X,
            )

    mu_initial = T * mu1_hat + (1 - T) * mu0_hat
    H = _compute_clever_covariate(T, e_hat)
    residual = Y - mu_initial

    eps_model = LinearRegression(fit_intercept=False)
    eps_model.fit(H.reshape(-1, 1), residual)
    epsilon = float(eps_model.coef_[0])
    logger.debug("TMLE epsilon update: %.6f", epsilon)

    H1 = 1.0 / np.clip(e_hat, 1e-6, None)
    H0 = -1.0 / np.clip(1.0 - e_hat, 1e-6, None)
    mu1_star = mu1_hat + epsilon * H1
    mu0_star = mu0_hat + epsilon * H0

    psi = (
        mu1_star
        - mu0_star
        + T * (Y - mu1_star) / e_hat
        - (1 - T) * (Y - mu0_star) / (1.0 - e_hat)
    )

    tau_hat = crossfit_predict(
        X,
        psi,
        lambda tr_X, tr_y, te_X: fit_predict_outcome(tr_X, tr_y, te_X),
        n_splits=n_splits,
        seed=seed + 1,
    )

    return UpliftResult(tau_hat=tau_hat, estimator="TMLE")
