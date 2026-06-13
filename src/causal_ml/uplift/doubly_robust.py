"""Doubly robust CATE estimation."""

from __future__ import annotations

import numpy as np

from causal_ml.uplift.meta_learners import UpliftResult
from causal_ml.uplift.propensity import fit_predict_outcome, fit_predict_propensity
from causal_ml.utils.crossfit import crossfit_predict


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

    from sklearn.model_selection import KFold

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
