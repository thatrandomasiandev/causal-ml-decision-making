"""Meta-learners for heterogeneous treatment effect estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from causal_ml.uplift.propensity import fit_predict_propensity, make_outcome_model
from causal_ml.utils.crossfit import crossfit_predict


@dataclass
class UpliftResult:
    tau_hat: np.ndarray
    estimator: str


def _augment_treatment(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.column_stack([X, T])


def s_learner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """Single model: predict Y from (X, T); CATE = mu1 - mu0."""
    n = len(Y)
    tau_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        tr_X, tr_T, tr_Y = X[train_idx], T[train_idx], Y[train_idx]
        te_X = X[test_idx]
        model = make_outcome_model()
        model.fit(_augment_treatment(tr_X, tr_T), tr_Y)
        mu1 = model.predict(_augment_treatment(te_X, np.ones(len(te_X))))
        mu0 = model.predict(_augment_treatment(te_X, np.zeros(len(te_X))))
        tau_hat[test_idx] = mu1 - mu0

    return UpliftResult(tau_hat=tau_hat, estimator="S")


def t_learner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """Separate outcome models per treatment arm."""
    n = len(Y)
    tau_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        tr_X, tr_T, tr_Y = X[train_idx], T[train_idx], Y[train_idx]
        te_X = X[test_idx]
        m1 = make_outcome_model()
        m0 = make_outcome_model()
        if (tr_T == 1).sum() > 0:
            m1.fit(tr_X[tr_T == 1], tr_Y[tr_T == 1])
        if (tr_T == 0).sum() > 0:
            m0.fit(tr_X[tr_T == 0], tr_Y[tr_T == 0])
        tau_hat[test_idx] = m1.predict(te_X) - m0.predict(te_X)

    return UpliftResult(tau_hat=tau_hat, estimator="T")


def x_learner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """X-learner with imputed treatment effects."""
    e_hat = crossfit_predict(
        X, T, lambda tr_X, tr_T, te_X: fit_predict_propensity(tr_X, tr_T, te_X),
        n_splits=n_splits,
        seed=seed,
    )

    n = len(Y)
    tau_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        tr_X, tr_T, tr_Y = X[train_idx], T[train_idx], Y[train_idx]
        te_X = X[test_idx]
        m1 = make_outcome_model()
        m0 = make_outcome_model()
        if (tr_T == 1).sum() > 0:
            m1.fit(tr_X[tr_T == 1], tr_Y[tr_T == 1])
        if (tr_T == 0).sum() > 0:
            m0.fit(tr_X[tr_T == 0], tr_Y[tr_T == 0])
        mu1 = m1.predict(tr_X)
        mu0 = m0.predict(tr_X)
        impute1 = tr_Y - mu0
        impute0 = mu1 - tr_Y
        tau1_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        tau0_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed + 1)
        if (tr_T == 1).sum() > 0:
            tau1_model.fit(tr_X[tr_T == 1], impute1[tr_T == 1])
        if (tr_T == 0).sum() > 0:
            tau0_model.fit(tr_X[tr_T == 0], impute0[tr_T == 0])
        g1 = tau1_model.predict(te_X)
        g0 = tau0_model.predict(te_X)
        e_te = e_hat[test_idx]
        tau_hat[test_idx] = e_te * g0 + (1 - e_te) * g1

    return UpliftResult(tau_hat=tau_hat, estimator="X")


def r_learner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """R-learner (Robinson decomposition)."""
    from causal_ml.uplift.propensity import fit_predict_outcome

    e_hat = crossfit_predict(
        X, T, lambda tr_X, tr_T, te_X: fit_predict_propensity(tr_X, tr_T, te_X),
        n_splits=n_splits,
        seed=seed,
    )
    m_hat = crossfit_predict(
        X, Y, lambda tr_X, tr_Y, te_X: fit_predict_outcome(tr_X, tr_Y, te_X),
        n_splits=n_splits,
        seed=seed,
    )

    residual_y = Y - m_hat
    residual_t = T - e_hat
    weights = np.clip(residual_t**2, 1e-6, None)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = len(Y)
    tau_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X_scaled):
        tr_X, te_X = X_scaled[train_idx], X_scaled[test_idx]
        ry = residual_y[train_idx]
        rt = residual_t[train_idx]
        w = weights[train_idx]
        pseudo = ry / np.where(np.abs(rt) < 1e-6, 1e-6, rt)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        model.fit(tr_X, pseudo, sample_weight=w)
        tau_hat[test_idx] = model.predict(te_X)

    return UpliftResult(tau_hat=tau_hat, estimator="R")


ESTIMATORS = {
    "S": s_learner,
    "T": t_learner,
    "X": x_learner,
    "R": r_learner,
}


def fit_uplift(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    estimator: str = "T",
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """Dispatch to the requested meta-learner."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator}. Choose from {list(ESTIMATORS)}")
    return ESTIMATORS[estimator](X, T, Y, n_splits=n_splits, seed=seed)
