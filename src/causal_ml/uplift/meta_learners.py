"""Meta-learners for heterogeneous treatment effect estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from causal_ml.uplift.propensity import fit_predict_propensity, make_outcome_model
from causal_ml.utils.crossfit import crossfit_predict

logger = logging.getLogger(__name__)

__all__ = [
    "UpliftResult",
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
    "causal_forest",
    "fit_uplift",
    "ESTIMATORS",
    "_cross_val_cate",
]


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


def _cross_val_cate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    learner_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ],
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Shared K-fold cross-validation harness for CATE estimation.

    Splits data into ``n_splits`` folds.  On each fold the caller-supplied
    ``learner_fn(train_X, train_T, train_Y, test_X) -> tau_hat`` is invoked
    and the out-of-fold predictions are stitched together.

    tau_hat_i = learner_fn(X_{-i}, T_{-i}, Y_{-i}, X_i)

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        Y: Observed outcome of shape ``(n,)``.
        learner_fn: A callable ``(train_X, train_T, train_Y, test_X) -> tau``
            that fits on training data and returns CATE predictions for
            ``test_X``.
        n_splits: Number of cross-validation folds.
        seed: Random seed for reproducible fold assignment.

    Returns:
        Out-of-fold CATE predictions of shape ``(n,)``.
    """
    n = len(Y)
    tau_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.debug("_cross_val_cate fold %d / %d", fold_idx + 1, n_splits)
        tau_hat[test_idx] = learner_fn(
            X[train_idx], T[train_idx], Y[train_idx], X[test_idx],
        )

    return tau_hat


def causal_forest(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    n_trees: int = 500,
    min_samples_leaf: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """Honest causal forest via ExtraTreesRegressor with honesty splitting.

    The algorithm follows the *honesty* principle of Athey & Imbens (2016):
    within each cross-validation fold the training set is split 50/50 into a
    *structure* half (used to grow tree splits) and an *estimation* half
    (used to populate leaf-level CATE estimates).

    tau_hat(x) = E[Y(1) - Y(0) | leaf(x)]  estimated on the held-out half

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        Y: Observed outcome of shape ``(n,)``.
        n_splits: Number of outer cross-validation folds.
        n_trees: Number of trees in the ensemble.
        min_samples_leaf: Minimum samples per leaf in each tree.
        seed: Random seed for reproducibility.

    Returns:
        ``UpliftResult`` with out-of-fold CATE predictions and estimator
        label ``"CF"``.
    """

    def _honest_forest_learner(
        train_X: np.ndarray,
        train_T: np.ndarray,
        train_Y: np.ndarray,
        test_X: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        n_train = len(train_Y)
        perm = rng.permutation(n_train)
        half = n_train // 2
        struct_idx, est_idx = perm[:half], perm[half:]

        forest = ExtraTreesRegressor(
            n_estimators=n_trees,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        forest.fit(train_X[struct_idx], train_Y[struct_idx])

        est_leaves = forest.apply(train_X[est_idx])
        test_leaves = forest.apply(test_X)

        est_T = train_T[est_idx]
        est_Y = train_Y[est_idx]

        n_test = len(test_X)
        tau_out = np.zeros(n_test)

        for tree_idx in range(n_trees):
            est_leaf_ids = est_leaves[:, tree_idx]
            test_leaf_ids = test_leaves[:, tree_idx]

            unique_leaves = np.unique(test_leaf_ids)
            for leaf_id in unique_leaves:
                in_leaf = est_leaf_ids == leaf_id
                t_mask = in_leaf & (est_T == 1)
                c_mask = in_leaf & (est_T == 0)

                mu1 = est_Y[t_mask].mean() if t_mask.sum() > 0 else 0.0
                mu0 = est_Y[c_mask].mean() if c_mask.sum() > 0 else 0.0
                leaf_tau = mu1 - mu0

                test_in_leaf = test_leaf_ids == leaf_id
                tau_out[test_in_leaf] += leaf_tau

        tau_out /= n_trees
        return tau_out

    tau_hat = _cross_val_cate(
        X, T, Y,
        learner_fn=_honest_forest_learner,
        n_splits=n_splits,
        seed=seed,
    )
    return UpliftResult(tau_hat=tau_hat, estimator="CF")


ESTIMATORS: dict[str, Callable[..., UpliftResult]] = {
    "S": s_learner,
    "T": t_learner,
    "X": x_learner,
    "R": r_learner,
    "CF": causal_forest,
}


def fit_uplift(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    estimator: str = "T",
    n_splits: int = 5,
    seed: int = 42,
) -> UpliftResult:
    """Dispatch to the requested meta-learner.

    Args:
        X: Covariate matrix of shape ``(n, p)``.
        T: Binary treatment indicator of shape ``(n,)``.
        Y: Observed outcome of shape ``(n,)``.
        estimator: Key in ``ESTIMATORS`` (e.g. ``"T"``, ``"CF"``).
        n_splits: Number of cross-validation folds.
        seed: Random seed for reproducibility.

    Returns:
        ``UpliftResult`` produced by the selected learner.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator}. Choose from {list(ESTIMATORS)}")
    return ESTIMATORS[estimator](X, T, Y, n_splits=n_splits, seed=seed)
