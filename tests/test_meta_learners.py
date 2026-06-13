"""Comprehensive tests for uplift meta-learners.

Uses a simple synthetic DGP with known treatment effect tau(x) = x[:,0] so
that every meta-learner can recover it with low PEHE on modest sample sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_ml.data.base import UpliftDataset
from causal_ml.uplift.doubly_robust import dr_learner
from causal_ml.uplift.meta_learners import UpliftResult, fit_uplift
from causal_ml.uplift.metrics import pehe

_SEED = 99
_N = 500
_D = 5
_N_SPLITS = 3


@pytest.fixture(scope="module")
def simple_dgp() -> UpliftDataset:
    """Generate a synthetic DGP where tau(x) = x[:,0].

    Y(0) = 0.5*x[:,1] + noise
    Y(1) = Y(0) + x[:,0]
    T ~ Bernoulli(0.5)   (RCT — no confounding)
    """
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N, _D))

    tau = X[:, 0].copy()
    y0 = 0.5 * X[:, 1] + rng.normal(0, 0.3, size=_N)
    y1 = y0 + tau

    propensity = np.full(_N, 0.5)
    T = rng.binomial(1, propensity).astype(float)
    Y = T * y1 + (1.0 - T) * y0

    return UpliftDataset(
        X=X,
        T=T,
        Y=Y,
        metadata={"dgp": "test_simple", "n_samples": _N, "n_features": _D, "seed": _SEED},
        ground_truth={
            "tau": tau,
            "ate": float(np.mean(tau)),
            "propensity": propensity,
            "y0": y0,
            "y1": y1,
        },
    )


class TestSLearner:
    """S-learner correctness and contract tests."""

    def test_returns_uplift_result(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="S", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert isinstance(result, UpliftResult)

    def test_correct_shape(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="S", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.tau_hat.shape == (_N,)

    def test_pehe_below_threshold(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="S", n_splits=_N_SPLITS, seed=_SEED,
        )
        score = pehe(result.tau_hat, simple_dgp.ground_truth["tau"])
        assert score < 1.0, f"S-learner PEHE={score:.3f} exceeds 1.0"

    def test_estimator_label(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="S", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.estimator == "S"


class TestTLearner:
    """T-learner correctness and contract tests."""

    def test_returns_uplift_result(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="T", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert isinstance(result, UpliftResult)

    def test_correct_shape(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="T", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.tau_hat.shape == (_N,)

    def test_pehe_below_threshold(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="T", n_splits=_N_SPLITS, seed=_SEED,
        )
        score = pehe(result.tau_hat, simple_dgp.ground_truth["tau"])
        assert score < 1.0, f"T-learner PEHE={score:.3f} exceeds 1.0"

    def test_estimator_label(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="T", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.estimator == "T"


class TestXLearner:
    """X-learner correctness and contract tests."""

    def test_returns_uplift_result(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="X", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert isinstance(result, UpliftResult)

    def test_correct_shape(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="X", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.tau_hat.shape == (_N,)

    def test_pehe_below_threshold(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="X", n_splits=_N_SPLITS, seed=_SEED,
        )
        score = pehe(result.tau_hat, simple_dgp.ground_truth["tau"])
        assert score < 1.0, f"X-learner PEHE={score:.3f} exceeds 1.0"

    def test_estimator_label(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="X", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.estimator == "X"


class TestRLearner:
    """R-learner correctness and contract tests."""

    def test_returns_uplift_result(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="R", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert isinstance(result, UpliftResult)

    def test_correct_shape(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="R", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.tau_hat.shape == (_N,)

    def test_pehe_below_threshold(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="R", n_splits=_N_SPLITS, seed=_SEED,
        )
        score = pehe(result.tau_hat, simple_dgp.ground_truth["tau"])
        assert score < 1.0, f"R-learner PEHE={score:.3f} exceeds 1.0"

    def test_estimator_label(self, simple_dgp: UpliftDataset) -> None:
        result = fit_uplift(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            estimator="R", n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.estimator == "R"


class TestDRLearner:
    """Doubly robust learner correctness and contract tests."""

    def test_returns_uplift_result(self, simple_dgp: UpliftDataset) -> None:
        result = dr_learner(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            n_splits=_N_SPLITS, seed=_SEED,
        )
        assert isinstance(result, UpliftResult)

    def test_correct_shape(self, simple_dgp: UpliftDataset) -> None:
        result = dr_learner(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.tau_hat.shape == (_N,)

    def test_pehe_below_threshold(self, simple_dgp: UpliftDataset) -> None:
        result = dr_learner(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            n_splits=_N_SPLITS, seed=_SEED,
        )
        score = pehe(result.tau_hat, simple_dgp.ground_truth["tau"])
        assert score < 1.0, f"DR-learner PEHE={score:.3f} exceeds 1.0"

    def test_estimator_label(self, simple_dgp: UpliftDataset) -> None:
        result = dr_learner(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            n_splits=_N_SPLITS, seed=_SEED,
        )
        assert result.estimator == "DR"


class TestAgreement:
    """Cross-learner consistency checks."""

    ESTIMATORS = ["S", "T", "X", "R"]

    def test_all_learners_agree_on_ate_sign(self, simple_dgp: UpliftDataset) -> None:
        """All meta-learners should agree on the sign of the ATE."""
        ate_true = simple_dgp.ground_truth["ate"]
        expected_sign = np.sign(ate_true)

        for est in self.ESTIMATORS:
            result = fit_uplift(
                simple_dgp.X, simple_dgp.T, simple_dgp.Y,
                estimator=est, n_splits=_N_SPLITS, seed=_SEED,
            )
            ate_hat = float(np.mean(result.tau_hat))
            assert np.sign(ate_hat) == expected_sign, (
                f"{est}-learner ATE sign mismatch: ate_hat={ate_hat:.3f}, "
                f"ate_true={ate_true:.3f}"
            )

        dr_result = dr_learner(
            simple_dgp.X, simple_dgp.T, simple_dgp.Y,
            n_splits=_N_SPLITS, seed=_SEED,
        )
        ate_dr = float(np.mean(dr_result.tau_hat))
        assert np.sign(ate_dr) == expected_sign, (
            f"DR-learner ATE sign mismatch: ate_hat={ate_dr:.3f}, "
            f"ate_true={ate_true:.3f}"
        )

    def test_invalid_estimator_raises(self, simple_dgp: UpliftDataset) -> None:
        with pytest.raises(ValueError, match="Unknown estimator"):
            fit_uplift(
                simple_dgp.X, simple_dgp.T, simple_dgp.Y,
                estimator="INVALID", n_splits=_N_SPLITS, seed=_SEED,
            )
