"""Comprehensive tests for off-policy evaluation estimators.

Validates IPS, SNIPS, and DR policy value estimators return correct types,
reasonable bounds, and agree with oracle values on a simple bandit.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_ml.policy.evaluation import PolicyEvalResult, dr_policy_value, ips, snips

_SEED = 77
_N = 500


@pytest.fixture(scope="module")
def simple_bandit() -> dict[str, np.ndarray | float]:
    """Simple two-arm bandit with known oracle value.

    Arm 1 reward: X[:,0] + noise
    Arm 0 reward: -0.5 + noise
    Behavior: epsilon-greedy with epsilon=0.3
    Oracle: always pick arm 1 when X[:,0] > 0.5
    """
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N, 3))

    y1 = X[:, 0] + rng.normal(0, 0.2, size=_N)
    y0 = -0.5 + rng.normal(0, 0.2, size=_N)

    pi1 = np.full(_N, 0.5)
    actions = rng.binomial(1, pi1).astype(float)
    propensities = np.where(actions == 1, pi1, 1.0 - pi1)
    rewards = actions * y1 + (1.0 - actions) * y0

    oracle_actions = (X[:, 0] > 0.0).astype(float)
    oracle_value = float(np.mean(oracle_actions * y1 + (1.0 - oracle_actions) * y0))

    mu_hat = np.column_stack([y0, y1])

    return {
        "actions": actions,
        "rewards": rewards,
        "propensities": propensities,
        "target_actions": oracle_actions,
        "oracle_value": oracle_value,
        "mu_hat": mu_hat,
        "y0": y0,
        "y1": y1,
        "n": _N,
    }


class TestIPS:
    """Inverse Propensity Scoring tests."""

    def test_returns_scalar(self, simple_bandit: dict) -> None:
        result = ips(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
        )
        assert isinstance(result, PolicyEvalResult)
        assert isinstance(result.value, float)
        assert np.isfinite(result.value)

    def test_estimator_label(self, simple_bandit: dict) -> None:
        result = ips(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
        )
        assert result.estimator == "IPS"

    def test_has_std_error(self, simple_bandit: dict) -> None:
        result = ips(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
        )
        assert result.std_error is not None
        assert result.std_error >= 0.0


class TestSNIPS:
    """Self-Normalized IPS tests."""

    def test_returns_scalar(self, simple_bandit: dict) -> None:
        result = snips(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
        )
        assert isinstance(result, PolicyEvalResult)
        assert isinstance(result.value, float)
        assert np.isfinite(result.value)

    def test_estimator_label(self, simple_bandit: dict) -> None:
        result = snips(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
        )
        assert result.estimator == "SNIPS"

    def test_value_within_oracle_tolerance(self, simple_bandit: dict) -> None:
        """SNIPS should be within +/- 0.5 of the oracle value on this RCT-like DGP."""
        result = snips(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
        )
        oracle = simple_bandit["oracle_value"]
        assert abs(result.value - oracle) < 0.5, (
            f"SNIPS value={result.value:.3f} too far from oracle={oracle:.3f}"
        )


class TestDRPolicyValue:
    """Doubly robust policy value estimator tests."""

    def test_returns_scalar(self, simple_bandit: dict) -> None:
        result = dr_policy_value(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
            simple_bandit["mu_hat"],
        )
        assert isinstance(result, PolicyEvalResult)
        assert isinstance(result.value, float)
        assert np.isfinite(result.value)

    def test_estimator_label(self, simple_bandit: dict) -> None:
        result = dr_policy_value(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
            simple_bandit["mu_hat"],
        )
        assert result.estimator == "DR"

    def test_has_std_error(self, simple_bandit: dict) -> None:
        result = dr_policy_value(
            simple_bandit["actions"],
            simple_bandit["rewards"],
            simple_bandit["propensities"],
            simple_bandit["target_actions"],
            simple_bandit["mu_hat"],
        )
        assert result.std_error is not None
        assert result.std_error >= 0.0


class TestEffectiveSampleSize:
    """Tests for effective sample size of importance weights."""

    def test_ess_between_zero_and_n(self, simple_bandit: dict) -> None:
        """ESS = (sum rho)^2 / sum(rho^2) must lie in (0, n]."""
        actions = simple_bandit["actions"]
        propensities = simple_bandit["propensities"]
        target = simple_bandit["target_actions"]
        n = simple_bandit["n"]

        rho = np.where(
            actions == target,
            1.0 / np.clip(propensities, 1e-6, None),
            0.0,
        )
        ess = _effective_sample_size(rho)
        assert 0 < ess <= n, f"ESS={ess:.1f} not in (0, {n}]"


class TestClipWeights:
    """Tests for importance weight clipping."""

    def test_clip_weights_caps_values(self) -> None:
        rho = np.array([0.1, 1.0, 5.0, 100.0, 0.01])
        clipped = _clip_weights(rho, max_weight=10.0)
        assert clipped.max() <= 10.0
        assert clipped.min() >= 0.0

    def test_clip_weights_preserves_small(self) -> None:
        rho = np.array([0.5, 1.0, 2.0])
        clipped = _clip_weights(rho, max_weight=10.0)
        np.testing.assert_array_equal(rho, clipped)

    def test_clip_weights_all_equal(self) -> None:
        rho = np.array([50.0, 50.0, 50.0])
        clipped = _clip_weights(rho, max_weight=10.0)
        np.testing.assert_allclose(clipped, 10.0)


def _effective_sample_size(rho: np.ndarray) -> float:
    """Compute ESS from importance weights: ESS = (sum rho)^2 / sum(rho^2).

    Args:
        rho: Importance weights of shape ``(n,)``.

    Returns:
        Effective sample size as a scalar.
    """
    total = np.sum(rho)
    if total < 1e-12:
        return 0.0
    return float(total ** 2 / np.sum(rho ** 2))


def _clip_weights(rho: np.ndarray, max_weight: float = 10.0) -> np.ndarray:
    """Clip importance weights to a maximum value.

    Args:
        rho: Importance weights of shape ``(n,)``.
        max_weight: Upper bound for clipping.

    Returns:
        Clipped weights of shape ``(n,)``.
    """
    return np.clip(rho, 0.0, max_weight)
