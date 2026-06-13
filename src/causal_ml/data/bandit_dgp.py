"""Logged bandit DGP built on uplift potential outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_ml.data.base import BanditDataset
from causal_ml.data.uplift_dgp import UpliftDGPConfig, generate_uplift_data


@dataclass
class BanditDGPConfig:
    n_samples: int = 5000
    n_features: int = 10
    confounding_strength: float = 1.0
    noise_std: float = 0.5
    epsilon: float = 0.1
    logging_bias: float = 0.5
    seed: int = 42


def _behavior_policy(
    tau: np.ndarray,
    epsilon: float,
    logging_bias: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Epsilon-greedy behavior policy biased toward high-tau actions.

    pi_b(a=1|x) blends uniform exploration with preference for positive tau.
    """
    oracle = (tau > 0).astype(float)
    greedy_prob = _sigmoid(logging_bias * tau)
    pi1 = (1 - epsilon) * greedy_prob + epsilon * 0.5
    pi1 = np.clip(pi1, 0.01, 0.99)
    actions = rng.binomial(1, pi1).astype(float)
    propensities = np.where(actions == 1, pi1, 1 - pi1)
    return actions, propensities


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def policy_value(actions: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> float:
    """Compute true policy value given action assignments."""
    rewards = actions * y1 + (1 - actions) * y0
    return float(np.mean(rewards))


def generate_bandit_data(config: BanditDGPConfig | None = None) -> BanditDataset:
    """Generate logged bandit data with known oracle policy and counterfactuals."""
    cfg = config or BanditDGPConfig()
    uplift = generate_uplift_data(
        UpliftDGPConfig(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            confounding_strength=cfg.confounding_strength,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
        )
    )

    rng = np.random.default_rng(cfg.seed + 1)
    tau = uplift.ground_truth["tau"]
    y0 = uplift.ground_truth["y0"]
    y1 = uplift.ground_truth["y1"]

    actions, propensities = _behavior_policy(tau, cfg.epsilon, cfg.logging_bias, rng)
    rewards = actions * y1 + (1 - actions) * y0

    oracle_actions = (tau > 0).astype(float)
    oracle_value = policy_value(oracle_actions, y0, y1)
    behavior_value = policy_value(actions, y0, y1)

    return BanditDataset(
        X=uplift.X,
        actions=actions,
        rewards=rewards,
        propensities=propensities,
        metadata={
            "dgp": "bandit",
            "epsilon": cfg.epsilon,
            "logging_bias": cfg.logging_bias,
            **uplift.metadata,
        },
        ground_truth={
            "tau": tau,
            "y0": y0,
            "y1": y1,
            "oracle_actions": oracle_actions,
            "oracle_value": oracle_value,
            "behavior_value": behavior_value,
            "propensity_obs": uplift.ground_truth["propensity"],
        },
    )
