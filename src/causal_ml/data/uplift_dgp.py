"""Synthetic uplift DGP with confounding and heterogeneous treatment effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_ml.data.base import UpliftDataset
from causal_ml.utils.seed import set_seed


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class UpliftDGPConfig:
    n_samples: int = 5000
    n_features: int = 10
    confounding_strength: float = 1.0
    noise_std: float = 0.5
    seed: int = 42


def generate_uplift_data(config: UpliftDGPConfig | None = None) -> UpliftDataset:
    """
    Generate observational data with known potential outcomes.

    DGP:
      X ~ N(0, I)
      e(x) = sigmoid(confounding_strength * (w·x))
      tau(x) = x0 + 0.5 * x1 * x2
      Y(0) = 0.5*x0 - 0.3*x1 + noise
      Y(1) = Y(0) + tau(x)
      T ~ Bernoulli(e(x)), Y = T*Y(1) + (1-T)*Y(0)
    """
    cfg = config or UpliftDGPConfig()
    rng = set_seed(cfg.seed)

    n, d = cfg.n_samples, cfg.n_features
    X = rng.standard_normal((n, d))

    w = rng.standard_normal(d)
    w = w / (np.linalg.norm(w) + 1e-8)
    logits = cfg.confounding_strength * (X @ w)
    propensity = _sigmoid(logits)

    tau = X[:, 0] + 0.5 * X[:, 1] * X[:, 2]
    y0 = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(0, cfg.noise_std, size=n)
    y1 = y0 + tau

    T = rng.binomial(1, propensity).astype(float)
    Y = T * y1 + (1 - T) * y0

    return UpliftDataset(
        X=X,
        T=T,
        Y=Y,
        metadata={
            "dgp": "uplift",
            "n_samples": n,
            "n_features": d,
            "confounding_strength": cfg.confounding_strength,
            "noise_std": cfg.noise_std,
            "seed": cfg.seed,
        },
        ground_truth={
            "tau": tau,
            "ate": float(np.mean(tau)),
            "propensity": propensity,
            "y0": y0,
            "y1": y1,
            "propensity_weights": w,
        },
    )
