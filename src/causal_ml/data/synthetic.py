"""
Controlled failure-mode synthetic DGP for CATE estimator benchmarks.

DGP specification
-----------------
Covariates:
    X ~ N(0, I_p)

Propensity (overlap-controlled):
    alpha = overlap_to_alpha(overlap)
    e(x) = sigmoid(alpha * X[:, 0])

Confounding (treatment assignment influenced by outcome-relevant covariates):
    score = confounding * (0.5 * X[:, 1] + 0.3 * X[:, 2])
    e_conf(x) = sigmoid(logit(e(x)) + score)
    T ~ Bernoulli(e_conf(x))

Nonlinear CATE component:
    f(x) = sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]

Heterogeneous treatment effect:
    ATE = 1.0
    tau(x) = heterogeneity * f(x) + (1 - heterogeneity) * ATE

Outcome model:
    mu(x) = 0.3 * X[:, 0] - 0.2 * X[:, 1] + confounding * 0.4 * X[:, 2]
    Y = mu(x) + T * tau(x) + eps,  eps ~ N(0, noise^2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_ml.data.base import UpliftDataset
from causal_ml.utils.seed import set_seed


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def overlap_to_alpha(overlap: float) -> float:
    """
    Map overlap in (0, 1] to propensity steepness.

    overlap=1.0 -> alpha≈0 (e(x)≈0.5), overlap=0.1 -> large alpha (extreme propensities).
    """
    overlap = float(np.clip(overlap, 0.05, 1.0))
    return float(10.0 * (1.0 - overlap))


@dataclass
class FailureModeDGP:
    """
    Generate synthetic datasets along axes of causal inference difficulty.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of covariates.
    overlap : float
        1.0 = full overlap, 0.1 = near positivity violation.
    confounding : float
        0.0 = unconfounded, 1.0 = heavy confounding.
    heterogeneity : float
        0.0 = homogeneous ATE, 1.0 = fully heterogeneous CATE.
    noise : float
        Outcome noise standard deviation.
    seed : int
        Random seed for reproducibility.
    """

    n: int = 2000
    p: int = 20
    overlap: float = 1.0
    confounding: float = 0.0
    heterogeneity: float = 0.5
    noise: float = 1.0
    seed: int = 42

    def generate(self) -> UpliftDataset:
        """Sample a dataset with known ground-truth CATE."""
        rng = set_seed(self.seed)
        n, p = self.n, self.p

        X = rng.standard_normal((n, p))
        alpha = overlap_to_alpha(self.overlap)
        base_propensity = _sigmoid(alpha * X[:, 0])

        conf_score = self.confounding * (0.5 * X[:, 1] + 0.3 * X[:, 2])
        propensity = _sigmoid(_logit(base_propensity) + conf_score)
        propensity = np.clip(propensity, 0.01, 0.99)

        f_x = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
        ate = 1.0
        tau = self.heterogeneity * f_x + (1.0 - self.heterogeneity) * ate

        mu = 0.3 * X[:, 0] - 0.2 * X[:, 1] + self.confounding * 0.4 * X[:, 2]
        eps = rng.normal(0, self.noise, size=n)

        T = rng.binomial(1, propensity).astype(float)
        Y = mu + T * tau + eps

        y0 = mu + eps
        y1 = mu + tau + eps

        return UpliftDataset(
            X=X,
            T=T,
            Y=Y,
            metadata={
                "dgp": "failure_mode",
                "n": n,
                "p": p,
                "overlap": self.overlap,
                "confounding": self.confounding,
                "heterogeneity": self.heterogeneity,
                "noise": self.noise,
                "seed": self.seed,
            },
            ground_truth={
                "tau": tau,
                "ate": float(np.mean(tau)),
                "propensity": propensity,
                "y0": y0,
                "y1": y1,
                "mu": mu,
            },
        )


def generate_failure_mode_data(
    n: int = 2000,
    p: int = 20,
    overlap: float = 1.0,
    confounding: float = 0.0,
    heterogeneity: float = 0.5,
    noise: float = 1.0,
    seed: int = 42,
) -> UpliftDataset:
    """Functional API for FailureModeDGP."""
    return FailureModeDGP(
        n=n,
        p=p,
        overlap=overlap,
        confounding=confounding,
        heterogeneity=heterogeneity,
        noise=noise,
        seed=seed,
    ).generate()
