"""IHDP semi-synthetic benchmark DGP (Hill 2011, Response Surface B).

The Infant Health and Development Program (IHDP) dataset is a widely used
benchmark in causal inference.  The original trial randomised low-birth-weight
premature infants into an intensive childcare programme, but six of the treated
units are dropped to induce observational confounding.

This module provides:

* ``IHDPDataset`` — a loader for the real IHDP covariate matrix with
  semi-synthetic response surfaces.
* ``generate_ihdp_like`` — a fully synthetic generator that mirrors the 25-
  covariate structure of IHDP without requiring the original data files,
  enabling reproducible CI on any machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from causal_ml.data.base import UpliftDataset
from causal_ml.utils.seed import set_seed

logger = logging.getLogger(__name__)

_N_COVARIATES = 25
_N_CONTINUOUS = 6
_N_BINARY = 19


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid: sigma(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _response_surface_b(
    X: np.ndarray,
    beta: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hill (2011) Response Surface B — nonlinear, heterogeneous.

    Y(0) = exp((X + 0.5) @ beta) + noise
    Y(1) = X @ beta + noise
    tau(x) = Y(1) - Y(0)

    Args:
        X: Covariate matrix of shape ``(n, d)``.
        beta: Coefficient vector of shape ``(d,)``.
        rng: NumPy random generator.
        noise_std: Standard deviation of Gaussian outcome noise.

    Returns:
        Tuple ``(y0, y1, tau)`` each of shape ``(n,)``.
    """
    n = X.shape[0]
    noise0 = rng.normal(0.0, noise_std, size=n)
    noise1 = rng.normal(0.0, noise_std, size=n)

    y0 = np.exp((X + 0.5) @ beta) + noise0
    y1 = X @ beta + noise1
    tau = y1 - y0 + noise1 - noise0
    tau_noiseless = (X @ beta) - np.exp((X + 0.5) @ beta)

    return y0, y1, tau_noiseless


@dataclass
class IHDPDataset:
    """Wrapper for loading IHDP covariate matrices with synthetic outcomes.

    The real IHDP covariates consist of 6 continuous and 19 binary features
    describing maternal and infant characteristics.  This class generates
    semi-synthetic potential outcomes via Hill's Response Surface B when the
    original covariate matrix is supplied.

    Attributes:
        n_realizations: Number of outcome realizations to generate.
        noise_std: Standard deviation of outcome noise.
        seed: Random seed for reproducibility.
    """

    n_realizations: int = 10
    noise_std: float = 1.0
    seed: int = 42

    def generate(self, X_real: np.ndarray) -> list[UpliftDataset]:
        """Generate multiple semi-synthetic outcome realizations over real covariates.

        Each realization draws fresh beta coefficients and noise, producing a
        different response surface while holding covariates fixed.

        Args:
            X_real: Real IHDP covariate matrix of shape ``(n, 25)``.

        Returns:
            List of ``UpliftDataset`` instances, one per realization.
        """
        if X_real.shape[1] != _N_COVARIATES:
            raise ValueError(
                f"Expected {_N_COVARIATES} covariates, got {X_real.shape[1]}."
            )

        rng = set_seed(self.seed)
        n = X_real.shape[0]
        datasets: list[UpliftDataset] = []

        for r in range(self.n_realizations):
            beta = rng.standard_normal(_N_COVARIATES)
            beta = beta / (np.linalg.norm(beta) + 1e-8)

            propensity = _sigmoid(0.75 * (X_real @ beta))
            propensity = np.clip(propensity, 0.05, 0.95)
            T = rng.binomial(1, propensity).astype(float)

            y0, y1, tau = _response_surface_b(
                X_real, beta, rng, self.noise_std
            )
            Y = T * y1 + (1.0 - T) * y0

            datasets.append(
                UpliftDataset(
                    X=X_real.copy(),
                    T=T,
                    Y=Y,
                    metadata={
                        "dgp": "ihdp_response_surface_b",
                        "realization": r,
                        "n_samples": n,
                        "n_features": _N_COVARIATES,
                        "noise_std": self.noise_std,
                        "seed": self.seed,
                    },
                    ground_truth={
                        "tau": tau,
                        "ate": float(np.mean(tau)),
                        "propensity": propensity,
                        "y0": y0,
                        "y1": y1,
                        "beta": beta,
                    },
                )
            )
            logger.debug("IHDP realization %d/%d generated", r + 1, self.n_realizations)

        return datasets


def generate_ihdp_like(
    n_samples: int = 747,
    noise_std: float = 1.0,
    confounding_strength: float = 0.75,
    seed: int = 42,
) -> UpliftDataset:
    """Generate fully synthetic data mimicking IHDP covariate structure.

    Produces a dataset with 25 covariates (6 continuous + 19 binary) that
    mirrors the statistical profile of the real IHDP data without requiring
    external files.  Outcomes follow Hill's Response Surface B.

    Y(0) = exp((X + 0.5) @ beta),  Y(1) = X @ beta,  tau = Y(1) - Y(0)

    Args:
        n_samples: Number of observations (real IHDP has 747).
        noise_std: Standard deviation of outcome noise.
        confounding_strength: Scale of confounding in the propensity model.
        seed: Random seed for full reproducibility.

    Returns:
        ``UpliftDataset`` with known ground-truth ``tau``, ``ate``,
        ``propensity``, ``y0``, and ``y1``.
    """
    rng = set_seed(seed)

    X_cont = rng.standard_normal((n_samples, _N_CONTINUOUS))
    X_cont[:, 0] = np.abs(X_cont[:, 0]) + 0.5
    X_cont[:, 1] = np.clip(X_cont[:, 1] * 5 + 25, 15, 45)

    binary_probs = rng.uniform(0.1, 0.7, size=_N_BINARY)
    X_bin = rng.binomial(1, binary_probs, size=(n_samples, _N_BINARY)).astype(
        float
    )

    X = np.column_stack([X_cont, X_bin])

    beta = rng.standard_normal(_N_COVARIATES)
    beta = beta / (np.linalg.norm(beta) + 1e-8)

    propensity = _sigmoid(confounding_strength * (X @ beta))
    propensity = np.clip(propensity, 0.05, 0.95)
    T = rng.binomial(1, propensity).astype(float)

    y0, y1, tau = _response_surface_b(X, beta, rng, noise_std)
    Y = T * y1 + (1.0 - T) * y0

    return UpliftDataset(
        X=X,
        T=T,
        Y=Y,
        metadata={
            "dgp": "ihdp_like_synthetic",
            "n_samples": n_samples,
            "n_features": _N_COVARIATES,
            "n_continuous": _N_CONTINUOUS,
            "n_binary": _N_BINARY,
            "noise_std": noise_std,
            "confounding_strength": confounding_strength,
            "seed": seed,
        },
        ground_truth={
            "tau": tau,
            "ate": float(np.mean(tau)),
            "propensity": propensity,
            "y0": y0,
            "y1": y1,
            "beta": beta,
        },
    )
