"""Rosenbaum sensitivity bounds for hidden confounding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rosenbaum_bounds(
    Y: np.ndarray,
    T: np.ndarray,
    gamma_range: list[float],
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity analysis for a binary treatment.

    For each sensitivity parameter gamma (odds ratio bound on hidden bias),
    computes upper and lower bounds on the Wilcoxon rank-sum p-value.

    Parameters
    ----------
    Y : array-like
        Outcome values.
    T : array-like
        Binary treatment indicator.
    gamma_range : list of float
        Values of Gamma >= 1 to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: gamma, p_lower, p_upper, significant_at_05
    """
    Y = np.asarray(Y, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()

    y1 = Y[T == 1]
    y0 = Y[T == 0]
    n1, n0 = len(y1), len(y0)
    if n1 == 0 or n0 == 0:
        raise ValueError("Both treatment arms must have at least one observation.")

    combined = np.concatenate([y0, y1])
    ranks = stats.rankdata(combined)
    ranks0 = ranks[:n0]
    ranks1 = ranks[n0:]
    W = ranks1.sum()

    rows = []
    for gamma in gamma_range:
        gamma = max(float(gamma), 1.0)
        # Rosenbaum (2002) bounds on Wilcoxon statistic under Gamma
        mu0 = n1 * (n0 + n1 + 1) / 2.0
        var0 = n1 * n0 * (n0 + n1 + 1) / 12.0
        z_obs = (W - mu0) / np.sqrt(var0) if var0 > 0 else 0.0

        # Gamma inflates/deflates the effective variance of the statistic
        z_upper = z_obs / np.sqrt(gamma)
        z_lower = z_obs * np.sqrt(gamma)

        p_upper = float(2 * stats.norm.sf(abs(z_upper)))
        p_lower = float(min(1.0, 2 * stats.norm.sf(abs(z_lower))))

        rows.append(
            {
                "gamma": gamma,
                "p_lower": p_lower,
                "p_upper": p_upper,
                "significant_at_05": p_lower < 0.05,
            }
        )

    return pd.DataFrame(rows)
