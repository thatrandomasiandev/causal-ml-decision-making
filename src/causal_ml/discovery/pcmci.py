"""PCMCI time-series causal discovery via tigramite.

Provides a pure-Python fallback when tigramite is not installed, using
partial-correlation-based PC skeleton discovery and Momentary Conditional
Independence (MCI) testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_HAS_TIGRAMITE = True
try:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI as _TigramitePCMCI
except ImportError:
    _HAS_TIGRAMITE = False


@dataclass
class PCMCIResult:
    """Container for PCMCI causal discovery results.

    Attributes:
        adjacency: Aggregated adjacency matrix of shape (d, d).
        p_matrix: Matrix of p-values from independence tests, or ``None``.
    """

    adjacency: np.ndarray
    p_matrix: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Pure-Python fallback helpers
# ---------------------------------------------------------------------------


def _partial_corr(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
) -> tuple[float, float]:
    """Partial correlation between *x* and *y* given *z* via OLS residuals.

    Computes corr(x, y | Z) = corr(ε_x, ε_y) where ε_x, ε_y are
    residuals from regressing x and y on Z.

    Args:
        x: 1-D array of length n.
        y: 1-D array of length n.
        z: Conditioning matrix of shape (n, k) or ``None``.

    Returns:
        Tuple of (partial correlation, two-sided p-value).
    """
    if z is None or z.shape[1] == 0:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)

    z_aug = np.column_stack([z, np.ones(len(x))])
    beta_x, *_ = np.linalg.lstsq(z_aug, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(z_aug, y, rcond=None)
    res_x = x - z_aug @ beta_x
    res_y = y - z_aug @ beta_y

    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        return 0.0, 1.0

    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def _benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Rejects H_0 for p_{(k)} ≤ (k / m) · α.

    Args:
        p_values: 1-D array of p-values.
        alpha: Target false discovery rate.

    Returns:
        Boolean array indicating which hypotheses are rejected.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    thresholds = alpha * np.arange(1, m + 1) / m
    max_k = 0
    for k in range(m):
        if sorted_p[k] <= thresholds[k]:
            max_k = k + 1

    rejected = np.zeros(m, dtype=bool)
    if max_k > 0:
        rejected[sorted_idx[:max_k]] = True
    return rejected


def _build_lagged_data(
    X: np.ndarray,
    tau_max: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Create a design matrix of lagged variables.

    Args:
        X: Time series of shape (T, d).
        tau_max: Maximum lag to include.

    Returns:
        Tuple of (lagged data matrix of shape (T - tau_max, d * (tau_max + 1)),
        list of (variable_index, lag) labels).
    """
    T, d = X.shape
    cols: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for tau in range(tau_max + 1):
        start = tau_max - tau
        end = T - tau if tau > 0 else T
        cols.append(X[start:end])
        for j in range(d):
            labels.append((j, tau))
    return np.hstack(cols), labels


def _pc_skeleton(
    data: np.ndarray,
    labels: list[tuple[int, int]],
    alpha: float,
    d: int,
    tau_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """PC algorithm skeleton phase with partial-correlation tests.

    Iteratively removes edges whose endpoints are conditionally independent
    given subsets of their shared neighbours, with BH FDR correction.

    Args:
        data: Lagged data matrix of shape (n_eff, n_vars).
        labels: Variable-lag labels for each column.
        alpha: Significance level for the conditional independence tests.
        d: Number of observed variables.
        tau_max: Maximum lag.

    Returns:
        Tuple of (adjacency matrix of shape (d, d),
        p-value matrix of shape (d, d)).
    """
    n_vars = data.shape[1]
    adj = np.ones((n_vars, n_vars), dtype=bool)
    np.fill_diagonal(adj, False)
    p_store = np.ones((n_vars, n_vars))

    cond_size = 0
    while True:
        any_removed = False
        all_pvals: list[float] = []
        test_coords: list[tuple[int, int]] = []

        for i in range(n_vars):
            neighbours = np.where(adj[i])[0]
            if len(neighbours) <= cond_size:
                continue
            for j in neighbours:
                other = [k for k in neighbours if k != j]
                if len(other) < cond_size:
                    continue
                best_p = 0.0
                for subset in combinations(other, cond_size):
                    z = data[:, list(subset)] if subset else None
                    _, p = _partial_corr(data[:, i], data[:, j], z)
                    if p > best_p:
                        best_p = p
                all_pvals.append(best_p)
                test_coords.append((i, j))

        if not all_pvals:
            break

        pvals_arr = np.array(all_pvals)
        rejected = _benjamini_hochberg(pvals_arr, alpha)
        for idx, (i, j) in enumerate(test_coords):
            p_store[i, j] = all_pvals[idx]
            if not rejected[idx]:
                adj[i, j] = False
                any_removed = True

        cond_size += 1
        max_neighbours = max(np.sum(adj[i]) for i in range(n_vars))
        if cond_size > max_neighbours or not any_removed:
            break

    adj_out = np.zeros((d, d))
    p_out = np.ones((d, d))
    contemp_indices = {(var, 0): col for col, (var, lag) in enumerate(labels) if lag == 0}
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            ci = contemp_indices.get((i, 0))
            cj = contemp_indices.get((j, 0))
            if ci is not None and cj is not None and adj[ci, cj]:
                adj_out[i, j] = 1.0
                p_out[i, j] = p_store[ci, cj]

    return adj_out, p_out


def _mci_test(
    X: np.ndarray,
    skeleton: np.ndarray,
    tau_max: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Momentary Conditional Independence (MCI) test.

    For each candidate link i→j at lag τ, tests independence conditioning
    on the parents of both i (at lag τ) and j (contemporaneous), identified
    from the skeleton.  Refines links via BH FDR correction.

    MCI statistic: corr(X_t^j, X_{t-τ}^i | Parents(X_t^j), Parents(X_{t-τ}^i)).

    Args:
        X: Time series of shape (T, d).
        skeleton: Binary adjacency from skeleton phase, shape (d, d).
        tau_max: Maximum lag.
        alpha: Significance level.

    Returns:
        Tuple of (value matrix of shape (d, d, tau_max + 1),
        p-value matrix of same shape).
    """
    T, d = X.shape
    val_matrix = np.zeros((d, d, tau_max + 1))
    p_matrix = np.ones((d, d, tau_max + 1))

    all_pvals: list[float] = []
    test_coords: list[tuple[int, int, int]] = []

    for tau in range(tau_max + 1):
        n_eff = T - tau_max
        for i in range(d):
            for j in range(d):
                if i == j and tau == 0:
                    continue

                target = X[tau_max:, j]
                source = X[tau_max - tau : T - tau if tau > 0 else T, i]

                cond_cols: list[np.ndarray] = []
                parents_j = np.where(skeleton[:, j] > 0)[0]
                for p in parents_j:
                    if p != i:
                        cond_cols.append(X[tau_max:, p])
                parents_i = np.where(skeleton[:, i] > 0)[0]
                for p in parents_i:
                    lag_offset = tau
                    start = tau_max - lag_offset
                    end = T - lag_offset if lag_offset > 0 else T
                    if end - start == n_eff:
                        cond_cols.append(X[start:end, p])

                z = np.column_stack(cond_cols) if cond_cols else None
                r, p = _partial_corr(target, source, z)

                val_matrix[i, j, tau] = r
                p_matrix[i, j, tau] = p
                all_pvals.append(p)
                test_coords.append((i, j, tau))

    if all_pvals:
        pvals_arr = np.array(all_pvals)
        rejected = _benjamini_hochberg(pvals_arr, alpha)
        for idx, (i, j, tau) in enumerate(test_coords):
            if not rejected[idx]:
                val_matrix[i, j, tau] = 0.0

    return val_matrix, p_matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pcmci(
    X: np.ndarray,
    tau_max: int = 2,
    pc_alpha: float = 0.05,
    seed: int = 42,
) -> PCMCIResult:
    """Run PCMCI on multivariate time series.

    Uses tigramite when available; otherwise falls back to a pure-Python
    implementation using partial-correlation PC skeleton + MCI tests with
    Benjamini-Hochberg FDR correction.

    Returns aggregated adjacency (max abs coefficient across lags) of shape (d, d).

    Args:
        X: Multivariate time series of shape (T, d).
        tau_max: Maximum lag to consider.
        pc_alpha: Significance level for conditional independence tests.
        seed: Random seed (used by tigramite backend only).

    Returns:
        PCMCIResult with adjacency and p-value matrices.
    """
    d = X.shape[1]

    if _HAS_TIGRAMITE:
        logger.debug("Running PCMCI with tigramite backend")
        dataframe = pp.DataFrame(X, datatime=np.arange(len(X)))
        parcorr = ParCorr(significance="analytic")
        pcmci_obj = _TigramitePCMCI(
            dataframe=dataframe, cond_ind_test=parcorr, verbosity=0,
        )
        results = pcmci_obj.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
        val_matrix = results["val_matrix"]
        p_matrix_raw = results["p_matrix"]

        adj = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    adj[i, j] = np.max(np.abs(val_matrix[i, j, :]))

        return PCMCIResult(adjacency=adj, p_matrix=p_matrix_raw)

    logger.info("tigramite not installed; using pure-Python PCMCI fallback")

    lagged_data, labels = _build_lagged_data(X, tau_max)
    skeleton, _ = _pc_skeleton(lagged_data, labels, pc_alpha, d, tau_max)
    val_matrix, p_matrix_raw = _mci_test(X, skeleton, tau_max, pc_alpha)

    adj = np.zeros((d, d))
    p_agg = np.ones((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                adj[i, j] = np.max(np.abs(val_matrix[i, j, :]))
                p_agg[i, j] = np.min(p_matrix_raw[i, j, :])

    return PCMCIResult(adjacency=adj, p_matrix=p_agg)
