"""PCMCI time-series causal discovery via tigramite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCMCIResult:
    adjacency: np.ndarray
    p_matrix: np.ndarray | None = None


def pcmci(
    X: np.ndarray,
    tau_max: int = 2,
    pc_alpha: float = 0.05,
    seed: int = 42,
) -> PCMCIResult:
    """
    Run PCMCI on multivariate time series.

    Returns aggregated adjacency (max abs coefficient across lags) of shape (d, d).
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError as exc:
        raise ImportError("tigramite is required for PCMCI. Install with: pip install tigramite") from exc

    dataframe = pp.DataFrame(X, datatime=np.arange(len(X)))
    parcorr = ParCorr(significance="analytic")
    pcmci_obj = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    results = pcmci_obj.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]

    d = X.shape[1]
    adj = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                adj[i, j] = np.max(np.abs(val_matrix[i, j, :]))

    return PCMCIResult(adjacency=adj, p_matrix=p_matrix)
