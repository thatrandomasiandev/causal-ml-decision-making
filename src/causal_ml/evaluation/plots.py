"""Diagnostic plots for causal ML evaluation.

Provides publication-ready matplotlib figures for inspecting uplift model
quality, CATE estimation accuracy, and propensity score overlap.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_uplift_curve(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    *,
    title: str = "Uplift Curve (AUUC)",
    figsize: tuple[float, float] = (7, 5),
) -> Figure:
    """Plot the cumulative uplift curve with oracle and random baselines.

    Sorts units by descending predicted CATE and plots the cumulative mean of
    the true effect as a function of the treated fraction.

    AUUC_model = integral of (cum_gain_model - random_baseline) df

    Args:
        tau_hat: Predicted CATE of shape ``(n,)``.
        tau_true: Ground-truth CATE of shape ``(n,)``.
        title: Plot title.
        figsize: Figure dimensions ``(width, height)`` in inches.

    Returns:
        Matplotlib ``Figure`` with the uplift curve.
    """
    n = len(tau_hat)
    fractions = np.linspace(0, 1, n)

    order_model = np.argsort(-tau_hat)
    cum_model = np.cumsum(tau_true[order_model]) / np.arange(1, n + 1)

    order_oracle = np.argsort(-tau_true)
    cum_oracle = np.cumsum(tau_true[order_oracle]) / np.arange(1, n + 1)

    random_baseline = np.full(n, np.mean(tau_true))

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.plot(fractions, cum_oracle, label="Oracle", linewidth=2, color="#2ca02c")
    ax.plot(fractions, cum_model, label="Model", linewidth=2, color="#1f77b4")
    ax.plot(
        fractions,
        random_baseline,
        label="Random",
        linewidth=1.5,
        linestyle="--",
        color="#7f7f7f",
    )
    ax.fill_between(
        fractions, random_baseline, cum_model, alpha=0.15, color="#1f77b4"
    )
    ax.set_xlabel("Fraction treated")
    ax.set_ylabel("Cumulative mean uplift")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    logger.debug("Uplift curve plotted with n=%d observations", n)
    return fig


def plot_cate_scatter(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    *,
    title: str = "CATE: Predicted vs True",
    figsize: tuple[float, float] = (6, 6),
) -> Figure:
    """Scatter plot of predicted versus true CATE with R-squared annotation.

    R^2 = 1 - SS_res / SS_tot

    Args:
        tau_hat: Predicted CATE of shape ``(n,)``.
        tau_true: Ground-truth CATE of shape ``(n,)``.
        title: Plot title.
        figsize: Figure dimensions ``(width, height)`` in inches.

    Returns:
        Matplotlib ``Figure`` with the scatter and 45-degree reference line.
    """
    ss_res = np.sum((tau_true - tau_hat) ** 2)
    ss_tot = np.sum((tau_true - np.mean(tau_true)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.scatter(tau_true, tau_hat, alpha=0.4, s=12, edgecolors="none", color="#1f77b4")

    lo = min(tau_true.min(), tau_hat.min())
    hi = max(tau_true.max(), tau_hat.max())
    margin = 0.05 * (hi - lo)
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
        label="y = x",
    )
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    ax.annotate(
        f"$R^2 = {r_squared:.3f}$",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
    )

    ax.set_xlabel(r"$\tau_{\mathrm{true}}$")
    ax.set_ylabel(r"$\hat{\tau}$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    logger.debug("CATE scatter plotted, R^2=%.4f", r_squared)
    return fig


def plot_propensity_overlap(
    propensity: np.ndarray,
    T: np.ndarray,
    *,
    n_bins: int = 40,
    title: str = "Propensity Score Overlap",
    figsize: tuple[float, float] = (7, 5),
) -> Figure:
    """Overlapping histograms of propensity scores by treatment arm.

    Visually diagnoses positivity violations — regions where treated and
    control propensity distributions do not overlap suggest extrapolation.

    Args:
        propensity: Estimated propensity scores of shape ``(n,)``.
        T: Binary treatment indicator of shape ``(n,)``.
        n_bins: Number of histogram bins.
        title: Plot title.
        figsize: Figure dimensions ``(width, height)`` in inches.

    Returns:
        Matplotlib ``Figure`` with overlapping histograms.
    """
    treated_mask = T.astype(bool)
    control_mask = ~treated_mask

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    bins = np.linspace(0, 1, n_bins + 1)
    ax.hist(
        propensity[treated_mask],
        bins=bins,
        alpha=0.55,
        label=f"Treated (n={int(treated_mask.sum())})",
        color="#d62728",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        propensity[control_mask],
        bins=bins,
        alpha=0.55,
        label=f"Control (n={int(control_mask.sum())})",
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axvline(x=0.5, linestyle=":", linewidth=1, color="#7f7f7f", alpha=0.6)
    ax.set_xlabel("Propensity score  e(x)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper center", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    logger.debug(
        "Propensity overlap plotted: treated=%d, control=%d",
        int(treated_mask.sum()),
        int(control_mask.sum()),
    )
    return fig
