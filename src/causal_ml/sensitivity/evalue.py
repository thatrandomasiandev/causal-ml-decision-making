"""E-value sensitivity analysis (VanderWeele & Ding, 2017).

Provides functions to quantify the minimum strength of unmeasured
confounding required to explain away an observed treatment effect.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats as _stats

logger = logging.getLogger(__name__)


def _evalue_from_rr(rr: float) -> float:
    """Core E-value formula: E = RR + √(RR(RR − 1)).

    Args:
        rr: Risk ratio (must be ≥ 1 after orienting).

    Returns:
        E-value on the risk-ratio scale.
    """
    if rr <= 1.0:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def evalue(ate: float, ate_se: float, null: float = 0.0) -> dict[str, float | str]:
    """Compute the E-value for an estimated treatment effect.

    Uses a conservative log-risk-ratio conversion for continuous outcomes:
    RR ≈ exp(0.91 · |ATE − null| / SE).

    Args:
        ate: Point estimate of the treatment effect.
        ate_se: Standard error of the estimate.
        null: Null hypothesis value (default 0).

    Returns:
        Dictionary with keys ``evalue``, ``evalue_lower``, and
        ``interpretation``.
    """
    if ate_se <= 0:
        raise ValueError("ate_se must be positive.")

    effect = abs(ate - null)
    z = effect / ate_se

    def _evalue_from_z(z_val: float) -> float:
        rr = float(np.exp(0.91 * z_val))
        return _evalue_from_rr(rr)

    ev = _evalue_from_z(z)
    ev_lower = _evalue_from_z(max(z - 1.96, 0.0))

    if ev >= 2.0:
        interp = (
            f"An unmeasured confounder would need E-value {ev:.2f} "
            f"(lower CI bound {ev_lower:.2f}) to explain away the effect."
        )
    else:
        interp = (
            f"E-value {ev:.2f} indicates limited robustness to unmeasured confounding."
        )

    return {
        "evalue": ev,
        "evalue_lower": ev_lower,
        "interpretation": interp,
    }


def evalue_ci(
    ate: float,
    ate_se: float,
    null: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """E-value for the confidence interval bound closest to the null.

    Computes the E-value at the CI limit, answering: "How strong must
    unmeasured confounding be to shift the CI to include the null?"

    The CI bound is: |ATE − null| − z_{α/2} · SE, converted via
    RR ≈ exp(0.91 · z_bound).

    Args:
        ate: Point estimate of the treatment effect.
        ate_se: Standard error of the estimate.
        null: Null hypothesis value.
        confidence: Confidence level (e.g. 0.95 for a 95 % CI).

    Returns:
        Dictionary with ``evalue_ci`` and ``ci_bound_z``.
    """
    if ate_se <= 0:
        raise ValueError("ate_se must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    z_crit = float(np.abs(_stats.norm.ppf((1.0 - confidence) / 2.0)))
    effect = abs(ate - null)
    z = effect / ate_se
    z_bound = max(z - z_crit, 0.0)

    rr_bound = float(np.exp(0.91 * z_bound))
    ev_ci = _evalue_from_rr(rr_bound)

    return {
        "evalue_ci": ev_ci,
        "ci_bound_z": z_bound,
    }


def evalue_rr(rr: float) -> dict[str, float | str]:
    """E-value directly from a risk ratio (no conversion needed).

    E = RR + √(RR · (RR − 1)) for RR ≥ 1.  If RR < 1 the reciprocal
    is used so the E-value is always ≥ 1.

    Args:
        rr: Observed risk ratio (must be positive).

    Returns:
        Dictionary with ``evalue``, ``rr_used`` (oriented ≥ 1), and
        ``interpretation``.
    """
    if rr <= 0:
        raise ValueError("Risk ratio must be positive.")

    rr_oriented = rr if rr >= 1.0 else 1.0 / rr
    ev = _evalue_from_rr(rr_oriented)

    if ev >= 2.0:
        interp = (
            f"E-value {ev:.2f}: an unmeasured confounder would need associations "
            f"of at least {ev:.2f} with both the treatment and the outcome "
            "to explain away the observed RR."
        )
    else:
        interp = (
            f"E-value {ev:.2f}: limited robustness to unmeasured confounding."
        )

    return {
        "evalue": ev,
        "rr_used": rr_oriented,
        "interpretation": interp,
    }
