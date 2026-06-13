"""Failure-mode benchmark grid execution and reporting."""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from causal_ml.data.synthetic import FailureModeDGP
from causal_ml.uplift.doubly_robust import dr_learner
from causal_ml.uplift.meta_learners import ESTIMATORS, fit_uplift
from causal_ml.uplift.metrics import ate_error, auuc, pehe, qini


def load_grid_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    default_path = config_path.parent / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            defaults = yaml.safe_load(f)
        cfg = {**defaults, **cfg}
    return cfg


def _run_single_cell(
    overlap: float,
    confounding: float,
    heterogeneity: float,
    n: int,
    estimator: str,
    seed: int,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Execute one grid cell."""
    t0 = time.perf_counter()
    dgp = FailureModeDGP(
        n=n,
        overlap=overlap,
        confounding=confounding,
        heterogeneity=heterogeneity,
        seed=seed,
    )
    data = dgp.generate()
    tau_true = data.ground_truth["tau"]

    if estimator == "DR":
        result = dr_learner(data.X, data.T, data.Y, n_splits=n_splits, seed=seed)
    else:
        result = fit_uplift(data.X, data.T, data.Y, estimator=estimator, n_splits=n_splits, seed=seed)

    runtime = time.perf_counter() - t0

    return {
        "overlap": overlap,
        "confounding": confounding,
        "heterogeneity": heterogeneity,
        "n": n,
        "estimator": estimator,
        "seed": seed,
        "pehe": pehe(result.tau_hat, tau_true),
        "ate_error": ate_error(result.tau_hat, tau_true),
        "auuc": auuc(result.tau_hat, tau_true),
        "qini": qini(result.tau_hat, tau_true),
        "runtime_sec": runtime,
    }


def build_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand config into list of parameter dicts."""
    keys = ["overlap", "confounding", "heterogeneity", "n", "estimators", "seeds"]
    values = [config[k] for k in keys]
    cells = []
    for combo in product(*values):
        overlap, confounding, heterogeneity, n, estimator, seed = combo
        cells.append(
            {
                "overlap": overlap,
                "confounding": confounding,
                "heterogeneity": heterogeneity,
                "n": n,
                "estimator": estimator,
                "seed": seed,
            }
        )
    return cells


def run_failure_mode_grid(
    config: dict[str, Any],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run full grid and return results DataFrame."""
    n_splits = config.get("n_splits", 3)
    cells = build_grid(config)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_run_single_cell)(
            cell["overlap"],
            cell["confounding"],
            cell["heterogeneity"],
            cell["n"],
            cell["estimator"],
            cell["seed"],
            n_splits,
        )
        for cell in cells
    )
    return pd.DataFrame(results)


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean ± std over seeds per grid cell."""
    group_cols = ["overlap", "confounding", "heterogeneity", "n", "estimator"]
    metrics = ["pehe", "ate_error", "auuc", "qini", "runtime_sec"]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in agg.columns
    ]
    return agg


def surprising_failures(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Find cells where DR-learner PEHE exceeds T-learner PEHE.

    Sorted by margin (DR - T) descending.
    """
    group_cols = ["overlap", "confounding", "heterogeneity", "n", "seed"]
    pivot = df.pivot_table(
        index=group_cols,
        columns="estimator",
        values="pehe",
        aggfunc="mean",
    )
    if "DR" not in pivot.columns or "T" not in pivot.columns:
        return pd.DataFrame()

    pivot = pivot.reset_index()
    pivot["margin"] = pivot["DR"] - pivot["T"]
    surprising = pivot[pivot["margin"] > 0].sort_values("margin", ascending=False).head(top_k)
    return surprising


def write_report(df: pd.DataFrame, summary: pd.DataFrame, path: Path, top_k: int = 10) -> None:
    """Write markdown report with surprising failure table."""
    lines = [
        "# Failure Mode Benchmark Report",
        "",
        f"Total runs: {len(df)}",
        "",
        "## Top Surprising Failure Cells (DR PEHE > T PEHE)",
        "",
    ]
    top = surprising_failures(df, top_k=top_k)
    if top.empty:
        lines.append("_No cells where DR underperformed T._")
    else:
        lines.append("| overlap | confounding | heterogeneity | n | seed | T PEHE | DR PEHE | margin |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['overlap']} | {row['confounding']} | {row['heterogeneity']} | "
                f"{int(row['n'])} | {int(row['seed'])} | {row['T']:.4f} | {row['DR']:.4f} | "
                f"{row['margin']:.4f} |"
            )
    lines.append("")
    lines.append("## Summary Statistics (first 20 rows)")
    lines.append("")
    head = summary.head(20)
    lines.append("| " + " | ".join(head.columns.astype(str)) + " |")
    lines.append("| " + " | ".join(["---"] * len(head.columns)) + " |")
    for _, row in head.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines))


def run_and_save(
    config_path: Path,
    output_dir: Path,
    n_jobs: int = 1,
    top_k: int = 10,
) -> Path:
    """Run grid, save parquet/csv/report, return run directory."""
    from datetime import datetime, timezone

    config = load_grid_config(config_path)
    df = run_failure_mode_grid(config, n_jobs=n_jobs)
    summary = aggregate_summary(df)

    run_dir = output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(run_dir / "results.parquet", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    write_report(df, summary, run_dir / "report.md", top_k=top_k)

    return run_dir
