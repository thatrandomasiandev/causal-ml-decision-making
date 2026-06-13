"""Smoke test for failure-mode benchmark grid."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_ml.data.paths import project_root
from causal_ml.evaluation.failure_mode import (
    aggregate_summary,
    build_grid,
    run_failure_mode_grid,
    run_and_save,
)


def test_mini_grid_schema():
    config_path = project_root() / "configs" / "failure_mode_mini.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cells = build_grid(config)
    assert len(cells) == 2 * 2 * 2 * 1 * 2 * 2  # 32 runs

    df = run_failure_mode_grid(config, n_jobs=1)
    assert len(df) == 32
    expected_cols = {
        "overlap",
        "confounding",
        "heterogeneity",
        "n",
        "estimator",
        "seed",
        "pehe",
        "ate_error",
        "auuc",
        "qini",
        "runtime_sec",
    }
    assert expected_cols.issubset(set(df.columns))
    assert not df[list(expected_cols - {"estimator"})].isna().any().any()


def test_run_and_save_mini_grid(tmp_path: Path):
    config_path = project_root() / "configs" / "failure_mode_mini.yaml"
    run_dir = run_and_save(config_path, tmp_path, n_jobs=1, top_k=5)
    assert (run_dir / "results.parquet").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "report.md").exists()
    df = pd.read_parquet(run_dir / "results.parquet")
    assert len(df) > 0
    assert not df["pehe"].isna().any()
