#!/usr/bin/env python3
"""Run failure-mode CATE estimator benchmark grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_ml.data.paths import project_root
from causal_ml.evaluation.failure_mode import run_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-mode CATE benchmark grid")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/failure_mode_benchmark.yaml",
        help="YAML grid config",
    )
    parser.add_argument(
        "--mini-grid",
        action="store_true",
        help="Use mini 2x2x2x1x2x2 grid for CI/smoke tests",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs (joblib)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/failure_mode",
        help="Output directory",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = root / ("configs/failure_mode_mini.yaml" if args.mini_grid else args.config)
    output_dir = root / args.output_dir

    run_dir = run_and_save(config_path, output_dir, n_jobs=args.n_jobs)
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
