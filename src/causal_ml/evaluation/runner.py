"""Benchmark runner for uplift, policy, and discovery modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from causal_ml.data.bandit_dgp import BanditDGPConfig, generate_bandit_data
from causal_ml.data.dag_dgp import DAGDGPConfig, generate_dag_data
from causal_ml.data.uplift_dgp import UpliftDGPConfig, generate_uplift_data
from causal_ml.discovery.metrics import edge_precision_recall, structural_hamming_distance
from causal_ml.discovery.notears import notears
from causal_ml.discovery.pcmci import pcmci
from causal_ml.policy.evaluation import dr_policy_value, ips, snips
from causal_ml.policy.learning import threshold_policy
from causal_ml.policy.simulation import simulate_from_tau
from causal_ml.uplift.doubly_robust import dr_learner
from causal_ml.uplift.meta_learners import fit_uplift
from causal_ml.uplift.metrics import ate_error, auuc, pehe, qini
from causal_ml.utils.seed import config_hash


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys if isinstance(results[0][k], (int, float))}


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def run_uplift_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run uplift meta-learner sweep."""
    seeds = config.get("seeds", [42])
    estimators = config.get("estimators", ["S", "T", "X", "R", "DR"])
    confounding_levels = config.get("confounding_levels", [0.5, 1.0, 2.0])
    sample_sizes = config.get("sample_sizes", [1000, 5000])
    n_splits = config.get("n_splits", 3)

    all_results = []
    for confounding in confounding_levels:
        for n_samples in sample_sizes:
            for estimator in estimators:
                seed_results = []
                for seed in seeds:
                    data = generate_uplift_data(
                        UpliftDGPConfig(
                            n_samples=n_samples,
                            confounding_strength=confounding,
                            seed=seed,
                        )
                    )
                    if estimator == "DR":
                        result = dr_learner(data.X, data.T, data.Y, n_splits=n_splits, seed=seed)
                    else:
                        result = fit_uplift(
                            data.X, data.T, data.Y, estimator=estimator, n_splits=n_splits, seed=seed
                        )
                    tau_true = data.ground_truth["tau"]
                    seed_results.append(
                        {
                            "pehe": pehe(result.tau_hat, tau_true),
                            "ate_error": ate_error(result.tau_hat, tau_true),
                            "auuc": auuc(result.tau_hat, tau_true),
                            "qini": qini(result.tau_hat, tau_true),
                        }
                    )
                mean = _aggregate(seed_results)
                std = _aggregate_std(seed_results)
                all_results.append(
                    {
                        "estimator": estimator,
                        "confounding": confounding,
                        "n_samples": n_samples,
                        **{f"{k}_mean": v for k, v in mean.items()},
                        **{f"{k}_std": v for k, v in std.items()},
                    }
                )
    return {"module": "uplift", "results": all_results}


def run_policy_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run offline policy evaluation benchmark."""
    seeds = config.get("seeds", [42])
    epsilons = config.get("epsilons", [0.1, 0.3])
    n_samples = config.get("n_samples", 5000)
    n_splits = config.get("n_splits", 3)

    all_results = []
    for epsilon in epsilons:
        seed_results = []
        for seed in seeds:
            data = generate_bandit_data(
                BanditDGPConfig(n_samples=n_samples, epsilon=epsilon, seed=seed)
            )
            uplift = fit_uplift(data.X, data.actions, data.rewards, estimator="T", n_splits=n_splits, seed=seed)
            target = threshold_policy(uplift.tau_hat)

            ips_res = ips(data.actions, data.rewards, data.propensities, target)
            snips_res = snips(data.actions, data.rewards, data.propensities, target)

            y0 = data.ground_truth["y0"]
            y1 = data.ground_truth["y1"]
            mu1 = y1.copy()
            mu0 = y0.copy()
            mu_hat = np.column_stack([mu0, mu1])
            dr_res = dr_policy_value(
                data.actions, data.rewards, data.propensities, target, mu_hat
            )

            sim = simulate_from_tau(
                uplift.tau_hat,
                y0,
                y1,
                data.ground_truth["oracle_value"],
            )

            seed_results.append(
                {
                    "ips_value": ips_res.value,
                    "snips_value": snips_res.value,
                    "dr_value": dr_res.value,
                    "true_oracle": data.ground_truth["oracle_value"],
                    "simulated_value": sim.policy_value,
                    "regret": sim.regret,
                }
            )
        mean = _aggregate(seed_results)
        std = _aggregate_std(seed_results)
        all_results.append(
            {
                "epsilon": epsilon,
                "n_samples": n_samples,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )
    return {"module": "policy", "results": all_results}


def run_discovery_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run NOTEARS and PCMCI graph recovery benchmarks."""
    seeds = config.get("seeds", [42])
    n_samples_list = config.get("sample_sizes", [1000, 5000])
    all_results = []

    for n_samples in n_samples_list:
        notears_results = []
        pcmci_results = []
        for seed in seeds:
            static = generate_dag_data(DAGDGPConfig(n_samples=n_samples, seed=seed, time_series=False))
            notears_res = notears(static.X, lambda1=config.get("lambda1", 0.1))
            true_adj = static.ground_truth["adjacency"]
            pr = edge_precision_recall(true_adj, notears_res.adjacency)
            notears_results.append(
                {
                    "shd": structural_hamming_distance(true_adj, notears_res.adjacency),
                    **pr,
                }
            )

            ts = generate_dag_data(
                DAGDGPConfig(n_timesteps=n_samples, seed=seed + 100, time_series=True)
            )
            pcmci_res = pcmci(ts.X, tau_max=config.get("tau_max", 2))
            true_ts = ts.ground_truth["adjacency"]
            pr_ts = edge_precision_recall(true_ts, pcmci_res.adjacency)
            pcmci_results.append(
                {
                    "shd": structural_hamming_distance(true_ts, pcmci_res.adjacency),
                    **pr_ts,
                }
            )

        all_results.append(
            {
                "n_samples": n_samples,
                "notears_shd_mean": _aggregate(notears_results).get("shd", 0),
                "notears_f1_mean": _aggregate(notears_results).get("f1", 0),
                "pcmci_shd_mean": _aggregate(pcmci_results).get("shd", 0),
                "pcmci_f1_mean": _aggregate(pcmci_results).get("f1", 0),
            }
        )

    return {"module": "discovery", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config = load_config(config_path)
    merged = {**load_config(Path(config_path).parent / "default.yaml"), **config}

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("uplift", "all"):
        results["modules"]["uplift"] = run_uplift_benchmark(merged)
    if module in ("policy", "all"):
        results["modules"]["policy"] = run_policy_benchmark(merged)
    if module in ("discovery", "all"):
        results["modules"]["discovery"] = run_discovery_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from causal_ml.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")

    return run_dir
