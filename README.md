# Causal ML Decision-Making

PhD-level causal ML suite covering **uplift estimation**, **offline policy learning**, and **causal discovery** — all evaluated on synthetic data with known ground truth.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Uplift** | S/T/X/R + DR meta-learners for heterogeneous treatment effects | PEHE, ATE error, AUUC, QINI |
| **Policy** | IPS, SNIPS, DR off-policy evaluation + regret simulation | Policy value, regret vs oracle |
| **Discovery** | NOTEARS (static DAGs) + PCMCI (time series) | SHD, precision, recall, F1 |

## Identification assumptions

- **Uplift:** Unconfoundedness given X, positivity (0 < e(x) < 1), SUTVA
- **Policy:** Logged propensities available, support overlap for target policy
- **Discovery:** Causal sufficiency, acyclicity (NOTEARS), stationarity (PCMCI)

## Setup

```bash
cd 01-causal-ml-decision-making
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python scripts/run_benchmark.py --config configs/uplift_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/uplift_benchmark.yaml --module uplift
python scripts/run_benchmark.py --config configs/policy_benchmark.yaml --module policy
python scripts/run_benchmark.py --config configs/discovery_benchmark.yaml --module discovery
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/causal_ml/
├── data/          # Synthetic DGPs with ground-truth accessors
├── uplift/        # Meta-learners and metrics
├── policy/        # Off-policy eval and simulation
├── discovery/     # NOTEARS + PCMCI
└── evaluation/    # Benchmark runner and reporting
```

## Notebooks

- `notebooks/01_synthetic_dgp_walkthrough.ipynb` — validate DGPs
- `notebooks/02_uplift_benchmark.ipynb` — uplift estimator comparison
- `notebooks/03_policy_offline_eval.ipynb` — off-policy evaluation
- `notebooks/04_causal_discovery.ipynb` — graph recovery

## Future work

- Real datasets (IHDP, Hillstrom) via the same `CausalDataset` interface
- Sensitivity analysis (E-values, Rosenbaum bounds)
- Streamlit policy simulation dashboard
