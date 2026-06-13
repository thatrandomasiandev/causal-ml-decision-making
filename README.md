# Causal ML Decision-Making

A reproducible research benchmark suite for **heterogeneous treatment effect estimation**, **offline policy learning**, and **causal structure discovery**. Every module is evaluated on synthetic data-generating processes (DGPs) with oracle ground truth, so estimator bias, policy regret, and graph recovery error can be measured directly rather than inferred from observational proxies.

The central research question across modules is: *given realistic identification assumptions, how do standard causal estimators behave under controlled confounding, logging bias, and sample-size constraints?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Uplift** | Estimate conditional average treatment effects τ(x) = E[Y(1)−Y(0)\|X] | S-, T-, X-, R-learners; doubly robust learner | PEHE, ATE error, AUUC, QINI |
| **Policy** | Evaluate and learn treatment policies from logged bandit data | IPS, SNIPS, DR off-policy evaluation; threshold policies | Policy value, regret vs. oracle |
| **Discovery** | Recover causal structure from observational data | NOTEARS (static DAGs); PCMCI (time series) | SHD, precision, recall, F1 |

---

## Module 1: Uplift modeling

### Problem formulation

Given observational tuples (X, T, Y) where T ∈ {0,1} is treatment and Y is outcome, the goal is to estimate the **conditional average treatment effect**:

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

Under the **potential outcomes framework** (Rubin, 1974), valid inference requires:

- **Unconfoundedness:** (Y(0), Y(1)) ⊥ T | X
- **Positivity:** 0 < P(T=1|X) < 1 for all x in support
- **SUTVA:** No interference between units; single version of treatment

### Implemented estimators

| Estimator | Idea | Reference |
|-----------|------|-----------|
| **S-learner** | Single model with treatment as feature | Künzel et al. (2019) |
| **T-learner** | Separate outcome models per treatment arm | Künzel et al. (2019) |
| **X-learner** | Impute counterfactuals, then regress CATE | Künzel et al. (2019) |
| **R-learner** | Residualize outcome and treatment, regress product | Nie & Wager (2021) |
| **DR-learner** | Doubly robust pseudo-outcome regression | Kennedy (2022) |

All learners use cross-fitted gradient boosting for nuisance functions (propensity e(x) and outcome μ(x,t)), following the meta-learner framework of Künzel et al. (2019).

### Synthetic DGP (`data/uplift_dgp.py`)

- Covariates X ~ N(0, I)
- Propensity e(x) = σ(wᵀx) with tunable confounding strength
- Heterogeneous effect: τ(x) = x₀ + 0.5·x₁·x₂
- Potential outcomes Y(t) = μₜ(x) + ε with known μ₀, μ₁

Ground-truth τ(x) is available for every sample, enabling exact **PEHE** (precision in estimation of heterogeneous effects).

### Evaluation metrics

- **PEHE:** √(E[(τ̂(x) − τ(x))²]) — primary CATE accuracy metric (Hill, 2011)
- **ATE error:** |τ̂_ATE − τ_ATE|
- **AUUC / QINI:** Uplift ranking quality for policy targeting (Radcliffe & Surry, 2011; Devriendt et al., 2020)

---

## Module 2: Offline policy learning

### Problem formulation

Given logged trajectories from a **behavior policy** π_b with known propensities, estimate the value of a **target policy** π_e and learn policies that maximize expected reward under logging constraints.

This is the **counterfactual learning** / **off-policy evaluation (OPE)** setting studied in Dudík et al. (2011) and Swaminathan & Joachims (2015).

### Implemented methods

| Method | Estimator | Reference |
|--------|-----------|-----------|
| **IPS** | Importance-weighted reward: Σ (π_e/π_b)·r / Σ (π_e/π_b) | Horvitz & Thompson (1952) |
| **SNIPS** | Self-normalized IPS (variance reduction) | Swaminathan & Joachims (2015) |
| **DR** | Doubly robust value estimator | Dudík et al. (2011) |

Policy learning uses cost-sensitive thresholding on estimated CATE; regret is measured against the oracle policy with full counterfactual access.

### Synthetic DGP (`data/bandit_dgp.py`)

- ε-greedy logging policy with known propensities
- Known reward functions y₀(x), y₁(x) and oracle optimal policy
- Tunable logging suboptimality via ε

### Evaluation metrics

- **Policy value:** Expected reward under target policy
- **Regret:** V(π*) − V(π̂) relative to oracle
- **Coverage:** Fraction of (s,a) pairs with sufficient logging support

---

## Module 3: Causal discovery

### Problem formulation

Recover the causal graph G from observational data under **causal sufficiency** (no unmeasured confounders) and **faithfulness** (independencies reflect graph structure).

### Implemented methods

| Method | Setting | Reference |
|--------|---------|-----------|
| **NOTEARS** | Static linear DAGs via continuous optimization with acyclicity constraint | Zheng et al. (2018) |
| **PCMCI** | Time-series causal discovery with conditional independence testing | Runge et al. (2019) |

NOTEARS solves min_W L(W; X) s.t. h(W) = 0 (acyclicity), using augmented Lagrangian optimization. PCMCI uses the PC algorithm with momentary conditional independence (ParCorr) via tigramite.

### Synthetic DGPs

- **Static DAG** (`data/dag_dgp.py`): Random DAG + linear structural equation model; ground-truth adjacency matrix
- **Time series** (`data/dag_dgp.py`): VAR-style dynamics for PCMCI validation

### Evaluation metrics

- **SHD** (structural Hamming distance): Edge additions/deletions/flips vs. true graph
- **Precision / Recall / F1** on edge detection

---

## Benchmark protocol

Configs in `configs/` define sweeps over confounding strength, sample size, logging ε, and regularization λ. Each configuration is run across multiple random seeds; results are seed-averaged and written to `results/{timestamp}/`.

```bash
pip install -e ".[dev]"

# All modules
python scripts/run_benchmark.py --config configs/uplift_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/uplift_benchmark.yaml --module uplift
python scripts/run_benchmark.py --config configs/policy_benchmark.yaml --module policy
python scripts/run_benchmark.py --config configs/discovery_benchmark.yaml --module discovery

pytest
```

---

## Project layout

```
src/causal_ml/
├── data/          # Uplift, bandit, and DAG DGPs with ground_truth accessors
├── uplift/        # S/T/X/R/DR meta-learners, propensity, metrics
├── policy/        # IPS/SNIPS/DR evaluation, policy learning, simulation
├── discovery/     # NOTEARS, PCMCI, graph recovery metrics
├── evaluation/    # Benchmark runner and markdown reporting
└── utils/         # Cross-fitting, seed control
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_synthetic_dgp_walkthrough.ipynb` | Validate DGP ground-truth accessors |
| `02_uplift_benchmark.ipynb` | Compare meta-learners under confounding sweeps |
| `03_policy_offline_eval.ipynb` | IPS/SNIPS/DR bias and variance |
| `04_causal_discovery.ipynb` | Graph recovery on known DAGs |

---

## Implementation notes

- NOTEARS implements the **linear** formulation of Zheng et al. (2018); nonlinear extensions are not included
- PCMCI uses partial correlation (ParCorr) as the conditional independence test
- Cross-fitting is applied to all nuisance models to reduce overfitting bias in CATE estimation

---

## References

- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment response. *Journal of Econometrics*, 113(2), 231–263. [DOI](https://doi.org/10.1016/S0304-4076(03)00070-7)
- Devriendt, F., et al. (2020). Why you should stop predicting customer churn and start uplifting them. *Machine Learning*, 109, 367–393. [DOI](https://doi.org/10.1007/s10994-020-05933-4)
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and optimization. *ICML*. [Proceedings](https://proceedings.mlr.press/v15/dudik11a.html)
- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1), 217–240. [DOI](https://doi.org/10.1198/jcgs.2010.08162)
- Kennedy, E. H. (2022). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*. [arXiv](https://arxiv.org/abs/2004.14497)
- Künzel, S. R., et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156–4165. [DOI](https://doi.org/10.1073/pnas.1804597116)
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319. [DOI](https://doi.org/10.1093/biomet/asaa097)
- Radcliffe, N., & Surry, P. (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees*. Stochastic Solutions.
- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701. [DOI](https://doi.org/10.1037/h0037350)
- Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11), eaau4996. [DOI](https://doi.org/10.1126/sciadv.aau4996)
- Swaminathan, A., & Joachims, T. (2015). The self-normalized estimator for counterfactual learning. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper/5747-the-self-normalized-estimator-for-counterfactual-learning)
- Zheng, X., et al. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS*. [arXiv](https://arxiv.org/abs/1803.01422)

---

## Future work

- Real benchmarks (IHDP, Hillstrom) via the shared `CausalDataset` interface
- Sensitivity analysis: E-values (VanderWeele & Ding, 2017), Rosenbaum bounds
- Nonlinear NOTEARS and FCI for latent confounding
