# CATE Estimator Failure Modes: A Systematic Empirical Benchmark

## 1. Introduction

### Motivation
- Meta-learners (S/T/X/R/DR) are widely used for conditional average treatment effect (CATE) estimation, yet practitioners lack guidance on **when each method fails**.
- Existing papers introduce new estimators or prove theoretical guarantees, but rarely provide a **systematic empirical map** of failure modes as a function of overlap, confounding, heterogeneity, and sample size.
- This paper fills that gap with a reproducible benchmark suite and controlled synthetic DGPs plus standard real-world datasets.

### Contributions
1. **FailureModeDGP**: a tunable synthetic generator with independently controlled overlap, confounding, heterogeneity, and noise.
2. **Large-scale grid benchmark** (4,800+ runs) comparing S/T/X/R/DR learners with PEHE, ATE error, AUUC, and QINI.
3. **Real-world validation** on IHDP, Jobs, and Twins with overlap diagnostics and sensitivity analysis (Rosenbaum bounds, E-values, placebo tests).
4. **Actionable decision guide**: when to prefer T-learner over DR-learner, and vice versa.

---

## 2. Background

### 2.1 Identification Assumptions
- **Unconfoundedness**: $(Y(0), Y(1)) \perp T \mid X$
- **Positivity**: $0 < e(x) < 1$
- **SUTVA**: no interference, well-defined treatment

### 2.2 Meta-Learner Definitions
- **S-learner**: single outcome model $\mu(X, T)$
- **T-learner**: separate models $\mu_0(X)$, $\mu_1(X)$
- **X-learner**: imputation + propensity-weighted CATE regression
- **R-learner**: Robinson residual-on-residual decomposition
- **DR-learner**: doubly robust pseudo-outcome regression (AIPW)

### 2.3 Evaluation Metrics
- **PEHE**: $\sqrt{\frac{1}{n}\sum_i (\hat{\tau}(x_i) - \tau(x_i))^2}$
- **ATE error**: $|\hat{\tau}_{ATE} - \tau_{ATE}|$
- **AUUC / QINI**: targeting/ranking quality for policy decisions

---

## 3. Experimental Setup

### 3.1 FailureModeDGP (Exact Equations)
See `src/causal_ml/data/synthetic.py` docstring:
- $X \sim \mathcal{N}(0, I_p)$
- $e(x) = \sigma(\alpha \cdot x_0)$ with $\alpha = f(\text{overlap})$
- Confounded propensity: $\tilde{e}(x) = \sigma(\text{logit}(e(x)) + \gamma \cdot g(x))$
- $\tau(x) = \lambda \cdot f(x) + (1-\lambda) \cdot ATE$
- $Y = \mu(x) + T \cdot \tau(x) + \epsilon$

### 3.2 Real Datasets
| Dataset | n | p | Outcome | tau_true |
|---------|---|---|---------|----------|
| IHDP | 747 | 25 | Continuous | Yes (100 realizations) |
| Jobs | ~722 | varies | Binary | No |
| Twins | ~11k | varies | Binary | Yes (counterfactuals) |

### 3.3 Grid Specification
- Overlap: [1.0, 0.5, 0.2, 0.1]
- Confounding: [0.0, 0.3, 0.6, 1.0]
- Heterogeneity: [0.0, 0.3, 0.7, 1.0]
- n: [500, 1000, 5000]
- Estimators: S, T, X, R, DR
- Seeds: 5 per cell

### 3.4 Sensitivity Analysis
- Rosenbaum bounds ($\Gamma \in [1, 3]$)
- E-values (VanderWeele & Ding, 2017)
- Placebo permutation tests (200 permutations)

---

## 4. Results

### 4.1 Failure Mode Grid
- PEHE heatmaps: overlap × confounding, faceted by estimator
- Learning curves: PEHE vs n at each heterogeneity level
- **Surprising failures**: cells where DR PEHE > T PEHE (report top-10 margins)

### 4.2 Real-World Datasets
- ATE error and overlap diagnostics per dataset
- Comparison of estimator rankings on IHDP vs synthetic DGP

### 4.3 Sensitivity Analysis
- Rosenbaum bounds for IHDP ATE
- E-values quantifying robustness to unmeasured confounding
- Placebo null distributions per estimator

---

## 5. Discussion

### Key Findings (to be populated from results)
- DR-learner advantages under moderate confounding with sufficient overlap
- T-learner robustness when overlap is poor but treatment groups are large
- X/R-learner tradeoffs at small n and high heterogeneity
- Overlap violations as the dominant failure driver across all estimators

### Limitations
- Linear/logistic nuisance models only (no neural TARNet/DragonNet in v1)
- Rosenbaum bounds assume matched-pair structure approximation
- Jobs dataset lacks ground-truth CATE

---

## 6. Conclusion

We provide the first systematic empirical map of CATE meta-learner failure modes, enabling practitioners to select estimators based on diagnosable data properties rather than defaults. All code, configs, and results are fully reproducible via `scripts/run_failure_mode_benchmark.py`.
