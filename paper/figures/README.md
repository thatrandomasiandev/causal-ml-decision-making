# Paper Figures Specification

## Figure 1: PEHE Heatmap (Overlap × Confounding)

- **Type:** 2D heatmap, one panel per estimator (S, T, X, R, DR)
- **Axes:** x = overlap [1.0, 0.5, 0.2, 0.1], y = confounding [0.0, 0.3, 0.6, 1.0]
- **Color:** mean PEHE (averaged over seeds, n=5000, heterogeneity=0.7)
- **Source:** `results/failure_mode/{timestamp}/summary.csv`
- **Script:** `paper/figures/plot_pehe_heatmap.py` (to be implemented)

## Figure 2: PEHE vs n Learning Curves

- **Type:** Line plot with error bars (±1 std over seeds)
- **Axes:** x = n [500, 1000, 5000], y = PEHE
- **Facets:** one row per heterogeneity level [0.0, 0.3, 0.7, 1.0], colored by estimator
- **Fixed:** overlap=1.0, confounding=0.3

## Figure 3: ATE Error Violin Plots (Real Datasets)

- **Type:** Violin or box plot
- **X-axis:** estimator (S, T, X, R, DR)
- **Y-axis:** |ATE_hat - ATE_true| (IHDP, Twins) or cross-fit ATE stability (Jobs)
- **Groups:** one violin per dataset

## Figure 4: DR vs T PEHE Margin Scatter

- **Type:** Scatter plot
- **X-axis:** T-learner PEHE
- **Y-axis:** DR-learner PEHE
- **Points:** one per grid cell (averaged over seeds)
- **Diagonal:** y=x reference line; points above diagonal = surprising DR failures
- **Highlight:** top-10 margin cells annotated

## Figure 5: Rosenbaum Bounds (IHDP)

- **Type:** Line plot with shaded significance region
- **X-axis:** gamma (1.0 to 3.0)
- **Y-axis:** p-value bounds (p_lower, p_upper)
- **Reference:** horizontal line at 0.05

## Figure 6: Placebo Test Null Distributions

- **Type:** Histogram overlay (one per estimator)
- **X-axis:** permuted ATE under null
- **Y-axis:** density
- **Vertical line:** observed ATE
- **Dataset:** FailureModeDGP with confounding=0 (true null after permutation)

## Data Dependencies

All figures read from:
- `results/failure_mode/{timestamp}/results.parquet`
- `results/failure_mode/{timestamp}/summary.csv`
- Real dataset results (future: `results/real_world/{timestamp}/`)

## Style Guidelines

- Colorblind-safe palette (Okabe-Ito)
- Font: serif, 10pt minimum
- Export: PDF vector + PNG 300dpi for review
