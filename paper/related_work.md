# Related Work — Bibliography Scaffold

## Core Meta-Learners

```bibtex
@inproceedings{kunzel2019metalearners,
  title={Metalearners for estimating heterogeneous treatment effects using machine learning},
  author={K{\"u}nzel, S{\"o}ren R and Sekhon, Jasjeet S and Bickel, Peter J and Yu, Bin},
  booktitle={PNAS},
  year={2019}
}
```
**Relation:** Defines S/T/X/R-learners that this paper benchmarks systematically; our contribution is empirical failure mapping, not a new learner.

```bibtex
@article{nie2021quasi,
  title={Quasi-oracle estimation of heterogeneous treatment effects},
  author={Nie, Xinkun and Wager, Stefan},
  journal={Biometrika},
  year={2021}
}
```
**Relation:** R-learner theory; we evaluate R-learner failure modes under overlap violation.

```bibtex
@article{kennedy2023towards,
  title={Towards optimal doubly robust estimation of heterogeneous causal effects},
  author={Kennedy, Edward H},
  journal={Electronic Journal of Statistics},
  year={2023}
}
```
**Relation:** DR-learner optimality; we test when DR underperforms simpler T-learner in practice.

```bibtex
@inproceedings{shalit2017estimating,
  title={Estimating individual treatment effect: generalization bounds and algorithms},
  author={Shalit, Uri and Johansson, Fredrik D and Sontag, David},
  booktitle={ICML},
  year={2017}
}
```
**Relation:** TARNet/CFR for neural CATE; motivates IHDP benchmark we extend with failure-mode grid.

```bibtex
@article{athey2016recursive,
  title={Recursive partitioning for heterogeneous causal effects},
  author={Athey, Susan and Imbens, Guido},
  journal={PNAS},
  year={2016}
}
```
**Relation:** Causal forest baseline; cited for completeness in heterogeneous effect literature.

## Benchmark Methodology

```bibtex
@article{dorie2019automated,
  title={Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition},
  author={Dorie, Vincent and Hill, Jennifer and Shalit, Uri and Scott, Marc and Cervone, Dan},
  journal={Statistical Science},
  year={2019}
}
```
**Relation:** ACIC competition methodology; precedent for systematic estimator comparison.

```bibtex
@article{hill2011bayesian,
  title={Bayesian nonparametric modeling for causal inference},
  author={Hill, Jennifer L},
  journal={JCGS},
  year={2011}
}
```
**Relation:** IHDP semi-synthetic dataset origin; primary real-world benchmark.

## Off-Policy Evaluation

```bibtex
@inproceedings{dudik2011doubly,
  title={Doubly robust policy evaluation and learning},
  author={Dudik, Miroslav and Langford, John and Li, Lihong},
  booktitle={ICML},
  year={2011}
}
```
**Relation:** DR policy evaluation connects to our policy module; cited for decision-making pipeline.

```bibtex
@article{thomas2016data,
  title={Data-efficient off-policy policy evaluation for reinforcement learning},
  author={Thomas, Philip and Brunskill, Emma},
  journal={JMLR},
  year={2016}
}
```
**Relation:** SNIPS/IPS variance reduction relevant to policy evaluation component.

## Sensitivity Analysis

```bibtex
@article{vanderweele2017sensitivity,
  title={Sensitivity analysis in observational research: Introducing the E-value},
  author={VanderWeele, Tyler J and Ding, Peng},
  journal={Annals of Internal Medicine},
  year={2017}
}
```
**Relation:** E-value implementation in our sensitivity module.

```bibtex
@book{rosenbaum2002observational,
  title={Observational Studies},
  author={Rosenbaum, Paul R},
  year={2002},
  publisher={Springer}
}
```
**Relation:** Rosenbaum bounds for hidden confounding sensitivity.

## Deep Causal Models

```bibtex
@inproceedings{louizos2017causal,
  title={Causal effect inference with deep latent-variable models},
  author={Louizos, Christos and Shalit, Uri and Mooij, Joris M and Sontag, David and Zemel, Richard and Welling, Max},
  booktitle={NeurIPS},
  year={2017}
}
```
**Relation:** CEVAE/Twins dataset source; semi-synthetic benchmark with counterfactuals.

```bibtex
@inproceedings{shi2019adapting,
  title={Adapting neural networks for the estimation of treatment effects},
  author={Shi, Claudia and Blei, David and Veitch, Victor},
  booktitle={NeurIPS},
  year={2019}
}
```
**Relation:** DragonNet; neural baseline cited as future extension.

## Additional References

```bibtex
@article{imbens2015causal,
  title={Causal inference in statistics, social, and biomedical sciences},
  author={Imbens, Guido W and Rubin, Donald B},
  year={2015},
  publisher={Cambridge University Press}
}
```
**Relation:** Foundational identification assumptions referenced throughout.

```bibtex
@article{chernozhukov2018double,
  title={Double/debiased machine learning for treatment and structural parameters},
  author={Chernozhukov, Victor and others},
  journal={The Econometrics Journal},
  year={2018}
}
```
**Relation:** DML framework underpinning cross-fitted nuisances in our implementation.

```bibtex
@article{curth2021nonparametric,
  title={Nonparametric estimation of heterogeneous treatment effects: Promises and pitfalls},
  author={Curth, Alicia and van der Schaar, Mihaela},
  journal={NeurIPS},
  year={2021}
}
```
**Relation:** Recent survey of CATE estimation pitfalls; directly motivates our failure-mode study.
