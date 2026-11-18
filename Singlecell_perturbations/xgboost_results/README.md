# XGBoost results

Existing result files in this directory are preserved as their original experiments.

| Run | Scope | Result and reproducibility |
|---|---|---|
| [2026-11-17: identity + PCA160 tuning](tuning/2026-11-17-identity-pca160/README.md) | 15 XGBoost configurations, fixed NK/CD4/CD8/Treg sparse-cell validation | T09 mean MRRMSE 1.007402 vs T00 1.032013; full CSVs, parameter comparisons, report, code and Slurm evidence |

Scores from different validation splits or input representations should not be compared as if they used the same evaluation protocol.
