# Analysis inputs

Tidy tables behind the Bayesian analyses of the paper. They are the exact inputs of the reported fits: the release fit scripts
reproduce the published log densities, posterior summaries and short NUTS runs.
Read them with `pbg.analysis.read_table`.

| File | Rows | Used by |
|---|---:|---|
| `scaling_many_peptides_v23_checkpoints.csv.gz` | 134 | `scripts/fit_scaling_model.py` (130 fitting and 4 held-out checkpoints: $N$, $D$, bin count, terminal NLL, role) |
| `scaling_many_peptides_v23_observations.csv.gz` | 36,582 | per checkpoint and test system: atom count, $\mathrm{rESS}^\ast$, $\mathcal{E}\text{-}\mathcal{W}_2^\ast$, $\mathbb{T}\mathrm{g}\text{-}\mathcal{SW}_2^\ast$ |
| `scaling_many_peptides_v23_init.json` | | NUTS starting values (posterior medians of an earlier fit) |
| `scaling_mdqm9_n36_checkpoints.csv.gz` | 36 | MDQM9-nc cohort; the paper's n28 cohort keeps runs with at least 3694 optimizer steps (`--cohort n28`, default) |
| `scaling_mdqm9_n36_observations.csv.gz` | 10,260 | MDQM9-nc per-system metrics |
| `sampling_rate_cells.csv.gz` | 674 | `scripts/fit_rate_model.py`: AR and speculative-decoding timings (median of five repeats, batch 512) with their work features |
| `sampling_rate_architectures.csv.gz` | 7 | architectures of the timed models (paper table "Model Architectures and Inference Covariates") |

The per-system metrics use the unclipped estimators of `pbg.eval.metrics.starred_metrics`
on one rollout of $10^4$ samples per checkpoint and test system (91 Many Peptides test
systems, YFPHAGYT excluded). The timings come from the protocol of
`scripts/benchmark_sampling_rate.py`.
