# Scripts

Each script reproduces one part of the paper. All scripts take their inputs as
arguments or environment variables and document the paper settings in `--help`.
Training, sampling, the $10^4 \times 10^4$ optimal transport and the MCMC fits
belong on a GPU or CPU cluster node.

| Paper | Script |
|---|---|
| Training of every reported model | `train.py`, `create_parent_checkpoints.py` |
| Table 1 and unclipped metrics, box-volume correction, YFPHAGYT variant | `sample_many_peptides.py`, `evaluate_many_peptides.py` |
| ROBIN baseline | `sample_many_peptides.py --robin` |
| Scaling model, its sensitivity analysis, MDQM9-nc scaling | `fit_scaling_model.py` |
| Speculative decoding and the sampling-rate model | `benchmark_sampling_rate.py`, `fit_rate_model.py` |
| Sampling cost against MD | `cost_vs_md.py` |
| Quantization roundtrips, mean squared force | `roundtrip.py` |
| Solvation free energies | `solvation_free_energy.py` |
| Small-molecule and vacuum-peptide sampling | `evaluate_small_molecules.py` |
| Microsolvation against ORCA SOLVATOR and tleap | `microsolvate.py` |
| Stereochemistry of generated samples | `stereo_statistics.py` |
| D-proline artifact | `dpro_artifact.py` |
| Counterions in the microsolvated MD | `counterion_audit.py` |

`fit_scaling_model.py` and `fit_rate_model.py` read their inputs from
[`data/`](../data/README.md). `pbg.analysis` turns on JAX x64 when imported, so
run the fits in their own process.
