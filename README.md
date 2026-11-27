# Prefix-Conditioned Boltzmann Generators (PBG)

JAX implementation of **Prefix-Conditioned Boltzmann Generators** (Rindtorff & Tong, 2026):
Autoregressive Boltzmann generators that represent the molecular environment, the chemical
graph (HELM / SMILES) and discretized atomic coordinates as one token sequence.

## Installation

Python 3.13. CUDA 12 or 13 for GPU training and sampling.

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[cuda13,training,analysis,dev]"
```

`openmm` computes all potential energies. The microsolvation baselines also need
[ORCA 6](https://www.faccts.de/orca/) with xtb (`OPI_ORCA`, `XTBEXE`) and AmberTools `tleap`
(`AMBERHOME`).

## Data and checkpoints

| Artifact | Location |
|---|---|
| Checkpoints of the models named in the paper and their parents | [pbg](https://huggingface.co/niklastr/pbg) |
| Sampled rollouts of these models | [pbg_rollouts](https://huggingface.co/datasets/niklastr/pbg_rollouts) |
| Many Peptides webdatasets and reference trajectories | [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md) |
| Vacuum Peptides webdatasets | [vacuum_peptides](https://huggingface.co/datasets/niklastr/vacuum_peptides) |
| Microsolvated Peptides webdatasets | [solvated_peptides](https://huggingface.co/datasets/niklastr/solvated_peptides) |
| FreeSolv webdatasets (vacuum, implicit and explicit water) | [freesolv](https://huggingface.co/datasets/niklastr/freesolv) |
| MDQM9-nc webdatasets | [smiles_mdqm9nc](https://huggingface.co/datasets/niklastr/smiles_mdqm9nc) |

The checkpoint release holds the models that the paper names: the PBG and PBG-Joint models and
the ROBIN baseline of Table 1, the 15M drafter and 85M target of speculative decoding, the
peptide-only control of PBG-Joint, the 47M and 85M MDQM9-nc models, the 85M vacuum and
microsolvation models, and the 85M Many Peptides model of the vacuum transfer row. It also holds
the untrained parents of these models; `scripts/create_parent_checkpoints.py` creates the parent of
any other config in [`configs/`](configs/README.md). The other checkpoints of
the scaling cohorts, the ablation models and the intermediate checkpoints are not released: a
run that continues from an intermediate checkpoint needs the run before it in its chain, and the
scaling fits read the measured values in [`data/`](data/README.md).

The MDQM9-nc runs and evaluations of the paper (`configs/mdqm9_scaling` and the two MDQM9-nc models
in `configs/completep`) used an earlier strict release of MDQM9-nc, which left out 1,180 of its
12,506 molecules and kept 95 test molecules. The Hugging Face dataset now holds all molecules.

## Quick start

Sample the Many Peptides test systems with three sampling seeds, as in the
paper (GPU):

```bash
for seed in 2026111101 2026111102 2026111103; do
  python scripts/sample_many_peptides.py \
      --checkpoint dipper_85m.ckpt --seed $seed \
      --output-dir rollouts/seed$seed
done
```

Score the rollouts against the reference MD with the unclipped metrics (CPU;
`--help` describes the other metric groups):

```bash
python scripts/evaluate_many_peptides.py rollouts/seed* \
    --metrics starred --output results
```

Train from a paper config (checkpoints go to `$PBG_CHECKPOINT_BASE`):

```bash
python scripts/create_parent_checkpoints.py core --models 15m \
    --output-dir $PBG_CHECKPOINT_BASE/parents
python scripts/train.py \
    configs/core_scaling/stable/dipper_15m_core_bin4096_ws_scaling.json
```

## Reproducing the paper

[`scripts/README.md`](scripts/README.md) lists the script behind each result of
the paper. The training configs of all reported runs are in
[`configs/`](configs/README.md).
