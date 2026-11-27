#!/usr/bin/env python
"""Sampling metrics of one small molecule or vacuum peptide from a PBG rollout.

Paper, Appendix "Solvation Free Energy Estimation", paragraph "Small Molecule
Sampling": the tables "Transferable Sampling of Small Molecules in MDQM9-nc"
and "... in FreeSolv", and the table "Vacuum Peptide Sampling across Training
Conditions". Mirrored stereoisomers are reflected (coordinates only; every
log-likelihood stays the one of the generated conformer). Rollout and
reference frames are scored with the scored potential, or the stored energies
are used. Importance weights are exp(-beta U - log q) of the stereo-valid
proposals, without clipping.

  rESS*  end_to_end_ress: Kish ESS over all generated proposals
  E-W2*  importance_weighted_energy_w2_kbt: weighted W2 of reduced energies beta U
  T-W2*  importance_weighted_global_torus_w2_sliced_rad: weighted sliced torus W2
         of all torsions around heavy-atom bonds (Sobolev slice weights,
         orders <= 2, max_slices=1024, n_random=128, slice RNG from --seed)

A table cell is the mean (MDQM9-nc, vacuum peptides) or median (FreeSolv) of
these per-system values over the systems of its group.

Examples:

  python scripts/evaluate_small_molecules.py --rollout 00189.npz \\
      --reference mdqm9-nc/test/XDIFZBLCRBYDHT-RXMQYKEDSA-N.npz \\
      --target gaff2-am1bcc --parameters params/00189 --seed 1851471103 --output 00189.json
  python scripts/evaluate_small_molecules.py --rollout AA.npz \\
      --reference vacuum_peptides/test/2AA/AA/state_00_300p00K.npz \\
      --target ff14sb --sequence AA --output AA.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.eval.solvation import atom_rows, model_energy, read_frames


def sampling_metrics(samples, logq, energy, smiles, reference_samples, reference_energy, *, temperature_K, seed):
    """rESS*, E-W2* and T-W2* of one rollout; rollout and reference share the model atom order."""
    from rdkit import Chem

    from pbg.core.stereo import intended_stereo_mask
    from pbg.estimator import (
        global_torsions,
        importance_weights,
        relative_ess,
        sliced_torus_w2,
        w2_1d,
    )

    valid = (intended_stereo_mask(smiles, samples[:, :, 1:4]) & np.isfinite(logq) & np.isfinite(energy)
             & np.all(np.isfinite(samples), axis=(1, 2)))
    metrics = dict(end_to_end_ress=0.0, importance_weighted_energy_w2_kbt=None,
                   importance_weighted_global_torus_w2_sliced_rad=None, stereo_valid_fraction=float(valid.mean()))
    if not valid.any():
        return metrics, valid
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * temperature_K)
    log_weight = -beta * energy[valid] - logq[valid]
    weights = importance_weights(log_weight)
    metrics["end_to_end_ress"] = relative_ess(weights, n_proposals=len(samples))
    metrics["importance_weighted_energy_w2_kbt"] = w2_1d(beta * energy[valid], beta * reference_energy, x_weights=weights)
    # Torsion indices follow SMILES encounter order; frames follow atom-map order.
    encounter = [atom.GetAtomMapNum() - 1 for atom in Chem.MolFromSmiles(smiles, sanitize=False).GetAtoms()]
    angles = global_torsions(samples[valid][:, encounter], smiles)
    if angles.shape[1]:
        metrics["importance_weighted_global_torus_w2_sliced_rad"] = sliced_torus_w2(
            angles, global_torsions(reference_samples[:, encounter], smiles),
            x_weights=weights, max_order=2, max_entry=2, max_nnz=2, max_slices=1024, n_random=128,
            rng=np.random.default_rng(seed))
    return metrics, valid


def main():
    from pbg.core.peptide import sequence_to_smiles
    from pbg.models.pbg.rollout import repair_all_mirrored_stereo

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout", type=Path, nargs="+", required=True,
                        help="NPZ with positions, generator_log_likelihood, smiles, helm, environment, temperature")
    parser.add_argument("--reference", type=Path, nargs="+", required=True,
                        help="reference MD NPZ with positions and smiles (peptides: --sequence) and energy")
    parser.add_argument("--target", help="scored energy protocol, e.g. gaff2-am1bcc, gaff2-am1bcc-obc1-2nm, ff14sb")
    topology = parser.add_mutually_exclusive_group()
    topology.add_argument("--parameters", type=Path,
                          help="directory with a GAFF2/AM1-BCC lig.prmtop in the rollout atom order; "
                               "created from the first reference frame when missing (AmberTools)")
    topology.add_argument("--sequence", help="peptide sequence (ff14SB topology)")
    parser.add_argument("--stored-energies", action="store_true",
                        help="use the energy arrays of the rollout and reference files instead of rescoring")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed of the random torus slices")
    parser.add_argument("--platform", default=os.environ.get("PBG_OPENMM_PLATFORM", "CPU"), help="OpenMM platform")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.stored_energies and not (args.target and (args.parameters or args.sequence)):
        parser.error("rescoring needs --target and --parameters or --sequence")

    rollout = read_frames(args.rollout)
    smiles, temperature = rollout["smiles"], float(rollout["temperature"])
    helm, environment = rollout.get("helm"), rollout.get("environment", "vacuum")
    # Reference atoms follow the rollout order; both are matched through the prompt topology.
    reference = read_frames(args.reference, smiles=sequence_to_smiles(args.sequence) if args.sequence else None,
                            rows=lambda source: atom_rows(smiles, source, helm, environment))
    samples, reflected, _ = repair_all_mirrored_stereo(rollout["positions"], smiles)
    if args.stored_energies:
        energy, reference_energy = rollout["energy"], reference["energy"]
    else:
        prmtop = args.parameters / "lig.prmtop" if args.parameters else None
        if prmtop and not prmtop.exists():
            from pbg.energy import parameterize_gaff

            parameterize_gaff(smiles, reference["positions"][0], args.parameters)
        score = model_energy(args.target, smiles, helm, environment, prmtop=prmtop, sequence=args.sequence,
                             platform=args.platform)
        energy, reference_energy = score(samples), score(reference["positions"])
    metrics, valid = sampling_metrics(samples, rollout["logq"], energy, smiles, reference["positions"],
                                      reference_energy, temperature_K=temperature, seed=args.seed)
    result = dict(system=rollout.get("id") or args.rollout[0].stem, smiles=smiles, temperature_k=temperature,
                  target=args.target, energies="stored" if args.stored_energies else "rescored", seed=args.seed,
                  rollout=[str(p) for p in args.rollout], reference=[str(p) for p in args.reference],
                  atom_count=int(samples.shape[1]), n_generated=len(samples), n_valid=int(valid.sum()),
                  n_reflected=int(reflected.sum()), n_reference=len(reference["positions"]), metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
