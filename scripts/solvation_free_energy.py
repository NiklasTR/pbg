#!/usr/bin/env python
"""Solvation free energies of FreeSolv molecules and peptides with PBG-Joint.

Paper, Appendix "Solvation Free Energy Estimation": "Zero Free Energy
Proposals", the BAR overlap of "Evaluation Metrics", and "Cutoff Offset of the
Implicit-Water Energies". Library code: ``pbg.eval.solvation``.

  freesolv       one FreeSolv molecule: vacuum (gaff2-am1bcc) and OBC1 water
                 (gaff2-am1bcc-obc1-2nm) legs with one shared GAFF2/AM1-BCC prmtop
  peptide        one peptide: vacuum (ff14sb) and OBC1 water (ff14sb-obc1-2nm) legs
  md-reference   MD-only BAR reference of one FreeSolv molecule in the scored
                 potentials, with the cutoff offset of its implicit-water frames
  cutoff-offset  cutoff offset of an implicit-water protocol on stored frames
  summarize      pass counts, Pearson r and errors over many systems

Each leg reads proposals or draws them with --checkpoint (GPU, batch 32, as in
the paper), reads or computes the model log-likelihood of the reference frames,
scores both with OpenMM, and estimates the leg with forward Zwanzig and BAR.
Mirrored stereoisomers are reflected (coordinates only); every log-likelihood
stays the one of the generated conformer. Sampled proposals and reference
log-likelihoods are kept in --output-dir and reused by a rerun.
All free energies are kJ/mol except in the summary (kcal/mol).

Example (paper settings):

  python scripts/solvation_free_energy.py freesolv --molecule mobley_1036761 \\
      --helm 'CHEM1{[NC1CCCCC1]}$$$$' --parameters params/mobley_1036761 \\
      --vacuum-reference vac/frame*/state_00_300p00K.npz \\
      --water-reference obc1/frame*/state_00_300p00K.npz \\
      --checkpoint dipper_160m_joint_bin4096_completep_d80n_v1_step_346240.ckpt \\
      --output-dir out/freesolv/mobley_1036761
  python scripts/solvation_free_energy.py peptide --sequence AA \\
      --vacuum-reference AA/state_00_300p00K.npz --water-reference AA_subsampled.npz \\
      --checkpoint ... --output-dir out/peptides/AA
  python scripts/solvation_free_energy.py md-reference --molecule mobley_1036761 \\
      --parameters params/mobley_1036761 --output-dir out/md_reference
  python scripts/solvation_free_energy.py cutoff-offset --system AA --sequence AA \\
      --reference AA_subsampled.npz --reference-temperature 310 --output out/offsets/AA.json
  python scripts/solvation_free_energy.py summarize --results out/freesolv/*/result.json \\
      --controls out/md_reference/*.json --experimental FreeSolv/database.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import cache
from pathlib import Path

import numpy as np

from pbg.core.constants import KJ_PER_KCAL
from pbg.eval.solvation import (
    OVERLAP_THRESHOLD,
    RESS_THRESHOLD,
    atom_rows,
    cutoff_offset,
    md_reference,
    md_reference_estimate,
    model_energy,
    read_frames,
    reference_selection,
    scored_system,
    solvation_estimate,
    zfep_leg,
)

# Kyte-Doolittle hydropathy; the mean over a sequence is its GRAVY score.
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    temporary.replace(path)


@cache
def load_model(checkpoint):
    """Load the PBG checkpoint once per process."""
    if checkpoint is None:
        raise SystemExit("--checkpoint is required to sample proposals or score reference frames")
    from pbg.models.pbg.model import PBG

    return PBG.load(checkpoint)


def leg(name, system, energy, frames, reference_energy, args):
    """One zero-free-energy-proposal leg (paper: Zero Free Energy Proposals)."""
    import jax

    from pbg.core.frame import DataFrame
    from pbg.models.pbg.rollout import repair_all_mirrored_stereo
    from pbg.models.pbg.sample import sample_molecular_conformers, score_frames

    def logq(conformers):
        """Teacher-forced log-likelihood at the paper's fixed batch size 32."""
        values = []
        for offset in range(0, len(conformers), 256):
            batch = [DataFrame(id=system.id, frame_id=offset + i, smiles=system.mapped_smiles, helm=system.helm,
                               environment=system.environment, temperature=system.temperature_K, conformer=x)
                     for i, x in enumerate(conformers[offset:offset + 256])]
            count = len(batch)
            padded = batch + [batch[-1]] * (-count % 32)
            values.append(score_frames(load_model(args.checkpoint), padded, batch_size=32)[:count])
        return np.concatenate(values)

    cached = args.output_dir / f"{name}_proposals.npz"
    sources = getattr(args, f"{name}_proposals") or ([cached] if cached.exists() else None)
    replay = None
    if sources:
        proposals = read_frames(sources)
        original, proposal_logq = proposals["positions"], proposals["logq"]
    else:
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{system.id}/{name}".encode()).digest()[:4], "little")
        # 256-proposal chunks keyed by fold_in(seed, chunk), each drawn in full batches of 32.
        chunks = [sample_molecular_conformers(load_model(args.checkpoint), system,
                                              jax.random.fold_in(jax.random.key(seed), index),
                                              num_samples=-(-min(256, args.proposals - offset) // 32) * 32, batch_size=32)
                  for index, offset in enumerate(range(0, args.proposals, 256))]
        original = np.concatenate([c.samples[:256] for c in chunks])[:args.proposals]
        proposal_logq = np.concatenate([c.logq[:256] for c in chunks])[:args.proposals]
        # Teacher forcing must reproduce the generated likelihood in this process.
        replay = float(np.max(np.abs(logq(original[:32]) - proposal_logq[:32])))
        if replay > 0.01:
            raise RuntimeError(f"{system.id}/{name}: likelihood replay error {replay:.3g} nats")
        np.savez(cached, positions=original, generator_log_likelihood=proposal_logq, seed=seed, replay_nats=replay)
    cached = args.output_dir / f"{name}_reference_logq.npz"
    sources = getattr(args, f"{name}_reference_logq") or ([cached] if cached.exists() else None)
    if sources:
        reference_logq = read_frames(sources)["logq"]
    else:
        reference_logq = logq(frames)
        np.savez(cached, logq=reference_logq)
    if len(reference_logq) != len(frames):
        raise ValueError(f"{name}: {len(reference_logq)} reference log-likelihoods for {len(frames)} frames")
    samples, reflected, _ = repair_all_mirrored_stereo(original, system.mapped_smiles)
    proposal_energy = energy(samples)
    np.savez(args.output_dir / f"{name}_inputs.npz", proposal_logq=proposal_logq,
             proposal_energy_kj_mol=proposal_energy, reference_logq=reference_logq,
             reference_energy_kj_mol=reference_energy)
    result = zfep_leg(proposal_logq, proposal_energy, reference_logq, reference_energy,
                      temperature_K=args.temperature, bootstrap=args.bootstrap, seed=args.bar_seed)
    return dict(result, target=system.environment, prompt_temperature_k=system.temperature_K,
                n_reflected=int(reflected.sum()), replay_error_nats=replay)


def finish(args, record, legs):
    record.update(checkpoint=str(args.checkpoint) if args.checkpoint else None, temperature_k=args.temperature,
                  legs=legs, solvation=solvation_estimate(legs["vacuum"], legs["water"]))
    write_json(args.output_dir / "result.json", record)
    print(json.dumps(dict(system=record["system"], **record["solvation"])), flush=True)


def freesolv(args):
    """FreeSolv molecule; reference frames are rescored with the shared prmtop (paper: FreeSolv)."""
    from pbg.energy import parameterize_gaff
    from pbg.models.pbg.system import MolecularSystem

    legs = {}
    for name, target in (("vacuum", "gaff2-am1bcc"), ("water", "gaff2-am1bcc-obc1-2nm")):
        system = MolecularSystem.from_protocol(id=args.molecule, helm=args.helm, target_id=target,
                                               temperature_K=args.temperature)
        reference = read_frames(getattr(args, f"{name}_reference"),
                                rows=lambda source: atom_rows(system.mapped_smiles, source, system.helm, system.environment))
        selection = reference_selection(reference["lengths"], args.reference_frames, args.discard_fraction)
        frames = reference["positions"][selection]
        if not (args.parameters / "lig.prmtop").exists():  # one parameter set for both legs (AmberTools)
            parameterize_gaff(system.mapped_smiles, frames[0], args.parameters)
        energy = model_energy(target, system.mapped_smiles, prmtop=args.parameters / "lig.prmtop", platform=args.platform)
        legs[name] = leg(name, system, energy, frames, energy(frames), args)
    finish(args, dict(kind="freesolv", system=args.molecule, helm=args.helm), legs)


def peptide(args):
    """Peptide; stored reference energies must match the scored potential (paper: Many Peptides)."""
    from pbg.core.peptide import sequence_to_helm, sequence_to_smiles
    from pbg.models.pbg.system import MolecularSystem

    legs = {}
    for name, target, temperature in (("vacuum", "ff14sb", args.temperature),
                                      ("water", "ff14sb-obc1-2nm", args.water_temperature)):
        system = MolecularSystem.from_protocol(id=args.sequence, helm=sequence_to_helm(args.sequence),
                                               target_id=target, temperature_K=temperature)
        reference = read_frames(getattr(args, f"{name}_reference"), smiles=sequence_to_smiles(args.sequence),
                                rows=lambda source: atom_rows(system.mapped_smiles, source, system.helm, system.environment))
        selection = reference_selection(reference["lengths"], getattr(args, f"{name}_reference_frames"), 0.0)
        frames, stored = reference["positions"][selection], reference["energy"][selection]
        energy = model_energy(target, system.mapped_smiles, system.helm, system.environment, sequence=args.sequence,
                              platform=args.platform)
        error = float(np.max(np.abs(energy(frames[:8]) - stored[:8])))
        if not error <= 0.1:
            raise ValueError(f"{args.sequence}/{name}: stored reference energies differ by {error} kJ/mol")
        legs[name] = dict(leg(name, system, energy, frames, stored, args), reference_energy_check_kj_mol=error)
    gravy = float(sum(KYTE_DOOLITTLE[aa] for aa in args.sequence) / len(args.sequence))
    finish(args, dict(kind="peptide", system=args.sequence, gravy=gravy), legs)


def md_reference_command(args):
    """MD-only BAR reference in the scored FreeSolv potentials (paper: Cutoff Offset paragraph)."""
    started = time.time()
    settings = dict(equilibration_ps=args.equilibration_ps, production_ns=args.production_ns, frames=args.frames,
                    seed=args.seed, temperature_K=args.temperature)
    energies, distance = md_reference(args.parameters, platform=args.platform, **settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.molecule}_energies.npz",
                        **{f"{state}__{potential}": value for (state, potential), value in energies.items()})
    result = dict(system=args.molecule, **md_reference_estimate(energies, temperature_K=args.temperature),
                  maximum_interatomic_distance_nm=distance, settings=settings,
                  parameter_sha256=hashlib.sha256((args.parameters / "lig.prmtop").read_bytes()).hexdigest(),
                  elapsed_seconds=time.time() - started)
    write_json(args.output_dir / f"{args.molecule}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "settings"}), flush=True)


def cutoff_offset_command(args):
    """F(OBC1 without cutoff) - F(scored 2 nm protocol) on stored implicit-water frames (topology order)."""
    from pbg.core.peptide import sequence_to_smiles
    from pbg.energy import ProtocolEvaluator

    target = args.target or ("ff14sb-obc1-2nm" if args.sequence else "gaff2-am1bcc-obc1-2nm")
    reference = read_frames(args.reference, smiles=sequence_to_smiles(args.sequence) if args.sequence else None)
    positions = reference["positions"]
    rows = np.linspace(0, len(positions) - 1, min(args.frames, len(positions))).round().astype(int)
    scored, uncut = (ProtocolEvaluator(scored_system(target, prmtop=args.prmtop, sequence=args.sequence, cutoff=cut),
                                       platform_name=args.platform)(positions[rows, :, 1:]) for cut in (True, False))
    shift = uncut - scored
    offset = cutoff_offset(scored, uncut, temperature_K=args.temperature, sample_temperature_K=args.reference_temperature)
    result = dict(system=args.system, target=target, reference=[str(p) for p in args.reference], frames=len(rows),
                  reference_temperature_k=args.reference_temperature or args.temperature,
                  stored_energy_max_error_kj_mol=float(np.max(np.abs(scored - reference["energy"][rows])))
                  if "energy" in reference else None,
                  cutoff_shift_mean_kj_mol=float(shift.mean()), cutoff_shift_std_kj_mol=float(shift.std()),
                  cutoff_shift_min_kj_mol=float(shift.min()), cutoff_shift_max_kj_mol=float(shift.max()),
                  cutoff_free_energy_kj_mol=offset,
                  maximum_interatomic_distance_nm=max(float(np.max(np.linalg.norm(x[:, None] - x[None], axis=-1)))
                                                      for x in positions[rows[::max(1, len(rows) // 40)], :, 1:]))
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


def summarize(args):
    """Inclusion rules and statistics of the paper figures and macros, in kcal/mol.

    BAR estimates pass with a BAR overlap >= 0.03 in both legs, forward Zwanzig
    estimates with a forward rESS* >= 0.03 % in both legs. The cutoff offset of
    each system (from --controls) is added to every solvation free energy.
    Peptides are compared with GRAVY, FreeSolv molecules with experiment.
    """
    from scipy.stats import pearsonr

    controls = {c["system"]: c for c in (json.loads(Path(p).read_text()) for p in args.controls or [])}
    experiment = {}
    if args.experimental and args.experimental.suffix == ".json":
        experiment = json.loads(args.experimental.read_text())
    elif args.experimental:  # FreeSolv database.txt: id; SMILES; IUPAC name; experiment (kcal/mol); ...
        lines = args.experimental.read_text().splitlines()
        experiment = {f[0].strip(): float(f[3]) for f in (line.split(";") for line in lines if line.strip() and line[0] != "#")}
    rows, kinds = [], set()
    for path in args.results:
        result = json.loads(Path(path).read_text())
        system, delta = result["system"], result["solvation"]
        if args.controls and system not in controls:
            raise ValueError(f"no cutoff offset for {system}")
        kinds.add(result["kind"])
        control = controls.get(system, {})
        row = dict(system=system, control=control, offset=control.get("cutoff_free_energy_kj_mol", 0.0) / KJ_PER_KCAL,
                   comparison=result["gravy"] if result["kind"] == "peptide" else experiment.get(system))
        for name, gate, threshold in (("bar", "min_bar_overlap", OVERLAP_THRESHOLD),
                                      ("forward", "min_forward_ress", RESS_THRESHOLD)):
            value, error = delta[f"{name}_kj_mol"], delta[f"{name}_se_kj_mol"]
            passes = None not in (value, error) and (delta[gate] or 0.0) >= threshold
            row[name] = value / KJ_PER_KCAL if passes else None  # in the scored potentials
        rows.append(row)

    def statistics(pairs):
        x, y = np.asarray(pairs, dtype=np.float64).reshape(-1, 2).T
        values = dict(n=len(x), pearson_r=float(pearsonr(x, y).statistic) if len(x) >= 3 else None)
        if kinds == {"freesolv"} and len(x):
            values.update(mean_error=float(np.mean(y - x)), mae=float(np.mean(np.abs(y - x))),
                          rmse=float(np.sqrt(np.mean((y - x) ** 2))))
        return values

    summary = dict(n_systems=len(rows))
    md_pass = {r["system"] for r in rows if r["control"].get("md_bar_overlap", 0.0) >= OVERLAP_THRESHOLD}
    for name in ("bar", "forward"):
        passing = [r for r in rows if r[name] is not None and r["comparison"] is not None]
        summary[name] = dict(statistics([(r["comparison"], r[name] + r["offset"]) for r in passing]),
                             systems=sorted(r["system"] for r in passing))
        if md_pass:  # estimator error against the MD-only reference, both in the scored potentials
            errors = [r[name] - r["control"]["md_bar_kj_mol"] / KJ_PER_KCAL for r in passing if r["system"] in md_pass]
            summary[name]["mean_minus_md_reference"] = float(np.mean(errors)) if errors else None
    if md_pass:
        md = [r for r in rows if r["system"] in md_pass and r["comparison"] is not None]
        summary["md_reference"] = dict(
            statistics([(r["comparison"], r["control"]["md_bar_nocutoff_kj_mol"] / KJ_PER_KCAL) for r in md]),
            max_half_difference=max(abs(np.subtract(*r["control"]["md_bar_half_kj_mol"])) for r in md) / KJ_PER_KCAL)
    if controls:
        offsets = [r["offset"] for r in rows]
        summary["cutoff_offset"] = dict(
            min=min(offsets), max=max(offsets), median=float(np.median(offsets)),
            forward_passing_median=float(np.median([r["offset"] for r in rows if r["forward"] is not None])),
            max_shift_std_kj_mol=max(r["control"]["cutoff_shift_std_kj_mol"] for r in rows))
    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, kind in (("freesolv", "FreeSolv molecule"), ("peptide", "peptide")):
        sub = commands.add_parser(name, help=f"both legs of one {kind}")
        if name == "freesolv":
            sub.add_argument("--molecule", required=True, help="FreeSolv id")
            sub.add_argument("--helm", required=True, help="CHEM1{[SMILES]}$$$$ prompt")
            sub.add_argument("--parameters", type=Path, required=True,
                             help="directory with lig.prmtop for both legs; created when missing")
            sub.add_argument("--reference-frames", type=int, default=2048, help="frames per leg over all files")
            sub.add_argument("--discard-fraction", type=float, default=0.2, help="equilibration share of each file")
        else:
            sub.add_argument("--sequence", required=True)
            sub.add_argument("--vacuum-reference-frames", type=int, default=20000)
            sub.add_argument("--water-reference-frames", type=int, default=10000)
            sub.add_argument("--water-temperature", type=float, default=310.0,
                             help="prompt temperature of the water proposal and its 310 K reference MD")
        for state in ("vacuum", "water"):
            sub.add_argument(f"--{state}-reference", type=Path, nargs="+", required=True,
                             help="reference MD NPZ files (positions, smiles; peptides: energy)")
            sub.add_argument(f"--{state}-proposals", type=Path, nargs="+",
                             help="proposal NPZ files (positions, generator_log_likelihood|logq)")
            sub.add_argument(f"--{state}-reference-logq", type=Path, nargs="+",
                             help="NPZ files with the model log-likelihood (logq) of the selected reference frames")
        sub.add_argument("--checkpoint", type=Path, help="PBG checkpoint; required unless all log-likelihoods are given")
        sub.add_argument("--proposals", type=int, default=10000)
        sub.add_argument("--seed", type=int, default=0, help="sampling seed")
        sub.add_argument("--bootstrap", type=int, default=200)
        sub.add_argument("--bar-seed", type=int, default=20260831)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.set_defaults(handler=freesolv if name == "freesolv" else peptide)
    sub = commands.add_parser("md-reference", help="MD-only BAR reference of one FreeSolv molecule")
    sub.add_argument("--molecule", required=True)
    sub.add_argument("--parameters", type=Path, required=True, help="directory with lig.prmtop and lig.inpcrd")
    sub.add_argument("--equilibration-ps", type=float, default=200.0)
    sub.add_argument("--production-ns", type=float, default=5.0)
    sub.add_argument("--frames", type=int, default=2000)
    sub.add_argument("--seed", type=int, default=20261123)
    sub.add_argument("--output-dir", type=Path, required=True)
    sub.set_defaults(handler=md_reference_command)
    sub = commands.add_parser("cutoff-offset", help="cutoff offset of the implicit-water protocol")
    sub.add_argument("--system", required=True)
    sub.add_argument("--reference", type=Path, nargs="+", required=True,
                     help="implicit-water frames in topology order (peptides: ff14SB order)")
    topology = sub.add_mutually_exclusive_group(required=True)
    topology.add_argument("--sequence")
    topology.add_argument("--prmtop", type=Path)
    sub.add_argument("--target", choices=("ff14sb-obc1-2nm", "gaff2-am1bcc-obc1-2nm"))
    sub.add_argument("--frames", type=int, default=2000)
    sub.add_argument("--reference-temperature", type=float, help="sampling temperature of the frames (default --temperature)")
    sub.add_argument("--output", type=Path, required=True)
    sub.set_defaults(handler=cutoff_offset_command)
    sub = commands.add_parser("summarize", help="pass counts and correlations of many systems")
    sub.add_argument("--results", type=Path, nargs="+", required=True, help="result.json of freesolv or peptide runs")
    sub.add_argument("--controls", type=Path, nargs="+", help="md-reference or cutoff-offset JSON files")
    sub.add_argument("--experimental", type=Path, help="FreeSolv database.txt or JSON {id: kcal/mol}")
    sub.add_argument("--output", type=Path)
    sub.set_defaults(handler=summarize)
    for sub in commands.choices.values():
        sub.add_argument("--temperature", type=float, default=300.0, help="target temperature of every estimate")
        sub.add_argument("--platform", default=os.environ.get("PBG_OPENMM_PLATFORM", "CPU"), help="OpenMM platform")
    args = parser.parse_args()
    if getattr(args, "output_dir", None):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    args.handler(args)


if __name__ == "__main__":
    main()
