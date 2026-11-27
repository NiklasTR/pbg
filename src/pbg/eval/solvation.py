"""Solvation free energies from zero-free-energy proposals.

Paper, Appendix "Solvation Free Energy Estimation":

* "Zero Free Energy Proposals": a normalized proposal q_theta is a state with
  zero free energy, so the forward Zwanzig and BAR estimates between q_theta
  and the Boltzmann density of one physical state give that state's absolute
  free energy. The solvation free energy is F(water) - F(vacuum) of two
  independent legs.
* Evaluation Metrics, "BAR bridge weight": the overlap O_BAR of one leg. A
  system passes when both legs have O_BAR >= 0.03 (BAR) or a forward rESS* of
  at least 0.03 % (forward Zwanzig).
* "Cutoff Offset of the Implicit-Water Energies": the constant free-energy
  offset F(OBC1 without cutoff) - F(scored OBC1 with 2 nm cutoff), and the
  MD-only BAR reference in the scored potentials.

Command lines: ``scripts/solvation_free_energy.py`` and
``scripts/evaluate_small_molecules.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.estimator import bar, zwanzig

__all__ = [
    "OVERLAP_THRESHOLD",
    "RESS_THRESHOLD",
    "atom_rows",
    "cutoff_offset",
    "md_reference",
    "md_reference_estimate",
    "model_energy",
    "read_frames",
    "reference_selection",
    "scored_system",
    "solvation_estimate",
    "zfep_leg",
]

OVERLAP_THRESHOLD = 0.03  # minimum BAR overlap of each leg (Klimovich et al., 2015)
RESS_THRESHOLD = 3e-4  # minimum forward rESS* of each leg (0.03 %)


def read_frames(paths, *, smiles=None, rows=None):
    """Concatenate the per-frame arrays of one or more NPZ files.

    Returns ``positions`` as Zxyz in nm (atomic numbers come from the file's
    ``smiles`` or from ``smiles`` when a file stores xyz only), ``logq``
    (stored as ``generator_log_likelihood`` or ``logq``), ``energy`` when every
    file has it, ``lengths`` per file, and the scalar entries of the first file.
    ``rows(file_smiles)`` reorders the atoms of each file, for example with
    ``atom_rows`` into the model atom order.
    """
    from pbg.core.molecule import atom_numbers_from_smiles

    result, parts, lengths = {}, {}, []
    for path in map(Path, paths):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if data[key].ndim == 0:
                    result.setdefault(key, data[key].item())
            source = str(data["smiles"].item()) if "smiles" in data.files else smiles
            for key in ("positions", "energy", "generator_log_likelihood", "logq"):
                if key not in data.files:
                    continue
                value = np.asarray(data[key], dtype=np.float64)
                if key == "positions":
                    if value.shape[-1] == 3:
                        z = atom_numbers_from_smiles(source).astype(np.float64)
                        value = np.concatenate([np.broadcast_to(z[None, :, None], (*value.shape[:2], 1)), value], axis=2)
                    if rows is not None:
                        value = value[:, np.asarray(rows(source))]
                else:
                    value = value.reshape(-1)
                parts.setdefault("logq" if key == "generator_log_likelihood" else key, []).append(value)
            lengths.append(len(value))
    for key, values in parts.items():
        if len(values) == len(lengths):
            result[key] = np.concatenate(values)
    result["lengths"] = lengths
    return result


def reference_selection(lengths, budget, discard_fraction=0.2):
    """Indices of up to ``budget`` reference frames balanced over trajectories of ``lengths`` frames.

    Each trajectory drops its first ``discard_fraction`` of frames. The budget is dealt out one frame
    per trajectory in turn, and the frames of each trajectory are evenly spaced over its kept part.
    Indices count into the concatenated trajectories.
    """
    lengths = np.asarray(lengths, dtype=int)
    starts = np.floor(discard_fraction * lengths).astype(int)
    available = lengths - starts
    if budget <= 0 or np.any(available <= 0):
        raise ValueError("empty trajectory or invalid reference budget")
    take = np.zeros(len(lengths), dtype=int)
    remaining = min(int(budget), int(available.sum()))
    while remaining:
        chosen = np.flatnonzero(take < available)[:remaining]
        take[chosen] += 1
        remaining -= len(chosen)
    offsets = np.r_[0, np.cumsum(lengths)]
    return np.concatenate([
        offsets[index] + np.linspace(starts[index], lengths[index] - 1, count, dtype=int)
        for index, count in enumerate(take) if count
    ])


def scored_system(target_id, *, prmtop=None, sequence=None, cutoff=True):
    """OpenMM System of a registered energy protocol (``pbg.core.environment``).

    The topology is a GAFF2/AM1-BCC ``prmtop`` file or a peptide ``sequence``
    (ff14SB atom order). ``cutoff=False`` rebuilds an implicit-water protocol
    as the same OBC1 model without the 2 nm cutoff.
    """
    from io import StringIO

    from openmm import app

    from pbg.core.environment import PROTOCOLS
    from pbg.core.peptide import sequence_to_pdb_block
    from pbg.energy import create_system

    topology = None if sequence is None else app.PDBFile(StringIO(sequence_to_pdb_block(sequence))).topology
    parameters = None if prmtop is None else app.AmberPrmtopFile(str(prmtop))
    if cutoff:
        return create_system(target_id, topology, prmtop=parameters)
    if PROTOCOLS[target_id].solvent != "implicit_water":
        raise ValueError(f"{target_id} has no implicit-water cutoff to remove")
    options = dict(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
    if parameters is not None:
        return parameters.createSystem(implicitSolvent=app.OBC1, soluteDielectric=1.0,
                                       solventDielectric=78.5, sasaMethod="ACE", **options)
    return app.ForceField("amber14-all.xml", "implicit/obc1.xml").createSystem(topology, **options)


def atom_rows(target_smiles, source_smiles, helm=None, environment="vacuum"):
    """Rows that reorder frames from the atom order of ``source_smiles`` into that of ``target_smiles``.

    Both orders are matched to the prompt topology of ``helm`` and
    ``environment`` (``PromptTopology.source_rows``), as for the paper rollouts.
    """
    from pbg.core.molecule import atom_numbers_from_smiles
    from pbg.core.topology import PromptTopology

    if target_smiles == source_smiles:
        return np.arange(atom_numbers_from_smiles(target_smiles).size)
    topology = PromptTopology.compile(helm, environment)
    return np.asarray(topology.source_rows(source_smiles))[np.argsort(topology.source_rows(target_smiles))]


def model_energy(target_id, smiles, helm=None, environment="vacuum", *, prmtop=None, sequence=None, cutoff=True,
                 platform="CPU"):
    """Return a function that scores Zxyz frames in the atom order of ``smiles`` in kJ/mol.

    A GAFF prmtop must share that atom order. Peptide frames are reordered to
    the ff14SB atom order of ``sequence`` before scoring.
    """
    from openmm import app

    from pbg.core.molecule import atom_numbers_from_smiles
    from pbg.core.peptide import sequence_to_smiles
    from pbg.energy import ProtocolEvaluator

    if sequence is None:
        order = slice(None)
        elements = [atom.element.atomic_number for atom in app.AmberPrmtopFile(str(prmtop)).topology.atoms()]
        if elements != atom_numbers_from_smiles(smiles).tolist():
            raise ValueError(f"{prmtop} does not follow the atom order of {smiles}")
    else:
        order = atom_rows(sequence_to_smiles(sequence), smiles, helm, environment)
    evaluator = ProtocolEvaluator(scored_system(target_id, prmtop=prmtop, sequence=sequence, cutoff=cutoff),
                                  platform_name=platform)
    return lambda frames: evaluator(np.asarray(frames, dtype=np.float64)[:, order, 1:4])


def zfep_leg(proposal_logq, proposal_energy, reference_logq, reference_energy, *,
             temperature_K=300.0, bootstrap=200, seed=20260831):
    """Absolute free energy of one physical state from its zero-free-energy proposal.

    Forward Zwanzig uses the proposal log weights ``-beta U - log q``; BAR adds
    reference MD frames with reverse log weights ``beta U + log q``. Frames with
    non-finite energy or log-likelihood are dropped. Free energies in kJ/mol;
    ``forward_ress`` is the unclipped Kish ESS over the retained proposals.
    """
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * temperature_K)
    proposal = beta * np.asarray(proposal_energy, np.float64) + np.asarray(proposal_logq, np.float64)
    reference = beta * np.asarray(reference_energy, np.float64) + np.asarray(reference_logq, np.float64)
    proposal, reference = proposal[np.isfinite(proposal)], reference[np.isfinite(reference)]
    if not proposal.size or not reference.size:
        raise ValueError("no finite proposal or reference weights")
    forward = zwanzig(-proposal, temperature_K=temperature_K, clip_percentile=None)
    two_sided = bar(-proposal, reference, temperature_K=temperature_K, bootstrap=bootstrap, seed=seed)
    values = dict(n_proposal=int(proposal.size), n_reference=int(reference.size),
                  forward_kj_mol=forward.delta_f_kj_mol, forward_se_kj_mol=forward.std_kj_mol,
                  forward_ress=forward.ress, forward_max_weight_fraction=forward.max_weight_fraction,
                  bar_kj_mol=two_sided.delta_f_kj_mol, bar_se_kj_mol=two_sided.conservative_std_kj_mol,
                  bar_mbar_se_kj_mol=two_sided.mbar_std_kj_mol, bar_overlap=two_sided.overlap,
                  bar_bridge_ess_fraction=two_sided.bridge_ess_fraction)
    return {key: None if value is None or not np.isfinite(value) else value for key, value in values.items()}


def solvation_estimate(vacuum, water):
    """Combine two independent legs: Delta F = F(water) - F(vacuum), errors in quadrature."""
    result = {}
    for estimator in ("forward", "bar"):
        a, b = vacuum[f"{estimator}_kj_mol"], water[f"{estimator}_kj_mol"]
        sa, sb = vacuum[f"{estimator}_se_kj_mol"], water[f"{estimator}_se_kj_mol"]
        result[f"{estimator}_kj_mol"] = None if a is None or b is None else b - a
        result[f"{estimator}_se_kj_mol"] = None if sa is None or sb is None else float(np.hypot(sa, sb))
    for key in ("bar_overlap", "forward_ress"):
        pair = (vacuum[key], water[key])
        result[f"min_{key}"] = None if None in pair else min(pair)
    return result


def cutoff_offset(scored, uncut, *, temperature_K=300.0, sample_temperature_K=None):
    """F(no cutoff) - F(scored cutoff protocol) in kJ/mol from frames of the scored state.

    ``scored`` and ``uncut`` are energies of the same frames. Frames drawn at
    ``sample_temperature_K`` are reweighted to ``temperature_K`` first.
    """
    from scipy.special import logsumexp

    rt = GAS_CONSTANT_KJ_MOL_K * temperature_K
    scored, uncut = np.asarray(scored, np.float64), np.asarray(uncut, np.float64)
    sample_rt = GAS_CONSTANT_KJ_MOL_K * (sample_temperature_K or temperature_K)
    log_weight = -(1.0 / rt - 1.0 / sample_rt) * scored
    return float(-rt * (logsumexp(log_weight - (uncut - scored) / rt) - logsumexp(log_weight)))


def md_reference(parameters, *, equilibration_ps=200.0, production_ns=5.0, frames=2000, seed=20261123,
                 temperature_K=300.0, platform="CPU"):
    """Langevin MD of both scored FreeSolv potentials and the energies for BAR.

    ``parameters`` holds the shared ``lig.prmtop`` and ``lig.inpcrd``. Vacuum
    (``gaff2-am1bcc``) and implicit water (``gaff2-am1bcc-obc1-2nm``) are each
    sampled with bonds to hydrogen constrained to their GAFF2 lengths (2 fs,
    friction 1/ps). Returns energies keyed ``(frames state, potential)`` with
    potentials ``vacuum``, ``implicit`` and ``implicit_nocutoff``, and the
    largest interatomic distance seen (nm).
    """
    import openmm
    from openmm import app, unit

    from pbg.energy import ProtocolEvaluator

    prmtop = Path(parameters) / "lig.prmtop"
    start = app.AmberInpcrdFile(str(Path(parameters) / "lig.inpcrd")).getPositions(asNumpy=True)
    hydrogens = {atom.index for atom in app.AmberPrmtopFile(str(prmtop)).topology.atoms() if atom.element.symbol == "H"}
    targets = {"vacuum": "gaff2-am1bcc", "implicit": "gaff2-am1bcc-obc1-2nm"}
    stride = int(round(production_ns * 1000.0 / 0.002 / frames))
    trajectories = {}
    for offset, (state, target) in enumerate(targets.items()):
        system = scored_system(target, prmtop=prmtop)
        bonds = next(force for force in system.getForces() if isinstance(force, openmm.HarmonicBondForce))
        for index in range(bonds.getNumBonds()):
            i, j, length, _ = bonds.getBondParameters(index)
            if i in hydrogens or j in hydrogens:
                system.addConstraint(i, j, length)
        integrator = openmm.LangevinMiddleIntegrator(temperature_K, 1.0, 0.002)
        integrator.setRandomNumberSeed(seed + offset)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform))
        context.setPositions(start)
        openmm.LocalEnergyMinimizer.minimize(context)
        context.setVelocitiesToTemperature(temperature_K, seed + offset)
        integrator.step(int(round(equilibration_ps / 0.002)))
        positions = np.empty((frames, system.getNumParticles(), 3))
        for frame in range(frames):
            integrator.step(stride)
            positions[frame] = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        trajectories[state] = positions
        del context, integrator
    potentials = {name: ProtocolEvaluator(scored_system(target, prmtop=prmtop), platform_name=platform)
                  for name, target in targets.items()}
    potentials["implicit_nocutoff"] = ProtocolEvaluator(
        scored_system(targets["implicit"], prmtop=prmtop, cutoff=False), platform_name=platform)
    energies = {(state, name): potential(positions)
                for state, positions in trajectories.items() for name, potential in potentials.items()}
    distance = max(float(np.max(np.linalg.norm(x[:, None] - x[None], axis=-1)))
                   for positions in trajectories.values() for x in positions[::50])
    return energies, distance


def md_reference_estimate(energies, *, temperature_K=300.0):
    """MD-only BAR solvation free energy and the cutoff offset from ``md_reference`` energies (kJ/mol).

    The overlap is the smaller mean BAR bridge weight of the two MD
    ensembles. ``md_bar_half_kj_mol`` repeats BAR on the first and second
    half of each trajectory.
    """
    rt = GAS_CONSTANT_KJ_MOL_K * temperature_K
    forward = (energies["vacuum", "implicit"] - energies["vacuum", "vacuum"]) / rt
    reverse = (energies["implicit", "vacuum"] - energies["implicit", "implicit"]) / rt
    estimate = bar(-forward, -reverse, temperature_K=temperature_K)
    half = len(forward) // 2
    shift = energies["implicit", "implicit_nocutoff"] - energies["implicit", "implicit"]
    offset = cutoff_offset(energies["implicit", "implicit"], energies["implicit", "implicit_nocutoff"],
                           temperature_K=temperature_K)
    return dict(
        md_bar_kj_mol=estimate.delta_f_kj_mol, md_bar_se_kj_mol=estimate.bar_std_kj_mol,
        md_bar_overlap=estimate.overlap,
        md_bar_half_kj_mol=[bar(-forward[part], -reverse[part], temperature_K=temperature_K).delta_f_kj_mol
                            for part in (slice(None, half), slice(half, None))],
        md_forward_zwanzig_kj_mol=zwanzig(-forward, temperature_K=temperature_K, clip_percentile=None).delta_f_kj_mol,
        md_reverse_zwanzig_kj_mol=-zwanzig(-reverse, temperature_K=temperature_K, clip_percentile=None).delta_f_kj_mol,
        md_bar_nocutoff_kj_mol=estimate.delta_f_kj_mol + offset,
        cutoff_free_energy_kj_mol=offset, cutoff_shift_mean_kj_mol=float(shift.mean()),
        cutoff_shift_std_kj_mol=float(shift.std()))
