"""OpenMM energies of the evaluation protocols in ``pbg.core.environment.PROTOCOLS``.

A molecule is parameterized once, never per frame. Peptides use the ff14SB
topology of their sequence followed by OHH waters (``ProtocolEvaluator.peptide``).
Small molecules use one durable GAFF2/AM1-BCC prmtop for every state
(``parameterize_gaff``). Coordinates are nm, energies kJ/mol.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np

from pbg.core.environment import PROTOCOLS
from pbg.core.peptide import sequence_to_pdb_block, sequence_to_smiles


def expected_solvated_smiles(sequence, n_waters):
    """Mapped explicit-H SMILES of ``sequence`` followed by ``n_waters`` OHH waters."""
    smiles = sequence_to_smiles(sequence)
    if n_waters == 0:
        return smiles
    start = max(int(index) for index in re.findall(r":(\d+)]", smiles)) + 1
    waters = (f"[O:{i}]([H:{i + 1}])[H:{i + 2}]" for i in range(start, start + 3 * n_waters, 3))
    return ".".join((smiles, *waters))


def parameterize_gaff(mapped_smiles, reference_zxyz_nm, directory):
    """Write GAFF2/AM1-BCC parameters of one mapped molecule to ``directory``; return its lig.prmtop.

    Antechamber assigns GAFF2 types and AM1-BCC charges on the reference
    conformer, then parmchk2 and tleap write lig.prmtop and lig.inpcrd in the
    atom-map order of ``mapped_smiles``. A parameters.json with SHA-256 sums
    marks a finished directory; later calls only check the SMILES and the sums.
    """
    from openmm import app, unit
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Geometry import Point3D

    files = ("lig.mol2", "lig.frcmod", "lig.prmtop", "lig.inpcrd")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "parameters.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["mapped_smiles"] != mapped_smiles:
            raise ValueError(f"{directory} holds parameters of {saved['mapped_smiles']!r}, not {mapped_smiles!r}")
        for name, digest in saved["sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"checksum mismatch of {directory / name}")
        return directory / "lig.prmtop"

    reference = np.asarray(reference_zxyz_nm, dtype=np.float64)
    mol = Chem.MolFromSmiles(mapped_smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"could not parse mapped SMILES: {mapped_smiles!r}")
    Chem.SanitizeMol(mol)
    maps = np.asarray([atom.GetAtomMapNum() for atom in mol.GetAtoms()], dtype=np.int64)
    if set(maps.tolist()) != set(range(1, mol.GetNumAtoms() + 1)):
        raise ValueError("mapped SMILES must use contiguous atom maps 1..N")
    mol = rdmolops.RenumberAtoms(mol, np.argsort(maps).tolist())
    if reference.shape != (mol.GetNumAtoms(), 4):
        raise ValueError(f"reference frame has shape {reference.shape}, expected {(mol.GetNumAtoms(), 4)}")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, xyz in enumerate(reference[:, 1:4]):
        conformer.SetAtomPosition(index, Point3D(*(float(value) * 10.0 for value in xyz)))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)

    with tempfile.TemporaryDirectory(prefix="pbg_gaff_") as temporary:
        work = Path(temporary)
        writer = Chem.SDWriter(str(work / "lig.sdf"))
        writer.write(mol)
        writer.close()
        (work / "leap.in").write_text(
            "source leaprc.gaff2\nMOL = loadmol2 lig.mol2\nloadamberparams lig.frcmod\n"
            "saveamberparm MOL lig.prmtop lig.inpcrd\nquit\n", encoding="utf-8")
        for command in (
            ["antechamber", "-i", "lig.sdf", "-fi", "sdf", "-o", "lig.mol2", "-fo", "mol2", "-c", "bcc",
             "-at", "gaff2", "-pf", "y"],
            ["parmchk2", "-i", "lig.mol2", "-f", "mol2", "-o", "lig.frcmod", "-s", "2"],
            ["tleap", "-f", "leap.in"],
        ):
            result = subprocess.run(command, cwd=work, text=True, capture_output=True, check=False, timeout=240)
            if result.returncode:
                raise RuntimeError(f"{command[0]} failed: {(result.stderr or result.stdout)[-3000:]}")
        prmtop = app.AmberPrmtopFile(str(work / "lig.prmtop"))
        expected = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()}
        actual = {tuple(sorted((a.index, b.index))) for a, b in prmtop.topology.bonds()}
        elements = [atom.element.atomic_number for atom in prmtop.topology.atoms()]
        if expected != actual or not np.array_equal(elements, reference[:, 0]):
            raise ValueError("GAFF parameterization changed the atom order")
        xyz = app.AmberInpcrdFile(str(work / "lig.inpcrd")).positions.value_in_unit(unit.nanometer)
        np.testing.assert_allclose(xyz, reference[:, 1:], atol=1e-4, rtol=0)
        for name in files:
            shutil.copyfile(work / name, directory / f"{name}.tmp")
            (directory / f"{name}.tmp").replace(directory / name)
    record = {"mapped_smiles": mapped_smiles, "forcefield": "gaff2-am1bcc",
              "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}
    marker.with_suffix(".tmp").write_text(json.dumps(record, indent=2) + "\n")
    marker.with_suffix(".tmp").replace(marker)
    return directory / "lig.prmtop"


def create_system(target_id, topology=None, *, prmtop=None):
    """OpenMM System of one protocol.

    ff14SB protocols take an OpenMM topology. GAFF2 protocols take the
    AmberPrmtopFile of ``parameterize_gaff`` in the atom order of the scored
    frames, with exactly the scored TIP3P waters for shells. The vacuum
    protocol ``ff14sb`` keeps the MD constraints on bonds to hydrogen, so
    OpenMM leaves out their stretch terms; every other protocol keeps all
    bond terms.
    """
    import openmm
    from openmm import app, unit

    target = PROTOCOLS[target_id]
    if target_id == "ff14sb-obc1-1nm":
        raise ValueError("pbg does not score cyclic peptides (ff14sb-obc1-1nm)")
    kwargs = dict(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
    if target.cutoff_nm is not None:
        kwargs.update(nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=target.cutoff_nm * unit.nanometer)
    if target.forcefield == "gaff2-am1bcc":
        if prmtop is None:
            raise ValueError("GAFF evaluation requires a fixed GAFF2/AM1-BCC AmberPrmtopFile")
        if target.solvent == "implicit_water":
            kwargs.update(implicitSolvent=app.OBC1, soluteDielectric=1.0, solventDielectric=78.5, sasaMethod="ACE")
        system = prmtop.createSystem(**kwargs)
    else:
        files = ["amber14-all.xml"]
        if target.solvent == "implicit_water":
            files.append("implicit/obc1.xml")
        elif target.solvent == "explicit_water":
            files.append("amber14/tip3p.xml")
        else:
            kwargs["constraints"] = app.HBonds
        system = app.ForceField(*files).createSystem(topology, **kwargs)
    if target.solvent == "explicit_water":
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setReactionFieldDielectric(target.dielectric)
                force.setUseSwitchingFunction(False)
    return system


class ProtocolEvaluator:
    """Score frames with one fixed protocol System.

    Frames are nm coordinates ``[frames, atoms, 3]``, or Zxyz ``[frames, atoms, 4]``
    when the evaluator knows its atomic numbers (``ProtocolEvaluator.peptide``);
    their Z column must then match the topology. Frames with non-finite
    coordinates, or that OpenMM cannot score, get NaN.
    """

    def __init__(self, system, platform_name=None, atomic_numbers=None):
        import openmm
        from openmm import unit

        self.integrator = openmm.VerletIntegrator(1 * unit.femtosecond)
        self.context = openmm.Context(system, self.integrator,
                                      openmm.Platform.getPlatformByName(platform_name or "CPU"))
        self.n_atoms = system.getNumParticles()
        self.atomic_numbers = atomic_numbers

    @classmethod
    def peptide(cls, sequence, target_id, *, n_waters=0, platform_name=None):
        """Evaluator of ``sequence`` (ff14SB atom order) followed by ``n_waters`` OHH waters."""
        from openmm import app

        topology = app.PDBFile(StringIO(sequence_to_pdb_block(sequence))).topology
        chain = topology.addChain()
        for _ in range(n_waters):
            residue = topology.addResidue("HOH", chain)
            oxygen = topology.addAtom("O", app.element.oxygen, residue)
            for name in ("H1", "H2"):
                topology.addBond(oxygen, topology.addAtom(name, app.element.hydrogen, residue))
        atomic_numbers = np.array([atom.element.atomic_number for atom in topology.atoms()])
        return cls(create_system(target_id, topology), platform_name, atomic_numbers)

    def __call__(self, frames, *, forces=False):
        """Energies (kJ/mol), and with ``forces`` also forces (kJ/(mol nm)), of each frame."""
        import openmm
        from openmm import unit

        values = np.asarray(frames, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != self.n_atoms or values.shape[2] not in (3, 4):
            raise ValueError(f"frames must have shape [frames, {self.n_atoms}, 3 or 4], got {values.shape}")
        if values.shape[2] == 4:
            if self.atomic_numbers is None or not np.array_equal(
                    values[:, :, 0], np.broadcast_to(self.atomic_numbers, values.shape[:2])):
                raise ValueError("the Z column of the frames does not match the topology")
            values = values[:, :, 1:]
        energy = np.full(len(values), np.nan)
        force = np.full(values.shape, np.nan)
        for index, xyz in enumerate(values):
            if not np.isfinite(xyz).all():
                continue
            self.context.setPositions(xyz * unit.nanometer)
            try:
                state = self.context.getState(getEnergy=True, getForces=forces)
            except openmm.OpenMMException:
                continue
            energy[index] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if forces:
                force[index] = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        return (energy, force) if forces else energy
