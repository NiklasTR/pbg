"""OpenMM energies of the evaluation protocols and the GAFF2/AM1-BCC parameters."""

from __future__ import annotations

import shutil
from io import StringIO

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from openmm import app, unit

from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_pdb_block
from pbg.energy import (
    ProtocolEvaluator,
    create_system,
    expected_solvated_smiles,
    parameterize_gaff,
)


def test_evaluation_protocols_set_the_published_openmm_options() -> None:
    pdb = app.PDBFile(StringIO(sequence_to_pdb_block("AG")))
    implicit = create_system("ff14sb-obc1-2nm", pdb.topology)
    nonbonded = next(force for force in implicit.getForces() if isinstance(force, openmm.NonbondedForce))
    assert nonbonded.getNonbondedMethod() == openmm.NonbondedForce.CutoffNonPeriodic
    assert nonbonded.getCutoffDistance().value_in_unit(unit.nanometer) == 2.0
    assert implicit.getNumConstraints() == 0

    class Prmtop:
        def createSystem(self, **kwargs):
            self.kwargs = kwargs
            system = openmm.System()
            system.addForce(openmm.NonbondedForce())
            return system

    prmtop = Prmtop()
    shell = create_system("gaff2-am1bcc-tip3p-rf2nm", prmtop=prmtop)
    assert prmtop.kwargs["rigidWater"] is False
    assert prmtop.kwargs["nonbondedMethod"] == app.CutoffNonPeriodic
    assert shell.getForce(0).getReactionFieldDielectric() == 78.5


def test_only_the_vacuum_peptide_protocol_drops_the_stretch_terms_of_bonds_to_hydrogen() -> None:
    topology = app.PDBFile(StringIO(sequence_to_pdb_block("AG"))).topology
    hydrogens = {atom.index for atom in topology.atoms() if atom.element.symbol == "H"}
    n_xh = sum(a.index in hydrogens or b.index in hydrogens for a, b in topology.bonds())

    for target_id, n_constraints in (("ff14sb", n_xh), ("ff14sb-obc1-2nm", 0), ("ff14sb-tip3p-rf2nm", 0)):
        system = create_system(target_id, topology)
        bonds = next(force for force in system.getForces() if isinstance(force, openmm.HarmonicBondForce))
        assert system.getNumConstraints() == n_constraints
        assert bonds.getNumBonds() == topology.getNumBonds() - n_constraints


def test_peptide_evaluator_follows_the_solvated_smiles_and_marks_failed_frames() -> None:
    z = atom_numbers_from_smiles(expected_solvated_smiles("AA", 2)).astype(np.float64)
    evaluator = ProtocolEvaluator.peptide("AA", "ff14sb-tip3p-rf2nm", n_waters=2)
    np.testing.assert_array_equal(evaluator.atomic_numbers, z)

    grid = 0.3 * np.stack(np.unravel_index(np.arange(z.size), (4, 4, 4)), axis=-1)
    frames = np.zeros((3, z.size, 4))
    frames[:, :, 0] = z
    frames[:, :, 1:] = grid + np.random.default_rng(0).normal(scale=0.01, size=(3, z.size, 3))
    frames[2, 0, 1] = np.nan
    energy, force = evaluator(frames, forces=True)

    assert np.isfinite(energy[:2]).all() and np.isnan(energy[2])
    assert force.shape == (3, z.size, 3) and np.isfinite(force[:2]).all() and np.isnan(force[2]).all()
    np.testing.assert_array_equal(evaluator(frames[:, :, 1:]), energy)
    wrong = frames.copy()
    wrong[:, -1, 0] = 8.0
    with pytest.raises(ValueError, match="Z column"):
        evaluator(wrong)


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="needs AmberTools on PATH")
def test_gaff_parameters_are_written_once_in_atom_map_order(tmp_path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol)
    mol = Chem.RenumberAtoms(mol, np.argsort([atom.GetAtomMapNum() for atom in mol.GetAtoms()]).tolist())
    AllChem.EmbedMolecule(mol, randomSeed=0)
    reference = np.column_stack(([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                                 mol.GetConformer().GetPositions() / 10.0))

    prmtop = parameterize_gaff(smiles, reference, tmp_path)
    elements = [atom.element.atomic_number for atom in app.AmberPrmtopFile(str(prmtop)).topology.atoms()]
    assert elements == [6, 8, 1, 1, 1, 1]
    assert parameterize_gaff(smiles, reference, tmp_path) == prmtop
    with pytest.raises(ValueError, match="holds parameters"):
        parameterize_gaff("[O:1]([H:2])[H:3]", reference, tmp_path)
    (tmp_path / "lig.frcmod").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        parameterize_gaff(smiles, reference, tmp_path)
