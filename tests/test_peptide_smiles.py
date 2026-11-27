"""Linear peptide SMILES in ff14SB atom order, and peptide HELM."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pytest

from pbg.core.constants import ONE_TO_THREE
from pbg.core.molecule import mapped_mol
from pbg.core.peptide import (
    _build_linear_peptide_mol,
    _load_ff14sb_templates,
    _normalize_atom_name,
    _ordered_atom_indices,
    infer_linear_peptide_helm_from_smiles,
    infer_linear_peptide_sequence_from_smiles,
    resolve_linear_peptide_helm,
    sequence_to_helm,
    sequence_to_pdb_block,
    sequence_to_smiles,
)

_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_ELEMENT_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
# Every amino acid at the N-terminal, internal and C-terminal position.
_COVERAGE_SEQUENCES = [
    _AA_ORDER[i] + _AA_ORDER[(i + 1) % 20] + _AA_ORDER[(i + 2) % 20] for i in range(20)
]
_WIC_PDB_ATOM_NAMES = [
    "N", "H1", "H2", "H3", "CA", "HA", "CB", "HB2", "HB3", "CG", "CD1", "HD1", "NE1", "HE1",
    "CE2", "CZ2", "HZ2", "CH2", "HH2", "CZ3", "HZ3", "CE3", "HE3", "CD2", "C", "O",
    "N", "H", "CA", "HA", "CB", "HB", "CG2", "HG21", "HG22", "HG23", "CG1", "HG12", "HG13",
    "CD1", "HD11", "HD12", "HD13", "C", "O",
    "N", "H", "CA", "HA", "CB", "HB2", "HB3", "SG", "HG", "C", "O", "OXT",
]


def _expected_xml_entries(sequence: str) -> list[tuple[int, str, str, str | None]]:
    """(residue, atom name, element, hydrogen anchor) in ff14SB template order."""
    templates = _load_ff14sb_templates()
    entries: list[tuple[int, str, str, str | None]] = []
    n_residues = len(sequence)
    for residue_idx, base in enumerate(sequence, start=1):
        residue_name = ONE_TO_THREE[base]
        if n_residues == 1:
            n_template = templates[f"N{residue_name}"]
            c_template = templates[f"C{residue_name}"]
            atom_names = list(n_template.atom_names)
            atom_names.extend(
                atom_name
                for atom_name in c_template.atom_names
                if atom_name not in n_template.atom_names and c_template.atom_elements[atom_name] != "H"
            )
            atom_elements = dict(c_template.atom_elements)
            atom_elements.update(n_template.atom_elements)
            hydrogen_anchors = n_template.hydrogen_anchors
        else:
            prefix = "N" if residue_idx == 1 else ("C" if residue_idx == n_residues else "")
            template = templates[f"{prefix}{residue_name}"]
            atom_names = list(template.atom_names)
            atom_elements = template.atom_elements
            hydrogen_anchors = template.hydrogen_anchors
        for atom_name in atom_names:
            entries.append((residue_idx, atom_name, atom_elements[atom_name], hydrogen_anchors.get(atom_name)))
    return entries


@pytest.mark.parametrize("sequence", _COVERAGE_SEQUENCES)
def test_atom_order_follows_the_ff14sb_templates(sequence: str) -> None:
    expected = _expected_xml_entries(sequence)
    mol = _build_linear_peptide_mol(sequence)
    order = _ordered_atom_indices(mol, list(sequence))

    assert len(order) == len(expected)
    for atom_idx, (residue_idx, atom_name, element, anchor) in zip(order, expected, strict=True):
        atom = mol.GetAtomWithIdx(atom_idx)
        assert atom.GetSymbol() == element
        # A hydrogen is identified by the heavy atom that carries it.
        named = atom.GetNeighbors()[0] if element == "H" else atom
        assert named.GetPDBResidueInfo().GetResidueNumber() == residue_idx
        assert _normalize_atom_name(named) == (anchor if element == "H" else atom_name)


@pytest.mark.parametrize("sequence", _COVERAGE_SEQUENCES)
def test_sequence_to_smiles_has_the_ff14sb_atomic_numbers(sequence: str) -> None:
    mol = mapped_mol(sequence_to_smiles(sequence))

    actual = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]
    expected = [_ELEMENT_TO_Z[element] for _, _, element, _ in _expected_xml_entries(sequence)]
    np.testing.assert_array_equal(actual, expected)


def test_wic_atom_order_matches_the_templates_and_the_pdb_block() -> None:
    openmm_app = pytest.importorskip("openmm.app")
    assert [name for _, name, _, _ in _expected_xml_entries("WIC")] == _WIC_PDB_ATOM_NAMES

    pdb = openmm_app.PDBFile(StringIO(sequence_to_pdb_block("WIC")))

    expected = list(_WIC_PDB_ATOM_NAMES)
    expected[1] = "H"
    assert [atom.name for atom in pdb.topology.atoms()] == expected


def test_single_residue_smiles_maps_every_atom_in_order() -> None:
    mol = mapped_mol(sequence_to_smiles("A"))

    assert mol.GetNumAtoms() == 13
    assert [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in range(13)] == list(range(1, 14))


def test_sequence_to_smiles_rejects_d_amino_acids() -> None:
    with pytest.raises(NotImplementedError, match="D-amino acids"):
        sequence_to_smiles("AdA")


@pytest.mark.parametrize("sequence", ["A", "AG", "WIC", "ANKSMIEA"])
def test_sequence_round_trips_through_smiles_and_helm(sequence: str) -> None:
    helm = f"PEPTIDE1{{{'.'.join(sequence)}}}$$$$"
    smiles = sequence_to_smiles(sequence)

    assert sequence_to_helm(sequence) == helm
    assert infer_linear_peptide_sequence_from_smiles(smiles) == sequence
    assert infer_linear_peptide_helm_from_smiles(smiles) == helm


def test_inference_ignores_water_in_solvated_smiles() -> None:
    smiles = f"{sequence_to_smiles('ANKSMIEA')}.[O:122]([H:123])[H:124]"

    assert infer_linear_peptide_sequence_from_smiles(smiles) == "ANKSMIEA"
    assert infer_linear_peptide_helm_from_smiles(smiles) == sequence_to_helm("ANKSMIEA")


def test_helm_writes_d_residues() -> None:
    assert sequence_to_helm("dAdTG") == "PEPTIDE1{[dA].[dT].G}$$$$"


@pytest.mark.parametrize("field", ["helm", "sequence"])
def test_frame_prompt_normalizes_dataset_cyclic_notation(field) -> None:
    result = resolve_linear_peptide_helm(**{field: "[#HeadTailLactam][G][P][A][dL]"})

    assert result == "PEPTIDE1{G.P.A.[dL]}$PEPTIDE1,PEPTIDE1,1:R1-4:R2$$$"
