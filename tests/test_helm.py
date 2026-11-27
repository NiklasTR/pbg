"""HELM parsing, monomer libraries, cyclic rotation and residue rewrites."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rdkit import Chem

from pbg.core.helm import (
    HELMParseError,
    cyclic_peptide_helm,
    cyclic_peptide_helm_from_smiles,
    load_helm_monomer_library,
    mapped_smiles_to_chem_helm,
    mol_from_helm,
    normalize_peptide_helm,
    rewrite_helm_residues,
    rotate_cyclic_helm,
    split_peptide,
)
from pbg.core.tokeniser import CoreTokenizer, core_vocab

_CUSTOM_HELM = "PEPTIDE1{D.F.E.E.I.[dP].E.E.[*N[C@H](C(=O)*)Cc1ccc(cc1)OS(=O)(=O)O |$_R1;;;;;_R2;;;;;;;;;;;;$|].L.[am]}$$$$"
_OFFICIAL_MONOMERS = (
    ("D", "[H:1]N[C@@H](CC([OH:3])=O)C([OH:2])=O"),
    ("F", "[H:1]N[C@@H](Cc1ccccc1)C([OH:2])=O"),
    ("E", "[H:1]N[C@@H](CCC([OH:3])=O)C([OH:2])=O"),
    ("I", "CC[C@H](C)[C@H](N[H:1])C([OH:2])=O"),
    ("L", "CC(C)C[C@H](N[H:1])C([OH:2])=O"),
    ("dP", "[H:1]N1CCC[C@@H]1C([OH:2])=O"),
    ("am", "N[H:1]"),
)


def _canonical_smiles(mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True)


@pytest.fixture()
def official_helm_json(tmp_path):
    records = [
        {"symbol": symbol, "polymerType": "PEPTIDE", "monomerType": "Backbone", "smiles": smiles}
        for symbol, smiles in _OFFICIAL_MONOMERS
    ]
    path = tmp_path / "HELMCoreLibrary.json"
    path.write_text(json.dumps(records))
    return path


# ── CHEM HELM from mapped SMILES ─────────────────────────────────────


@pytest.mark.parametrize(
    ("smiles", "prune_water", "expected"),
    [
        ("[CH3:2][C@@H:1]([OH:3])[H:4]", False, "CHEM1{[CCO]}$$$$"),
        ("[F:1][C@H:2]([Cl:3])[Br:4]", False, "CHEM1{[F[C@H](Cl)Br]}$$$$"),
        ("[NH4+:1]", False, "CHEM1{[[NH4+]]}$$$$"),
        ("[CH3:1][OH:2].[O:3]([H:4])[H:5].[O:6]([H:7])[H:8]", True, "CHEM1{[CO]}$$$$"),
        ("[O:1]([H:2])[H:3]", True, "CHEM1{[O]}$$$$"),
    ],
    ids=["canonical-no-maps-no-h", "stereo", "charge", "pruned-water", "single-water-solute"],
)
def test_mapped_smiles_to_chem_helm(smiles, prune_water, expected) -> None:
    assert mapped_smiles_to_chem_helm(smiles, prune_water=prune_water) == expected


@pytest.mark.parametrize(("smiles", "match"), [("not smiles", "Could not parse SMILES"), ("  ", "must not be empty")])
def test_mapped_smiles_to_chem_helm_rejects_invalid_smiles(smiles, match) -> None:
    with pytest.raises(ValueError, match=match):
        mapped_smiles_to_chem_helm(smiles)


# ── Monomer libraries ────────────────────────────────────────────────


def test_load_helm_monomer_library_reads_atom_mapped_rgroups(official_helm_json) -> None:
    library = load_helm_monomer_library(official_helm_json)

    assert set(library) == {"PEPTIDE"}
    assert library["PEPTIDE"]["F"].rgroup_atoms == {1: 0, 2: 11}
    assert library["PEPTIDE"]["D"].rgroup_atoms == {1: 0, 2: 8, 3: 5}
    assert library["PEPTIDE"]["am"].rgroup_atoms == {1: 1}


def test_default_helm_library_is_the_official_peptide_core() -> None:
    library = load_helm_monomer_library()

    assert set(library) == {"PEPTIDE"}
    assert len(library["PEPTIDE"]) == 322
    for symbol, _ in _OFFICIAL_MONOMERS:
        assert symbol in library["PEPTIDE"]
    terminal = library["PEPTIDE"]["ac"]
    assert set(terminal.rgroup_atoms) == {2}
    assert terminal.mol.GetAtomWithIdx(terminal.rgroup_atoms[2]).GetAtomMapNum() == 2


# ── Linear peptides ──────────────────────────────────────────────────


def test_mol_from_helm_parses_standard_and_inline_monomers(official_helm_json) -> None:
    library = load_helm_monomer_library(official_helm_json)

    standard = mol_from_helm("PEPTIDE1{D.F.E.E.I.[dP].E.E.L.[am]}$$$$", library)
    custom = mol_from_helm(_CUSTOM_HELM, library)

    assert standard.GetNumAtoms() == 79
    assert "C(=O)N" in _canonical_smiles(standard)
    assert custom.GetNumAtoms() == 95
    assert "OS(=O)(=O)O" in _canonical_smiles(custom)
    assert "*" not in _canonical_smiles(custom)


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{A.G}$$$$",
        "PEPTIDE1{D.F.E.E.I.P.E.E.L.[am]}$$$$",
        "PEPTIDE1{D.F.E.E.I.[dP].E.E.L.[am]}$$$$",
        _CUSTOM_HELM,
    ],
)
def test_mol_from_helm_matches_helmkit_for_supported_peptides(helm: str) -> None:
    helmkit = pytest.importorskip("helmkit")
    ours = mol_from_helm(helm, load_helm_monomer_library())
    theirs = helmkit.Molecule(helm).mol

    assert ours.GetNumAtoms() == theirs.GetNumAtoms()
    assert _canonical_smiles(ours) == _canonical_smiles(theirs)


def test_linear_terminal_rewrites_preserve_chemistry() -> None:
    library = load_helm_monomer_library()
    helm = "PEPTIDE1{G.P.A}$$$$"
    rewritten = rewrite_helm_residues(helm, [1, 2, 3], library)

    assert _canonical_smiles(mol_from_helm(rewritten, library)) == _canonical_smiles(mol_from_helm(helm, library))


def test_duplicate_inline_attachment_rejected() -> None:
    with pytest.raises(HELMParseError, match="Duplicate R1"):
        mol_from_helm("PEPTIDE1{[[*:1]NCC(=O)[*:1]].G}$$$$", load_helm_monomer_library())


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{G..G}$$$$",
        "PEPTIDE1{G.G}$",
        "PEPTIDE1{G.G}$PEPTIDE1,PEPTIDE1,1:R1-2:R2",
        "PEPTIDE1{G.[G}$$$$",
        "PEPTIDE1{G.G}$$group$$",
        "PEPTIDE1{G.G}$$$annotation$",
        "PEPTIDE1{G.G}$$$$V3.0",
        "PEPTIDE1{G}$PEPTIDE1,PEPTIDE1,1:R1-1:R2$$$",
    ],
)
def test_unsupported_or_malformed_helm_rejected(helm) -> None:
    with pytest.raises(HELMParseError):
        mol_from_helm(helm, load_helm_monomer_library())


# ── Cyclic peptides ──────────────────────────────────────────────────


def test_cyclic_diglycine_reference() -> None:
    library = load_helm_monomer_library()
    helm = cyclic_peptide_helm(["G", "G"], library)
    mol = mol_from_helm(helm, library)

    assert _canonical_smiles(mol) == _canonical_smiles(Chem.MolFromSmiles("O=C1CNC(=O)CN1"))
    assert Chem.AddHs(mol).GetNumAtoms() == 14
    assert mol.GetRingInfo().NumRings() == 1
    reversed_helm = helm.replace("1:R1-2:R2", "2:R2-1:R1")
    assert _canonical_smiles(mol_from_helm(reversed_helm, library)) == _canonical_smiles(mol)


@pytest.mark.parametrize(
    "sequence",
    [
        ["G", "G"],
        ["G", "P", "A", "dL"],
        ["meL", "meY", "G", "dP"],
        ["A", "A", "A", "A"],
        ["C", "H", "D", "E", "R", "K"],
    ],
)
def test_cyclic_rotations_and_rewrites_keep_the_molecule(sequence) -> None:
    library = load_helm_monomer_library()
    helm = cyclic_peptide_helm(sequence, library)
    reference = _canonical_smiles(mol_from_helm(helm, library))
    size = len(sequence)
    selections = (
        [[i] for i in range(1, size + 1)]
        + [[i, i % size + 1] for i in range(1, size + 1)]
        + [list(range(1, size + 1))]
    )
    for positions in selections:
        rewritten = rewrite_helm_residues(helm, positions, library)
        actual = mol_from_helm(rewritten, library)
        assert rewritten.split("}$", 1)[1] == helm.split("}$", 1)[1]
        assert _canonical_smiles(actual) == reference
        assert not any(atom.GetAtomicNum() == 0 for atom in actual.GetAtoms())
        assert rewrite_helm_residues(rewritten, positions, library) == rewritten
    for offset in range(-size, 2 * size + 1):
        rotated = rotate_cyclic_helm(helm, offset, library)
        shift = offset % size
        assert rotated == cyclic_peptide_helm(sequence[shift:] + sequence[:shift], library)
        assert rotate_cyclic_helm(rotated, -offset, library) == helm
        assert _canonical_smiles(mol_from_helm(rotated, library)) == reference
        for positions in selections:
            rewritten = rewrite_helm_residues(rotated, positions, library)
            assert _canonical_smiles(mol_from_helm(rewritten, library)) == reference
            # Rewrite then rotate must move the same physical residue tokens.
            original_positions = [(p - 1 + shift) % size + 1 for p in positions]
            first_rewritten = rewrite_helm_residues(helm, original_positions, library)
            assert rotate_cyclic_helm(first_rewritten, offset, library) == rewritten


@pytest.mark.parametrize("code", list("ACDEFGHIKLMNPQRSTVWY"))
def test_all_standard_cyclic_residues_rewrite(code) -> None:
    library = load_helm_monomer_library()
    helm = cyclic_peptide_helm([code, "G", code], library)
    rewritten = rewrite_helm_residues(helm, [1, 2, 3], library)

    assert _canonical_smiles(mol_from_helm(rewritten, library)) == _canonical_smiles(mol_from_helm(helm, library))


def test_cyclic_preserves_explicit_amber_sidechain_states() -> None:
    library = load_helm_monomer_library()
    residues = [
        "[O=C([O-])C[C@H](N[*:1])C(=O)[*:2]]",
        "[[NH3+]CCCC[C@H](N[*:1])C(=O)[*:2]]",
        "[O=C([C@H](Cc1c[nH]cn1)N[*:1])[*:2]]",
    ]
    helm = cyclic_peptide_helm(residues, library)
    mol = mol_from_helm(helm, library)

    assert Chem.GetFormalCharge(mol) == 0
    assert len([atom for atom in mol.GetAtoms() if atom.GetFormalCharge()]) == 2
    rewritten = rewrite_helm_residues(helm, [1, 2, 3], library)
    assert _canonical_smiles(mol_from_helm(rewritten, library)) == _canonical_smiles(mol)


@pytest.mark.parametrize(
    "connection",
    [
        "PEPTIDE2,PEPTIDE1,1:R1-3:R2",
        "PEPTIDE1,PEPTIDE1,0:R1-3:R2",
        "PEPTIDE1,PEPTIDE1,1:R1-4:R2",
        "PEPTIDE1,PEPTIDE1,1:R3-3:R3",
        "PEPTIDE1,PEPTIDE1,1:R2-3:R1",
        "PEPTIDE1,PEPTIDE1,1:R1-2:R2",
        "PEPTIDE1,PEPTIDE1,1:R1-3:R2|PEPTIDE1,PEPTIDE1,1:R1-3:R2",
        "bad",
    ],
)
def test_invalid_cyclic_connections_fail(connection) -> None:
    with pytest.raises(HELMParseError):
        mol_from_helm("PEPTIDE1{G.G.G}$" + connection + "$$$", load_helm_monomer_library())


@pytest.mark.parametrize("positions", [[0], [4], [1, 1], [True], [1.5]])
def test_rewrite_rejects_invalid_positions(positions) -> None:
    library = load_helm_monomer_library()
    with pytest.raises(HELMParseError):
        rewrite_helm_residues(cyclic_peptide_helm(["G", "A", "P"], library), positions, library)


def test_cyclic_missing_attachment_rejected() -> None:
    with pytest.raises(HELMParseError, match="R2"):
        cyclic_peptide_helm(["G", "am"], load_helm_monomer_library())


def test_cyclic_rewrite_does_not_mutate_library() -> None:
    library = load_helm_monomer_library()

    def snapshot():
        return {key: (_canonical_smiles(value.mol), dict(value.rgroup_atoms)) for key, value in library["PEPTIDE"].items()}

    before = snapshot()
    rewrite_helm_residues(cyclic_peptide_helm(["D", "P", "meL", "dT"], library), [1, 2, 3, 4], library)
    assert snapshot() == before


def test_cyclic_core_tokenization_round_trip() -> None:
    vocab = core_vocab(4096)
    library = load_helm_monomer_library()
    helm = rewrite_helm_residues(cyclic_peptide_helm(["G", "P", "A"], library), [1, 3], library)
    inverse = {value: key for key, value in vocab["core_tokens"].items()}

    assert "".join(inverse[i] for i in CoreTokenizer(vocab).encode(helm)) == helm


@pytest.mark.parametrize(
    "smiles",
    [
        "O=C1CNC(=O)CN1",
        "CN1C(=O)[C@H](CCC(N)=O)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H]1CS",
        "C[C@@H](O)[C@@H]1NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O",
    ],
)
def test_cyclic_source_graph_round_trip(smiles) -> None:
    helm = cyclic_peptide_helm_from_smiles(smiles)

    assert _canonical_smiles(mol_from_helm(helm, {})) == _canonical_smiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", ["not smiles", "NCC(=O)NCC(=O)O", "C1CCCCC1", "O=C1CNC(=O)CN1.O"])
def test_cyclic_source_graph_rejects_unsupported(smiles) -> None:
    with pytest.raises(HELMParseError):
        cyclic_peptide_helm_from_smiles(smiles)


def test_cyclic_rotations_compose() -> None:
    library = load_helm_monomer_library()
    helm = cyclic_peptide_helm(["G", "P", "A", "dL"], library)
    current, total = helm, 0
    for increment in np.random.default_rng(42).integers(4, size=32):
        current = rotate_cyclic_helm(current, increment, library)
        total += int(increment)
        assert current == rotate_cyclic_helm(helm, total, library)


def test_cyclic_rotation_preserves_inline_text_and_sections() -> None:
    library = load_helm_monomer_library()
    # The CXSMILES dollar signs must not be mistaken for HELM separators.
    inline = "[*NCC(=O)* |$_R1;;;;;_R2$|]"
    helm = f"PEPTIDE7{{A.{inline}.[dP]}}$PEPTIDE7,PEPTIDE7,3:R2-1:R1$$$V2.0"

    rotated = rotate_cyclic_helm(helm, 1, library)

    assert rotated == f"PEPTIDE7{{{inline}.[dP].A}}$PEPTIDE7,PEPTIDE7,3:R2-1:R1$$$V2.0"
    assert split_peptide(rotated)[1][0] == inline
    assert _canonical_smiles(mol_from_helm(rotated, library)) == _canonical_smiles(mol_from_helm(helm, library))
    vocab = core_vocab(4096)
    inverse = {value: key for key, value in vocab["core_tokens"].items()}
    assert "".join(inverse[i] for i in CoreTokenizer(vocab).encode(rotated)) == rotated


def test_cyclic_rotation_preserves_amber_source_graph() -> None:
    smiles = "C[C@@H](O)[C@@H]1NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@H](Cc2c[nH]c3ccccc23)NC1=O"
    helm = cyclic_peptide_helm_from_smiles(smiles)
    expected = _canonical_smiles(Chem.MolFromSmiles(smiles))

    for offset in range(4):
        assert _canonical_smiles(mol_from_helm(rotate_cyclic_helm(helm, offset, {}), {})) == expected


@pytest.mark.parametrize("offset", [True, False, 1.5, "1", None])
def test_cyclic_rotation_rejects_invalid_offsets(offset) -> None:
    library = load_helm_monomer_library()
    with pytest.raises(HELMParseError, match="integer"):
        rotate_cyclic_helm(cyclic_peptide_helm(["G", "A"], library), offset, library)


@pytest.mark.parametrize(
    "helm",
    ["PEPTIDE1{G.A}$$$$", "PEPTIDE1{G.A}", "PEPTIDE1{C.A.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"],
)
def test_cyclic_rotation_rejects_linear_or_unsupported_topology(helm) -> None:
    with pytest.raises(HELMParseError):
        rotate_cyclic_helm(helm, 0, load_helm_monomer_library())


# ── Notation normalisation ───────────────────────────────────────────


@pytest.mark.parametrize(
    "notation",
    [
        "[#HeadTailLactam][G][P][A][dL]",
        " PEPTIDE1{[G].[P].[A].dL}$PEPTIDE1,PEPTIDE1,1:R1-4:R2$$$ \n",
        "PEPTIDE1{G.P.A.[dL]}$PEPTIDE1,PEPTIDE1,1:R1-4:R2$$$",
    ],
)
def test_normalize_dataset_and_helm_notation(notation) -> None:
    library = load_helm_monomer_library()
    expected = "PEPTIDE1{G.P.A.[dL]}$PEPTIDE1,PEPTIDE1,1:R1-4:R2$$$"

    assert normalize_peptide_helm(notation, library) == expected
    assert normalize_peptide_helm(expected, library) == expected
    assert rotate_cyclic_helm(notation, 1, library) == rotate_cyclic_helm(expected, 1, library)
    assert rewrite_helm_residues(notation, [1, 2], library) == rewrite_helm_residues(expected, [1, 2], library)


def test_normalize_preserves_standard_nested_smiles_and_version() -> None:
    library = load_helm_monomer_library()
    # Pistoia HELM2NotationToolkit SMILESTest.getInlineSmilesPeptideNotation
    inline = "[C[C@H](N[*])C([*])=O |$;;;_R1;;_R2;$|]"
    helm = f"PEPTIDE1{{[G].{inline}.[dP]}}$$$$V2.0"
    expected = f"PEPTIDE1{{G.{inline}.[dP]}}$$$$V2.0"

    assert normalize_peptide_helm(helm, library) == expected
    assert _canonical_smiles(mol_from_helm(helm, library)) == _canonical_smiles(mol_from_helm(expected, library))


def test_normalize_completes_linear_helm_and_constructor_tokens() -> None:
    library = load_helm_monomer_library()

    assert normalize_peptide_helm("PEPTIDE1{[G].dP}", library) == "PEPTIDE1{G.[dP]}$$$$"
    assert cyclic_peptide_helm(["[G]", "dP"], library) == "PEPTIDE1{G.[dP]}$PEPTIDE1,PEPTIDE1,1:R1-2:R2$$$"


@pytest.mark.parametrize(
    "notation",
    [
        "[#HeadTailLactam][G][dm]",
        "[#HeadTailLactam][G][mL]",
        "[#HeadTailLactam][G][P][[#HeadTailLactam]",
        "[#HeadTailLactam][G][P]garbage",
        "[#HeadTailLactam][G]",
        "[G][P]",
    ],
)
def test_normalize_rejects_ambiguous_dataset_notation(notation) -> None:
    with pytest.raises(HELMParseError):
        normalize_peptide_helm(notation, load_helm_monomer_library())
