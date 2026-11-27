"""Linear peptides in the ff14SB atom order: mapped SMILES, HELM and PDB blocks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from beartype import beartype
from rdkit import Chem, rdBase
from rdkit.Chem import rdmolops

from pbg.core.constants import ONE_TO_THREE
from pbg.core.helm import (
    load_helm_monomer_library,
    normalize_peptide_helm,
    rewrite_helm_residues,
    split_peptide,
)
from pbg.core.molecule import mapped_mol

_XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "forcefields" / "protein.ff14SB.xml"
_MAX_INFERRED_SEQUENCE_CANDIDATES = 10_000


@dataclass(frozen=True)
class _ResidueTemplate:
    atom_names: tuple[str, ...]
    atom_elements: dict[str, str]
    hydrogen_anchors: dict[str, str]


@lru_cache(maxsize=1)
def _load_ff14sb_templates() -> dict[str, _ResidueTemplate]:
    tree = ET.parse(_XML_PATH)
    root = tree.getroot()

    atom_types: dict[str, str] = {}
    atom_types_node = root.find("AtomTypes")
    if atom_types_node is None:
        raise ValueError("protein.ff14SB.xml is missing AtomTypes")
    for type_node in atom_types_node.findall("Type"):
        type_name = type_node.attrib["name"]
        atom_types[type_name] = type_node.attrib["element"]

    residues_node = root.find("Residues")
    if residues_node is None:
        raise ValueError("protein.ff14SB.xml is missing Residues")

    templates: dict[str, _ResidueTemplate] = {}
    for residue_node in residues_node.findall("Residue"):
        residue_name = residue_node.attrib["name"]
        atom_names: list[str] = []
        atom_elements: dict[str, str] = {}
        for atom_node in residue_node.findall("Atom"):
            atom_name = atom_node.attrib["name"]
            atom_names.append(atom_name)
            atom_elements[atom_name] = atom_types[atom_node.attrib["type"]]

        hydrogen_anchors: dict[str, str] = {}
        for bond_node in residue_node.findall("Bond"):
            atom_name_1 = bond_node.attrib["atomName1"]
            atom_name_2 = bond_node.attrib["atomName2"]
            elem_1 = atom_elements[atom_name_1]
            elem_2 = atom_elements[atom_name_2]
            if elem_1 == "H" and elem_2 != "H":
                hydrogen_anchors[atom_name_1] = atom_name_2
            elif elem_2 == "H" and elem_1 != "H":
                hydrogen_anchors[atom_name_2] = atom_name_1

        templates[residue_name] = _ResidueTemplate(
            atom_names=tuple(atom_names),
            atom_elements=atom_elements,
            hydrogen_anchors=hydrogen_anchors,
        )
    return templates


def _parse_sequence(sequence: str, *, d_residues: bool = False) -> list[str]:
    """One-letter residues of a sequence; ``d`` before a letter marks a D-residue (``dA``).

    Without ``d_residues`` a D-residue raises ``NotImplementedError``.
    """
    residues: list[str] = []
    i = 0
    while i < len(sequence):
        if sequence[i] == "d" and i + 1 < len(sequence) and sequence[i + 1].upper() in ONE_TO_THREE:
            if not d_residues:
                raise NotImplementedError("sequence_to_smiles does not support D-amino acids")
            residues.append("d" + sequence[i + 1].upper())
            i += 2
            continue
        base = sequence[i].upper()
        if base not in ONE_TO_THREE:
            raise ValueError(f"Unknown token at position {i}: {sequence[i]!r}")
        residues.append(base)
        i += 1
    if not residues:
        raise ValueError("a peptide sequence must not be empty")
    return residues


@lru_cache(maxsize=256)
def _residue_template(base: str, context: str) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    """Atom names, their elements, and the heavy anchor of each H, of a residue in a ``LINEAR_CONTEXTS`` position.

    ``first`` and ``last`` take the N- and C-terminal ff14SB templates. A
    ``single`` residue takes the N-terminal template plus the heavy atoms that
    only the C-terminal template has.
    """
    templates = _load_ff14sb_templates()
    name = ONE_TO_THREE[base]
    if context != "single":
        template = templates[{"first": f"N{name}", "last": f"C{name}", "internal": name}[context]]
        return template.atom_names, template.atom_elements, template.hydrogen_anchors
    n_template, c_template = templates[f"N{name}"], templates[f"C{name}"]
    atom_names = n_template.atom_names + tuple(
        atom for atom in c_template.atom_names
        if atom not in n_template.atom_names and c_template.atom_elements[atom] != "H"
    )
    return atom_names, {**c_template.atom_elements, **n_template.atom_elements}, n_template.hydrogen_anchors


def _fragment_signature(mol, atom_indices: tuple[int, ...]):
    atom_set = set(atom_indices)
    local_idx = {atom_idx: i for i, atom_idx in enumerate(atom_indices)}
    bonds: list[tuple[int, int, str]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in atom_set or end not in atom_set:
            continue
        i = local_idx[begin]
        j = local_idx[end]
        if i > j:
            i, j = j, i
        bonds.append((i, j, str(bond.GetBondType())))
    return (
        tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atom_indices),
        tuple(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atom_indices),
        tuple(str(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atom_indices),
        tuple(sorted(bonds)),
    )


def _infer_sequences_from_element_order(elements: tuple[str, ...]) -> list[str]:
    def element_order(base: str, context: str) -> tuple[str, ...]:
        names, atom_elements, _ = _residue_template(base, context)
        return tuple(atom_elements[name] for name in names)

    bases = tuple(ONE_TO_THREE)
    candidates = [base for base in bases if element_order(base, "single") == elements]
    n_prefixes = {base: element_order(base, "first") for base in bases}
    mid_prefixes = {base: element_order(base, "internal") for base in bases}
    c_prefixes = {base: element_order(base, "last") for base in bases}

    def parse_tail(pos: int, prefix: str) -> None:
        if len(candidates) >= _MAX_INFERRED_SEQUENCE_CANDIDATES:
            raise ValueError("Too many peptide sequence candidates inferred from SMILES")
        for base, suffix in c_prefixes.items():
            end = pos + len(suffix)
            if end == len(elements) and elements[pos:end] == suffix:
                candidates.append(prefix + base)
        for base, suffix in mid_prefixes.items():
            end = pos + len(suffix)
            if end < len(elements) and elements[pos:end] == suffix:
                parse_tail(end, prefix + base)

    for base, prefix in n_prefixes.items():
        end = len(prefix)
        if end < len(elements) and elements[:end] == prefix:
            parse_tail(end, base)
    return candidates


@lru_cache(maxsize=32768)
@beartype
def infer_linear_peptide_sequence_from_smiles(smiles: str) -> str:
    """Infer a PBG canonical linear-peptide sequence from mapped SMILES.

    The public contract stays SMILES-only. This is intentionally strict: it
    supports explicit-H, atom-mapped SMILES emitted by ``sequence_to_smiles``
    and solvated extensions that append water/ion fragments after the peptide.
    """
    mol = mapped_mol(smiles)
    if not mol.GetAtomWithIdx(0).GetAtomMapNum():
        raise ValueError(f"SMILES must carry atom maps 1..N: {smiles!r}")
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    matches: list[str] = []

    for fragment in fragments:
        atom_indices = tuple(sorted(int(i) for i in fragment))
        atomic_numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atom_indices)
        if 6 not in atomic_numbers or 7 not in atomic_numbers:
            continue
        elements = tuple(mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices)
        for sequence in _infer_sequences_from_element_order(elements):
            expected_mol = mapped_mol(sequence_to_smiles(sequence))
            expected_indices = tuple(range(expected_mol.GetNumAtoms()))
            if _fragment_signature(mol, atom_indices) == _fragment_signature(
                expected_mol, expected_indices
            ):
                matches.append(sequence)

    unique_matches = sorted(set(matches))
    if len(unique_matches) != 1:
        raise ValueError(
            "Could not infer a unique PBG canonical linear-peptide sequence "
            f"from SMILES; candidates={unique_matches}"
        )
    return unique_matches[0]


@beartype
def sequence_to_helm(sequence: str) -> str:
    """Return standard linear peptide HELM for a one-letter sequence."""
    residues = _parse_sequence(sequence.strip(), d_residues=True)
    return f"PEPTIDE1{{{'.'.join(f'[{r}]' if r.startswith('d') else r for r in residues)}}}$$$$"


# Residue context -> (peptide pattern, 1-based position of the residue in it).
LINEAR_CONTEXTS = {'first': ('{}G', 1), 'last': ('G{}', 2), 'internal': ('G{}G', 2), 'single': ('{}', 1)}


def linear_context(index: int, n_residues: int) -> str:
    """The ``LINEAR_CONTEXTS`` key of residue ``index`` (0-based) in a linear peptide."""
    if n_residues == 1:
        return 'single'
    return 'first' if index == 0 else 'last' if index == n_residues - 1 else 'internal'


@lru_cache(maxsize=256)
def linear_inline_residue_token(symbol: str, context: str) -> str:
    """Return the inline token that ``rewrite_helm_residues`` writes for a residue in this context."""
    pattern, position = LINEAR_CONTEXTS[context]
    rewritten = rewrite_helm_residues(sequence_to_helm(pattern.format(symbol)), [position], load_helm_monomer_library())
    return split_peptide(rewritten)[1][position - 1]


@lru_cache(maxsize=1024)
def ff14sb_inline_residue_token(symbol: str, context: str) -> str:
    """Return the inline token in the ff14SB state of the coordinates.

    Unlike the neutral HELM-library token, it states the terminal and side-chain
    charges and the HIE tautomer of ``sequence_to_smiles``. Bonds to neighbouring
    residues become ``[*:1]`` (on N) and ``[*:2]`` (on C), in canonical SMILES.
    """
    pattern, position = LINEAR_CONTEXTS[context]
    mol = _build_linear_peptide_mol(pattern.format(symbol))
    residue_of = {a.GetIdx(): a.GetPDBResidueInfo().GetResidueNumber() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    cut = [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in residue_of and b.GetEndAtomIdx() in residue_of
           and residue_of[b.GetBeginAtomIdx()] != residue_of[b.GetEndAtomIdx()]
           and position in (residue_of[b.GetBeginAtomIdx()], residue_of[b.GetEndAtomIdx()])]
    pieces = Chem.FragmentOnBonds(mol, cut, addDummies=True) if cut else Chem.Mol(mol)
    fragment_of = []
    parts = Chem.GetMolFrags(pieces, asMols=True, frags=fragment_of, sanitizeFrags=True)
    anchor = next(i for i, number in residue_of.items() if number == position)
    residue = Chem.RWMol(parts[fragment_of[anchor]])
    for atom in residue.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(1 if atom.GetNeighbors()[0].GetSymbol() == 'N' else 2)
    return '[' + Chem.MolToSmiles(Chem.RemoveHs(residue.GetMol()), canonical=True, isomericSmiles=True) + ']'


@lru_cache(maxsize=4096)
def linear_state_token(token: str, context: str) -> str:
    """Inline token of one linear-peptide position in the ff14SB state of the coordinates.

    A canonical residue takes ``ff14sb_inline_residue_token``. A D-residue, i.e. a
    residue whose mirror image is canonical, takes the mirror image of that token,
    so its termini and side chain keep their ff14SB charges. Any other residue keeps
    the token as given.
    """
    def mirror(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.InvertChirality()
        return Chem.MolToSmiles(mol)

    try:
        return ff14sb_inline_residue_token(linear_inline_residue_symbol(token, context), context)
    except ValueError:
        pass
    try:
        symbol = linear_inline_residue_symbol(f"[{mirror(token[1:-1])}]", context)
    except ValueError:
        return token
    return f"[{mirror(ff14sb_inline_residue_token(symbol, context)[1:-1])}]"


@lru_cache(maxsize=4)
def linear_inline_residue_symbols(context: str) -> dict[str, str]:
    """Recognize the residue chemistry of linear inline tokens.

    Both the neutral HELM-library token and the ff14SB-state token resolve to
    the same residue; ``sequence_to_smiles`` then applies the ff14SB states.
    """
    result = {}
    for symbol in ONE_TO_THREE:
        for token in (linear_inline_residue_token(symbol, context), ff14sb_inline_residue_token(symbol, context)):
            mol = Chem.MolFromSmiles(token[1:-1])
            key = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
            if key in result and result[key] != symbol:
                raise ValueError('Ambiguous linear HELM residue templates')
            result[key] = symbol
    return result


@lru_cache(maxsize=256)
def linear_inline_residue_symbol(token: str, context: str) -> str:
    # Named library monomers ([dL], [meA]) are not SMILES; they fail quietly here.
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(token[1:-1]) if token.startswith('[') and token.endswith(']') else None
    if mol is None:
        raise ValueError(f'Unsupported linear peptide monomer: {token}')
    key = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
    symbol = linear_inline_residue_symbols(context).get(key)
    if symbol is None:
        raise ValueError(f'Linear monomer has no registered ff14SB sequence rule: {token}')
    return symbol


@lru_cache(maxsize=32768)
def linear_sequence_from_helm(helm: str) -> str:
    """Resolve standard L-peptide HELM, including equivalent inline rewrites.

    Terminal and side-chain protonation then follow ``sequence_to_smiles``.
    Modified or unregistered monomers fail rather than changing their chemistry.
    """
    polymer, tokens, sections = split_peptide(helm)
    if len(sections) not in {0, 4} or any(sections[:3]) or (len(sections) == 4 and sections[3] not in {'', 'V2.0'}):
        raise ValueError('Expected one linear PEPTIDE polymer without connections or annotations')
    if not re.fullmatch(r'PEPTIDE[1-9][0-9]*', polymer):
        raise ValueError('Expected one linear PEPTIDE polymer')
    symbols = []
    for index, token in enumerate(tokens):
        symbol = token[1:-1] if token.startswith('[') and token.endswith(']') else token
        if symbol not in ONE_TO_THREE:
            symbol = linear_inline_residue_symbol(token, linear_context(index, len(tokens)))
        symbols.append(symbol)
    return ''.join(symbols)


@beartype
def infer_linear_peptide_helm_from_smiles(smiles: str) -> str:
    """Infer PBG canonical linear-peptide HELM from mapped SMILES."""
    return sequence_to_helm(infer_linear_peptide_sequence_from_smiles(smiles))


@beartype
def resolve_linear_peptide_helm(
    *,
    helm: str | None = None,
    sequence: str | None = None,
    smiles: str | None = None,
) -> str:
    """Resolve HELM for a peptide frame from the most specific field available."""
    notation = helm if helm is not None else sequence
    if notation is not None and notation.strip().startswith("[#HeadTailLactam]"):
        return normalize_peptide_helm(notation, load_helm_monomer_library())
    if helm is not None:
        return helm
    if sequence is not None:
        return sequence_to_helm(sequence)
    if smiles is None:
        raise ValueError("resolve_linear_peptide_helm requires helm, sequence, or smiles")
    return infer_linear_peptide_helm_from_smiles(smiles)


def _build_linear_peptide_mol(sequence: str):
    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        raise ValueError(f"RDKit could not build peptide from sequence {sequence!r}")
    mol = Chem.AddHs(mol)
    rw_mol = Chem.RWMol(mol)

    first_n_idx: int | None = None
    last_oxt_idx: int | None = None
    last_residue_number = -1
    for atom in rw_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        atom_name = info.GetName().strip()
        residue_number = info.GetResidueNumber()
        if residue_number == 1 and atom_name == "N":
            first_n_idx = atom.GetIdx()
        if atom_name == "OXT" and residue_number > last_residue_number:
            last_oxt_idx = atom.GetIdx()
            last_residue_number = residue_number

    if first_n_idx is None or last_oxt_idx is None:
        raise ValueError("Could not locate peptide termini in RDKit molecule")

    oxt_h_idx: int | None = None
    for neighbor in rw_mol.GetAtomWithIdx(last_oxt_idx).GetNeighbors():
        if neighbor.GetAtomicNum() == 1:
            oxt_h_idx = neighbor.GetIdx()
            break
    if oxt_h_idx is None:
        raise ValueError("Could not locate terminal OXT hydrogen")

    rw_mol.RemoveAtom(oxt_h_idx)
    if oxt_h_idx < first_n_idx:
        first_n_idx -= 1

    n_term_h_idx = rw_mol.AddAtom(Chem.Atom(1))
    rw_mol.AddBond(first_n_idx, n_term_h_idx, Chem.BondType.SINGLE)

    for atom in rw_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        atom_name = info.GetName().strip()
        residue_number = info.GetResidueNumber()
        if residue_number == 1 and atom_name == "N":
            atom.SetFormalCharge(1)
        elif residue_number == last_residue_number and atom_name == "OXT":
            atom.SetFormalCharge(-1)

    _adjust_sidechain_states(rw_mol)
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _find_residue_atom_index(rw_mol, residue_number: int, atom_name: str) -> int | None:
    for atom in rw_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if (
            info.GetResidueNumber() == residue_number
            and info.GetName().strip() == atom_name
        ):
            return atom.GetIdx()
    return None


# ff14SB side-chain states: (atom, formal charge, H count to reach); None removes one H.
_SIDECHAIN_STATES = {"ARG": ("NH1", 1, 2), "LYS": ("NZ", 1, 3), "ASP": ("OD2", -1, None), "GLU": ("OE2", -1, None)}


def _adjust_sidechain_states(rw_mol) -> None:
    residues: dict[int, str] = {}
    for atom in rw_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None:
            residues.setdefault(info.GetResidueNumber(), info.GetResidueName().strip())
    for residue_number, residue_name in sorted(residues.items()):
        if residue_name not in _SIDECHAIN_STATES:
            continue
        atom_name, charge, n_hydrogens = _SIDECHAIN_STATES[residue_name]
        atom_idx = _find_residue_atom_index(rw_mol, residue_number, atom_name)
        if atom_idx is None:
            continue
        hydrogens = [n.GetIdx() for n in rw_mol.GetAtomWithIdx(atom_idx).GetNeighbors() if n.GetAtomicNum() == 1]
        if n_hydrogens is None:
            if hydrogens:
                rw_mol.RemoveAtom(hydrogens[0])
            atom_idx = _find_residue_atom_index(rw_mol, residue_number, atom_name)
            if atom_idx is None:
                continue
        else:
            for _ in range(n_hydrogens - len(hydrogens)):
                rw_mol.AddBond(atom_idx, rw_mol.AddAtom(Chem.Atom(1)), Chem.BondType.SINGLE)
        rw_mol.GetAtomWithIdx(atom_idx).SetFormalCharge(charge)


def _normalize_atom_name(atom) -> str:
    info = atom.GetPDBResidueInfo()
    if info is None:
        raise ValueError("Atom is missing PDB residue metadata")
    atom_name = info.GetName().strip()
    residue_name = info.GetResidueName().strip()
    if residue_name == "ILE" and atom_name in {"CG1", "CG2"}:
        neighbor_names = {
            neighbor.GetPDBResidueInfo().GetName().strip()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetPDBResidueInfo() is not None
        }
        if atom_name == "CG1" and "CD1" not in neighbor_names:
            return "CG2"
        if atom_name == "CG2" and "CD1" in neighbor_names:
            return "CG1"
    return atom_name


def _ordered_atom_indices(mol, bases: list[str]) -> list[int]:
    """Order atoms by the ff14SB residue templates."""
    heavy_atoms: dict[int, dict[str, int]] = {}
    hydrogens: dict[int, dict[str, list[int]]] = {}

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise ValueError("Heavy atom is missing PDB residue metadata")
        residue_number = info.GetResidueNumber()
        atom_name = _normalize_atom_name(atom)
        heavy_atoms.setdefault(residue_number, {})[atom_name] = atom.GetIdx()

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbor = atom.GetNeighbors()[0]
        info = neighbor.GetPDBResidueInfo()
        if info is None:
            raise ValueError("Hydrogen is attached to atom without residue metadata")
        residue_number = info.GetResidueNumber()
        anchor_name = _normalize_atom_name(neighbor)
        residue_hydrogens = hydrogens.setdefault(residue_number, {})
        residue_hydrogens.setdefault(anchor_name, []).append(atom.GetIdx())

    for residue_hydrogens in hydrogens.values():
        for atom_indices in residue_hydrogens.values():
            atom_indices.sort()

    ordered_indices: list[int] = []
    n_residues = len(bases)
    for residue_idx, base in enumerate(bases, start=1):
        atom_names, _, hydrogen_anchors = _residue_template(base, linear_context(residue_idx - 1, n_residues))
        residue_heavy_atoms = heavy_atoms[residue_idx]
        residue_hydrogens = hydrogens.get(residue_idx, {})
        for atom_name in atom_names:
            if atom_name in residue_heavy_atoms:
                ordered_indices.append(residue_heavy_atoms[atom_name])
                continue
            anchor_name = hydrogen_anchors[atom_name]
            anchor_hydrogens = residue_hydrogens.get(anchor_name, [])
            if not anchor_hydrogens:
                raise ValueError(
                    f"Could not match hydrogen {atom_name} in residue {residue_idx}"
                )
            ordered_indices.append(anchor_hydrogens.pop(0))

    if len(ordered_indices) != mol.GetNumAtoms():
        raise ValueError("Amber atom ordering did not cover every atom")
    if len(set(ordered_indices)) != len(ordered_indices):
        raise ValueError("Amber atom ordering assigned duplicate atom indices")
    return ordered_indices


@lru_cache(maxsize=32768)
@beartype
def sequence_to_smiles(sequence: str) -> str:
    """Return explicit-H mapped SMILES for a linear peptide in ff14SB template atom order."""
    bases = _parse_sequence(sequence)
    mol = _build_linear_peptide_mol("".join(bases))
    mol = rdmolops.RenumberAtoms(mol, _ordered_atom_indices(mol, bases))
    for atom_idx, atom in enumerate(mol.GetAtoms(), start=1):
        atom.SetAtomMapNum(atom_idx)
    return Chem.MolToSmiles(
        mol,
        canonical=False,
        allHsExplicit=True,
        isomericSmiles=True,
    )


@beartype
def sequence_to_pdb_block(sequence: str) -> str:
    """Return a valid peptide PDB block with atoms in ff14SB order."""
    bases = _parse_sequence(sequence)
    mol = _build_linear_peptide_mol("".join(bases))
    order = _ordered_atom_indices(mol, bases)
    mol = rdmolops.RenumberAtoms(mol, order)

    n_residues = len(bases)
    atom_cursor = 0

    for residue_idx, base in enumerate(bases, start=1):
        three_letter = ONE_TO_THREE[base]
        atom_names = _residue_template(base, linear_context(residue_idx - 1, n_residues))[0]

        for aname in atom_names:
            atom = mol.GetAtomWithIdx(atom_cursor)
            info = Chem.AtomPDBResidueInfo()
            padded = f" {aname:<3s}" if len(aname) <= 3 else aname
            info.SetName(padded)
            info.SetResidueName(three_letter)
            info.SetResidueNumber(residue_idx)
            info.SetChainId("A")
            info.SetIsHeteroAtom(False)
            atom.SetMonomerInfo(info)
            atom_cursor += 1

    if atom_cursor != mol.GetNumAtoms():
        raise RuntimeError(
            f"Template walk covered {atom_cursor} atoms but mol has {mol.GetNumAtoms()} atoms for sequence {sequence!r}"
        )

    return Chem.MolToPDBBlock(mol)
