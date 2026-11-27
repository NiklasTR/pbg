"""Minimal HELM-to-RDKit support for official HELM monomer JSON files.

The official Pistoia HELM library stores attachment sites as atom-mapped cap
atoms in each monomer ``smiles`` field, for example
``[H:1]N...C([OH:2])=O``.  This module uses that representation directly
instead of relying on a pre-transformed library with dummy atoms and custom
metadata.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path

from rdkit import Chem


class HELMParseError(ValueError):
    """Raised when a HELM string or monomer library cannot be interpreted."""


@dataclass(frozen=True)
class HELMMonomer:
    symbol: str
    polymer_type: str
    mol: object
    rgroup_atoms: dict[int, int]


HELMMonomerLibrary = dict[str, dict[str, HELMMonomer]]

# The PEPTIDE monomers of the Pistoia HELM core library, records unchanged.
_DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "assets" / "helm" / "HELMCoreLibrary.json"


def mapped_smiles_to_chem_helm(
    smiles: str,
    *,
    prune_water: bool = False,
) -> str:
    """Convert mapped, explicit-H SMILES to canonical CHEM HELM."""

    if not smiles.strip():
        raise ValueError("SMILES must not be empty")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    if prune_water:
        fragments = Chem.GetMolFrags(mol)
        water_fragments = [
            fragment
            for fragment in fragments
            if len(fragment) == 1
            and mol.GetAtomWithIdx(fragment[0]).GetAtomicNum() == 8
            and mol.GetAtomWithIdx(fragment[0]).GetFormalCharge() == 0
            and mol.GetAtomWithIdx(fragment[0]).GetTotalNumHs() == 2
        ]
        if water_fragments:
            removable = (
                water_fragments[1:]
                if len(water_fragments) == len(fragments)
                else water_fragments
            )
            editable = Chem.RWMol(mol)
            for atom_idx in sorted(
                (fragment[0] for fragment in removable),
                reverse=True,
            ):
                editable.RemoveAtom(atom_idx)
            mol = editable.GetMol()
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return f"CHEM1{{[{canonical_smiles}]}}$$$$"


def load_helm_monomer_library(json_path: str | Path | None = None) -> HELMMonomerLibrary:
    """Load a Pistoia-style HELM monomer JSON file (default: the bundled core library).

    The loader uses the official fields ``symbol``, ``polymerType`` and
    ``smiles``. R-groups come from the atom-map numbers in ``smiles``.
    """

    if json_path is None:
        return _load_default_helm_monomer_library()
    path = Path(json_path)
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise HELMParseError(f"HELM JSON library must contain a list: {path}")
    library: HELMMonomerLibrary = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise HELMParseError(f"Invalid HELM monomer record {index} in {path}")
        symbol = str(record.get("symbol") or "")
        polymer_type = str(record.get("polymerType") or "")
        smiles = str(record.get("smiles") or "")
        if symbol and polymer_type and smiles:
            library.setdefault(polymer_type, {})[symbol] = _monomer_from_smiles(symbol, polymer_type, smiles)
    return library


@lru_cache(maxsize=1)
def _load_default_helm_monomer_library() -> HELMMonomerLibrary:
    return load_helm_monomer_library(_DEFAULT_LIBRARY)


def mol_from_helm(
    helm: str,
    monomer_library: HELMMonomerLibrary,
):
    """Assemble one linear or head-to-tail cyclic PEPTIDE polymer.

    Inline SMILES/CXSMILES monomers are supported. Protonation comes from the
    supplied monomers, not from an implicit force-field or terminal-state rule.
    """

    sections = _split_helm_sections(helm)
    if len(sections) not in {1, 5}:
        raise HELMParseError("Expected a polymer alone or five HELM sections")
    polymer_section = sections[0]
    if not polymer_section:
        raise HELMParseError("HELM string does not contain a polymer section")

    polymers = _split_outside_brackets(polymer_section, "|")
    if len(polymers) != 1:
        raise HELMParseError("Only single-polymer HELM strings are supported")

    polymer_type, sequence = _parse_polymer(polymers[0])
    if polymer_type != "PEPTIDE":
        raise HELMParseError(f"Unsupported polymer type: {polymer_type}")

    monomers = [
        _resolve_monomer(name, polymer_type, monomer_library)
        for name in _split_sequence(sequence)
    ]
    if not monomers:
        raise HELMParseError("PEPTIDE sequence is empty")

    connections = [(i, 2, i + 1, 1) for i in range(len(monomers) - 1)]
    connection_section = sections[1] if len(sections) > 1 else ""
    if connection_section:
        polymer_id = polymers[0].partition("{")[0]
        match = re.fullmatch(
            r"([^,]+),([^,]+),([1-9][0-9]*):R([1-9][0-9]*)-([1-9][0-9]*):R([1-9][0-9]*)",
            connection_section,
        )
        if match is None:
            raise HELMParseError("Expected one explicit head-to-tail connection")
        source, target, left, left_r, right, right_r = match.groups()
        if source != polymer_id or target != polymer_id:
            raise HELMParseError("Connection must reference the declared polymer")
        endpoints = {(int(left), int(left_r)), (int(right), int(right_r))}
        if len(monomers) < 2 or endpoints != {(1, 1), (len(monomers), 2)}:
            raise HELMParseError("Only first R1 to last R2 head-to-tail closure is supported")
        connections.append((len(monomers) - 1, 2, 0, 1))
    if any(sections[2:4]) or len(sections) > 5 or (
        len(sections) == 5 and sections[4] not in {"", "V2.0"}
    ):
        raise HELMParseError("Unsupported HELM grouping, annotation, or version section")
    return _assemble_peptide(monomers, connections)


def normalize_peptide_helm(
    notation: str, monomer_library: HELMMonomerLibrary
) -> str:
    """Normalize one supported PEPTIDE polymer to complete HELM syntax.

    Accept the dataset's non-HELM ``[#HeadTailLactam][G][dP]`` form only
    for standard amino-acid symbols and explicit D variants. Do not infer
    methylation, protonation, missing residues, or malformed closure markers.
    Inline SMILES/CXSMILES text and existing HELM version are preserved.
    This is syntax normalization, not canonical molecular ordering.
    """
    notation = notation.strip()
    if notation.startswith("[#HeadTailLactam]"):
        body = notation.removeprefix("[#HeadTailLactam]")
        if not re.fullmatch(r"(?:\[(?:d)?[ACDEFGHIKLMNPQRSTVWY]\])+", body):
            raise HELMParseError("Invalid or ambiguous dataset residue notation; provide authoritative HELM or SMILES")
        names = re.findall(r"\[([^\[\]]+)\]", body)
        if len(names) < 2:
            raise HELMParseError("A head-to-tail peptide needs at least two residues")
        notation = "PEPTIDE1{" + ".".join(
            name if len(name) == 1 else f"[{name}]" for name in names
        ) + "}$PEPTIDE1,PEPTIDE1,1:R1-" + str(len(names)) + ":R2$$$"
    sections = _split_helm_sections(notation)
    if len(sections) == 1:
        sections.extend([""] * 4)
    polymer_id, _, body = sections[0].partition("{")
    # Normalize named tokens only. SMILES atom brackets are chemical syntax.
    if body.endswith("}"):
        names = _split_sequence(body[:-1])
        normalized = []
        for name in names:
            symbol = name[1:-1] if name.startswith("[") and name.endswith("]") else name
            if symbol in monomer_library.get("PEPTIDE", {}):
                normalized.append(symbol if len(symbol) == 1 else f"[{symbol}]")
            else:
                normalized.append(name)
        sections[0] = polymer_id + "{" + ".".join(normalized) + "}"
    result = "$".join(sections)
    mol_from_helm(result, monomer_library)
    return result


def split_peptide(helm: str) -> tuple[str, list[str], list[str]]:
    """Split a PEPTIDE HELM into its polymer ID, its monomer tokens and its other sections."""
    sections = _split_helm_sections(helm)
    polymer, sep, body = sections[0].partition("{")
    if not sep or not body.endswith("}"):
        raise HELMParseError(f"Invalid polymer section: {sections[0]}")
    return polymer, _split_sequence(body[:-1]), sections[1:]


def join_peptide(polymer: str, tokens: Iterable[str], sections: Iterable[str]) -> str:
    """Write the HELM that ``split_peptide`` splits."""
    return "$".join([polymer + "{" + ".".join(tokens) + "}", *sections])


def rewrite_helm_residues(
    helm: str,
    residue_positions: Iterable[int],
    monomer_library: HELMMonomerLibrary,
) -> str:
    """Replace selected 1-based residue positions with equivalent inline SMILES.

    Selection is explicit: callers may use their training RNG to choose any
    subset, including adjacent positions and the last/first closure pair.
    Connections and all unselected residue tokens remain unchanged.
    """
    polymer_id, names, sections = split_peptide(normalize_peptide_helm(helm, monomer_library))
    positions = list(residue_positions)
    if any(type(p) is not int or not 1 <= p <= len(names) for p in positions):
        raise HELMParseError("Residue positions must be integers in 1..N")
    if len(set(positions)) != len(positions):
        raise HELMParseError("Duplicate residue positions")
    cyclic = bool(sections) and bool(sections[0])
    for position in positions:
        monomer = _resolve_monomer(names[position - 1], "PEPTIDE", monomer_library)
        editable = Chem.RWMol(monomer.mol)
        used = set()
        if position > 1 or cyclic:
            used.add(1)
        if position < len(names) or cyclic:
            used.add(2)
        for atom in editable.GetAtoms():
            atom.SetAtomMapNum(0)
            for prop in ("atomLabel", "dummyLabel", "molFileValue"):
                if atom.HasProp(prop):
                    atom.ClearProp(prop)
        for label in used:
            index = monomer.rgroup_atoms.get(label)
            if index is None:
                raise HELMParseError(f"Residue {position} is missing R{label}")
            # Replace the cap in place, preserving neighbour order and chirality.
            atom = editable.GetAtomWithIdx(index)
            atom.SetAtomicNum(0)
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
            atom.SetAtomMapNum(label)
        inline = editable.GetMol()
        Chem.SanitizeMol(inline)
        inline = Chem.RemoveHs(inline)
        names[position - 1] = "[" + Chem.MolToSmiles(
            inline, canonical=True, isomericSmiles=True
        ) + "]"
    return join_peptide(polymer_id, names, sections)


def rotate_cyclic_helm(
    helm: str, offset: int, monomer_library: HELMMonomerLibrary
) -> str:
    """Rotate a head-to-tail peptide prompt left by ``offset`` residues.

    Offsets wrap modulo the residue count; negative offsets rotate right.
    For random increments, pass ``rng.integers(n_residues)`` from the caller's
    seeded RNG. Zero is allowed. Residue tokens, stereochemistry, polymer ID,
    and the closure section are preserved; residue positions are now relative
    to the new start. This is not a reversal or a coordinate permutation.

    Keep the frame's mapped SMILES and coordinate order authoritative. An
    RDKit molecule assembled from the rotated HELM may have a different atom
    order and must not replace the frame's mapped topology.
    """
    if isinstance(offset, bool) or not isinstance(offset, Integral):
        raise HELMParseError("Rotation offset must be an integer")
    helm = normalize_peptide_helm(helm, monomer_library)
    polymer_id, names, sections = split_peptide(helm)
    if len(sections) != 4 or not sections[0]:
        raise HELMParseError("Rotation requires an explicit head-to-tail closure")
    shift = int(offset) % len(names)
    if shift == 0:
        return helm
    # The new cut is also first R1 / last R2, so the closure text stays valid.
    return join_peptide(polymer_id, names[shift:] + names[:shift], sections)


def cyclic_peptide_helm(
    residue_symbols: Iterable[str], monomer_library: HELMMonomerLibrary
) -> str:
    """Encode a head-to-tail cycle from explicit HELM monomer symbols.

    Use official symbols (e.g. ``meL``), not dataset-specific abbreviations.
    Inline monomers may be supplied with their outer HELM brackets.
    """
    names = list(residue_symbols)
    if len(names) < 2:
        raise HELMParseError("A head-to-tail peptide needs at least two residues")
    tokens = [name if name.startswith("[") or len(name) == 1 else f"[{name}]" for name in names]
    helm = "PEPTIDE1{" + ".".join(tokens) + "}$PEPTIDE1,PEPTIDE1,1:R1-" + str(len(tokens)) + ":R2$$$"
    return normalize_peptide_helm(helm, monomer_library)


def cyclic_peptide_helm_from_smiles(smiles: str) -> str:
    """Encode a head-to-tail alpha-peptide graph as inline HELM residues.

    Source charges and stereochemistry are retained, including N-methyl and D
    residues. No atom or coordinate mapping is returned or changed. Residue
    order follows the input graph; ambiguous or cross-linked backbones fail.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HELMParseError("Invalid cyclic peptide SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    pattern = Chem.MolFromSmarts("[N]-[CX4]-[CX3](=[OX1])")
    matches = mol.GetSubstructMatches(pattern)
    by_n: dict[int, list[tuple[int, ...]]] = {}
    for match in matches:
        by_n.setdefault(match[0], []).append(match)
    cycles: dict[frozenset[int], list[tuple[int, ...]]] = {}
    for start in matches:
        path = [start]
        while True:
            carbonyl = mol.GetAtomWithIdx(path[-1][2])
            following = [match for atom in carbonyl.GetNeighbors()
                         for match in by_n.get(atom.GetIdx(), [])]
            if len(following) != 1:
                break
            next_match = following[0]
            if next_match == start:
                if len(path) >= 2:
                    cycles.setdefault(frozenset(m[0] for m in path), path)
                break
            if next_match in path:
                break
            path.append(next_match)
    if len(cycles) != 1:
        raise HELMParseError("Expected one unambiguous head-to-tail alpha-peptide backbone")
    backbone = next(iter(cycles.values()))
    bonds = []
    for index, residue in enumerate(backbone):
        next_n = backbone[(index + 1) % len(backbone)][0]
        bond = mol.GetBondBetweenAtoms(residue[2], next_n)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise HELMParseError("Peptide connection must be a single bond")
        bonds.append(bond.GetIdx())
    fragmented = Chem.FragmentOnBonds(mol, bonds, addDummies=True)
    for atom in fragmented.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbor = atom.GetNeighbors()[0]
            atom.SetIsotope(0)
            atom.SetAtomMapNum(1 if neighbor.GetAtomicNum() == 7 else 2)
    atom_fragments = Chem.GetMolFrags(fragmented)
    fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
    if len(fragments) != len(backbone):
        raise HELMParseError("Cross-linked or disconnected peptide is not supported")
    by_backbone_n = {}
    for indices, fragment in zip(atom_fragments, fragments, strict=True):
        labels = [a.GetAtomMapNum() for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
        if sorted(labels) != [1, 2]:
            raise HELMParseError("Each cyclic residue must have exactly R1 and R2")
        for residue in backbone:
            if residue[0] in indices:
                by_backbone_n[residue[0]] = Chem.MolToSmiles(fragment, isomericSmiles=True)
    names = ["[" + by_backbone_n[residue[0]] + "]" for residue in backbone]
    return cyclic_peptide_helm(names, {})


def _monomer_from_smiles(symbol: str, polymer_type: str, smiles: str) -> HELMMonomer:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise HELMParseError(f"Could not parse monomer {symbol!r} SMILES: {smiles}")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise HELMParseError(f"Could not sanitize monomer {symbol!r}: {smiles}") from exc
    rgroup_atoms: dict[int, int] = {}
    for atom in mol.GetAtoms():
        label = int(atom.GetAtomMapNum())
        if label > 0:
            if label in rgroup_atoms:
                raise HELMParseError(f"Duplicate R{label} in monomer {symbol!r}")
            rgroup_atoms[label] = int(atom.GetIdx())
    if not rgroup_atoms:
        raise HELMParseError(f"Monomer {symbol!r} has no atom-mapped R-groups")
    return HELMMonomer(symbol, polymer_type, mol, rgroup_atoms)


def _custom_monomer_from_smiles(smiles: str, polymer_type: str) -> HELMMonomer:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HELMParseError(f"Could not parse inline monomer SMILES: {smiles}")

    rgroup_atoms: dict[int, int] = {}
    for atom in mol.GetAtoms():
        rgroup = _dummy_atom_rgroup(atom)
        if rgroup is not None:
            if rgroup in rgroup_atoms:
                raise HELMParseError(f"Duplicate R{rgroup} in inline monomer")
            rgroup_atoms[rgroup] = int(atom.GetIdx())
    if not rgroup_atoms:
        raise HELMParseError(f"Inline monomer has no R-group dummy atoms: {smiles}")
    return HELMMonomer(smiles, polymer_type, mol, rgroup_atoms)


def _dummy_atom_rgroup(atom) -> int | None:
    if atom.GetAtomicNum() != 0:
        return None
    if atom.GetAtomMapNum() > 0:
        return int(atom.GetAtomMapNum())
    if atom.GetIsotope() > 0:
        return int(atom.GetIsotope())
    for prop in ("atomLabel", "dummyLabel"):
        if atom.HasProp(prop):
            value = atom.GetProp(prop).lstrip("_")
            if value.startswith("R") and value[1:].isdigit():
                return int(value[1:])
    return None


def _split_helm_sections(helm: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for char in helm:
        if char in "[({":
            bracket_depth += 1
        elif char in "])}" and bracket_depth > 0:
            bracket_depth -= 1
        if char == "$" and bracket_depth == 0:
            sections.append("".join(current))
            current = []
        else:
            current.append(char)
    sections.append("".join(current))
    if bracket_depth:
        raise HELMParseError("Unbalanced HELM brackets")
    return sections


def _split_outside_brackets(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for char in text:
        if char in "[({":
            bracket_depth += 1
        elif char in "])}" and bracket_depth > 0:
            bracket_depth -= 1
        if char == separator and bracket_depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _parse_polymer(polymer: str) -> tuple[str, str]:
    prefix, sep, rest = polymer.partition("{")
    if not sep or not rest.endswith("}"):
        raise HELMParseError(f"Invalid polymer section: {polymer}")
    match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", prefix)
    if match is None:
        raise HELMParseError(f"Invalid polymer identifier: {prefix}")
    return match[1], rest[:-1]


def _split_sequence(sequence: str) -> list[str]:
    parts = _split_outside_brackets(sequence, ".")
    if not all(parts):
        raise HELMParseError("Empty peptide residue")
    return parts


def _resolve_monomer(
    name: str,
    polymer_type: str,
    monomer_library: HELMMonomerLibrary,
) -> HELMMonomer:
    if name.startswith("[") and name.endswith("]"):
        inner = name[1:-1]
        if inner in monomer_library.get(polymer_type, {}):
            return monomer_library[polymer_type][inner]
        return _custom_monomer_from_smiles(inner, polymer_type)
    try:
        return monomer_library[polymer_type][name]
    except KeyError as exc:
        raise HELMParseError(f"Unknown {polymer_type} monomer: {name}") from exc


def _assemble_peptide(
    monomers: list[HELMMonomer], connections: list[tuple[int, int, int, int]]
):
    combo = Chem.RWMol()
    offsets: list[int] = []
    for monomer in monomers:
        offsets.append(combo.GetNumAtoms())
        combo.InsertMol(monomer.mol)

    atoms_to_remove: set[int] = set()
    for left_idx, left_label, right_idx, right_label in connections:
        left = monomers[left_idx]
        right = monomers[right_idx]
        left_r2 = _global_rgroup_atom(left, offsets[left_idx], left_label)
        right_r1 = _global_rgroup_atom(right, offsets[right_idx], right_label)
        if left_r2 in atoms_to_remove or right_r1 in atoms_to_remove:
            raise HELMParseError("An attachment site cannot be used twice")
        left_attach = _single_neighbor(combo, left_r2)
        right_attach = _single_neighbor(combo, right_r1)
        if left_attach == right_attach or combo.GetBondBetweenAtoms(left_attach, right_attach):
            raise HELMParseError("Connection would create a duplicate or self bond")
        combo.AddBond(left_attach, right_attach, Chem.BondType.SINGLE)
        atoms_to_remove.update((left_r2, right_r1))

    for monomer, offset in zip(monomers, offsets, strict=True):
        for local_idx in monomer.rgroup_atoms.values():
            atom_idx = offset + int(local_idx)
            if combo.GetAtomWithIdx(atom_idx).GetAtomicNum() == 1:
                atoms_to_remove.add(atom_idx)

    for atom in combo.GetAtoms():
        atom.SetAtomMapNum(0)
        for prop in ("atomLabel", "dummyLabel", "molFileValue"):
            if atom.HasProp(prop):
                atom.ClearProp(prop)

    for atom_idx in sorted(atoms_to_remove, reverse=True):
        combo.RemoveAtom(atom_idx)

    mol = combo.GetMol()
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise HELMParseError("Unconsumed inline attachment dummy atom")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise HELMParseError(f"Invalid assembled peptide: {exc}") from exc
    return mol


def _global_rgroup_atom(monomer: HELMMonomer, offset: int, rgroup: int) -> int:
    try:
        return offset + int(monomer.rgroup_atoms[rgroup])
    except KeyError as exc:
        raise HELMParseError(
            f"Monomer {monomer.symbol!r} does not have R{rgroup}"
        ) from exc


def _single_neighbor(mol, atom_idx: int) -> int:
    neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
    if len(neighbors) != 1:
        raise HELMParseError(f"R-group atom {atom_idx} has {len(neighbors)} neighbors")
    return int(neighbors[0])
