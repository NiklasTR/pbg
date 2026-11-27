"""One prompt-derived topology and traversal contract for all training systems.

The v1 convention clears map labels, writes and reparses canonical isomeric
SMILES, then adds H. Solvent waters follow the solute as ordered O/H/H blocks.
Linear PEPTIDE graph construction uses the registered ff14SB sequence rules;
v1 keeps the ff14SB atom order for linear peptides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from rdkit import Chem, rdBase

from pbg.core.helm import (
    join_peptide,
    load_helm_monomer_library,
    mol_from_helm,
    rewrite_helm_residues,
    split_peptide,
)
from pbg.core.peptide import (
    linear_context,
    linear_sequence_from_helm,
    linear_state_token,
    sequence_to_smiles,
)

ATOM_ORDER_VERSION = 'prompt_canonical_v1'
ATOM_ORDER_IMPLEMENTATION = f'{ATOM_ORDER_VERSION}:rdkit-{rdBase.rdkitVersion}:linear-ff14sb-v1'
# Checkpoints store the implementation string; the config compares it with this one.
ATOM_ORDER_IMPLEMENTATIONS = {ATOM_ORDER_VERSION: ATOM_ORDER_IMPLEMENTATION}


@lru_cache(maxsize=32768)
def ordered_mol(mapped_smiles: str):
    """Parse a complete explicit-H graph and number atoms by coordinate row."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(mapped_smiles, params)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid mapped topology')
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if sorted(maps) != list(range(1, mol.GetNumAtoms() + 1)):
        raise ValueError('Topology requires unique explicit atom maps 1..N')
    if Chem.AddHs(mol).GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError('Topology must include all hydrogen coordinate rows')
    return Chem.RenumberAtoms(mol, sorted(range(len(maps)), key=lambda i: maps[i]))


@dataclass(frozen=True)
class PromptTopology:
    """An ordered graph: coordinate row j is atom map j+1 in ``mapped_smiles``.

    ``atoms`` contains (atomic number, formal charge, isotope) in traversal
    order. ``bonds`` contains zero-based atom indices and RDKit bond types.
    Isomeric mapped SMILES retains the graph's stereochemistry. The optional
    mapping returned by ``source_rows`` connects this traversal to stored data.
    """

    canonical_smiles: str
    mapped_smiles: str
    atoms: tuple[tuple[int, int, int], ...] = field(hash=False)
    bonds: tuple[tuple[int, int, str], ...] = field(hash=False)
    n_solute_atoms: int
    n_waters: int
    order: str = ATOM_ORDER_VERSION

    @property
    def traversal(self) -> tuple[int, ...]:
        """Atom-map IDs in coordinate-token order, including H and solvent."""
        return tuple(range(1, len(self.atoms) + 1))

    @staticmethod
    def applies_to(helm: str) -> bool:
        return helm.startswith(('CHEM', 'PEPTIDE'))

    @classmethod
    @lru_cache(maxsize=32768)
    def compile(cls, helm: str, environment: str = 'vacuum', order: str = ATOM_ORDER_VERSION) -> PromptTopology:
        """Compile prompt fields into a graph with one explicit atom traversal."""
        if order not in ATOM_ORDER_IMPLEMENTATIONS:
            raise ValueError(f'Unknown prompt atom order: {order!r}')
        counts = []
        for item in ''.join(environment.split()).split(','):
            match = re.fullmatch(r'(\d+)water', item)
            if match:
                counts.append(int(match[1]))
            elif item == 'explicit_water' or (item.endswith('water') and item not in {'water', 'implicit_water'}):
                raise ValueError('Explicit-water topology needs a nonnegative water count')
        if len(counts) > 1:
            raise ValueError('Prompt contains multiple water counts')
        n_waters = counts[0] if counts else 0
        if helm.startswith('CHEM1{[') and helm.endswith(']}$$$$'):
            solute = Chem.MolFromSmiles(helm[len('CHEM1{['):-len(']}$$$$')])
            if solute is None:
                raise ValueError('Invalid CHEM prompt')
        elif helm.startswith('PEPTIDE'):
            _, residues, sections = split_peptide(helm)
            linear = not sections or not sections[0]
            sequence = None
            if linear:
                try:
                    sequence = linear_sequence_from_helm(helm)
                except ValueError:
                    pass  # a residue without an ff14SB template (noncanonical or D); see below
            if sequence is not None:
                return cls.from_mapped_solute(sequence_to_smiles(sequence), n_waters, order)
            library = load_helm_monomer_library()
            inline = helm
            if linear:
                # A linear peptide with a residue outside ff14SB: canonical residues
                # and D-residues take their ff14SB-state token (termini, side-chain
                # charges, HIE); the others keep the chemistry their token states.
                n = len(residues)
                polymer, tokens, rest = split_peptide(rewrite_helm_residues(helm, list(range(1, n + 1)), library))
                tokens = [linear_state_token(token, linear_context(i, n)) for i, token in enumerate(tokens)]
                inline = join_peptide(polymer, tokens, rest)
            # v1 keeps the whole-graph canonical order for every peptide outside ff14SB.
            solute = mol_from_helm(inline, library)
        else:
            raise ValueError('Expected CHEM or PEPTIDE prompt')
        for atom in solute.GetAtoms():
            atom.SetAtomMapNum(0)
        solute = Chem.RemoveHs(solute)
        canonical = Chem.MolToSmiles(solute, canonical=True, isomericSmiles=True)
        solute = Chem.AddHs(Chem.MolFromSmiles(canonical))
        if any(a.GetAtomicNum() == 0 for a in solute.GetAtoms()):
            raise ValueError('Prompt contains unresolved attachment atoms')
        for i, atom in enumerate(solute.GetAtoms(), 1):
            atom.SetAtomMapNum(i)
        mapped = Chem.MolToSmiles(solute, canonical=False, allHsExplicit=True, isomericSmiles=True)
        return cls.from_mapped_solute(mapped, n_waters, order)

    @classmethod
    @lru_cache(maxsize=32768)
    def from_mapped_solute(cls, mapped: str, n_waters: int = 0, order: str = ATOM_ORDER_VERSION) -> PromptTopology:
        """Cache the common graph build after representation normalization."""
        if type(n_waters) is not int or n_waters < 0:
            raise ValueError('Water count must be a nonnegative integer')
        solute = Chem.Mol(ordered_mol(mapped))
        n_solute = solute.GetNumAtoms()
        for atom in solute.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(Chem.RemoveHs(solute), canonical=True, isomericSmiles=True)
        waters = [f'[O:{i}]([H:{i + 1}])[H:{i + 2}]' for i in range(n_solute + 1, n_solute + 3 * n_waters + 1, 3)]
        mapped = '.'.join([mapped, *waters])
        graph = ordered_mol(mapped)
        atoms = tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in graph.GetAtoms())
        bonds = tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
            max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType())) for b in graph.GetBonds()))
        return cls(canonical, mapped, atoms, bonds, n_solute, n_waters, order)

    @lru_cache(maxsize=32768)
    def agrees_with(self, mapped_smiles: str) -> bool:
        """Compare indexed graphs, including stereo, without relying on text order."""
        if mapped_smiles == self.mapped_smiles:
            return True
        return Chem.MolToSmiles(ordered_mol(mapped_smiles), canonical=True, isomericSmiles=True) == Chem.MolToSmiles(
            ordered_mol(self.mapped_smiles), canonical=True, isomericSmiles=True)

    @lru_cache(maxsize=32768)
    def source_rows(self, mapped_smiles: str) -> tuple[int, ...]:
        """Return zero-based source rows in canonical traversal order.

        Equivalent atoms use the first stereo-preserving RDKit match on the
        map-sorted source graph. Removed H on each matched atom use ascending
        source row. This depends on the source graph, never on the HELM start
        or the coordinates. Appended water blocks retain their input order.
        """
        if mapped_smiles == self.mapped_smiles:
            return tuple(range(len(self.atoms)))
        source = Chem.Mol(ordered_mol(mapped_smiles))
        target = Chem.Mol(ordered_mol(self.mapped_smiles))
        if source.GetNumAtoms() != len(self.atoms):
            raise ValueError('Prompt topology and coordinates have different atom counts')
        if self.agrees_with(mapped_smiles):
            return tuple(range(len(self.atoms)))
        source_solute, target_solute = Chem.RWMol(source), Chem.RWMol(target)
        for i in range(len(self.atoms) - 1, self.n_solute_atoms - 1, -1):
            source_solute.RemoveAtom(i)
            target_solute.RemoveAtom(i)
        source_solute, target_solute = source_solute.GetMol(), target_solute.GetMol()
        for graph in (source_solute, target_solute):
            for atom in graph.GetAtoms():
                atom.SetIntProp('coordinate_row', atom.GetIdx())
                atom.SetAtomMapNum(0)
            Chem.SanitizeMol(graph)
        source_core, target_core = Chem.RemoveHs(source_solute), Chem.RemoveHs(target_solute)
        if Chem.MolToSmiles(source_core, isomericSmiles=True) != Chem.MolToSmiles(target_core, isomericSmiles=True):
            raise ValueError('Prompt and stored solute differ in chemistry or stereochemistry')
        # A Mol-derived substructure query can admit a heavy-graph symmetry
        # that exchanges atoms with different protonation (e.g. n and [nH]).
        # Require the complete atom state before accepting a correspondence.
        source_states = [(a.GetFormalCharge(), a.GetIsotope(), a.GetTotalNumHs(includeNeighbors=True))
            for a in source_core.GetAtoms()]
        target_states = [(a.GetFormalCharge(), a.GetIsotope(), a.GetTotalNumHs(includeNeighbors=True))
            for a in target_core.GetAtoms()]
        parameters = Chem.SubstructMatchParameters()
        parameters.useChirality = True
        parameters.setExtraFinalCheck(lambda mol, candidate:
            all(source_states[j] == target_states[i] for i, j in enumerate(candidate)))
        match = source_core.GetSubstructMatch(target_core, parameters)
        if len(match) != target_core.GetNumAtoms():
            raise ValueError('No complete stereo-preserving solute mapping')
        rows = {target_core.GetAtomWithIdx(i).GetIntProp('coordinate_row'):
                source_core.GetAtomWithIdx(j).GetIntProp('coordinate_row') for i, j in enumerate(match)}
        core_rows = dict(rows)
        assigned_source = set(rows.values())
        for target_row, source_row in core_rows.items():
            target_h = sorted(a.GetIdx() for a in target_solute.GetAtomWithIdx(target_row).GetNeighbors()
                if a.GetAtomicNum() == 1 and a.GetIdx() not in core_rows)
            source_h = sorted(a.GetIdx() for a in source_solute.GetAtomWithIdx(source_row).GetNeighbors()
                if a.GetAtomicNum() == 1 and a.GetIdx() not in assigned_source)
            if len(target_h) != len(source_h):
                raise ValueError(f'Hydrogen mapping mismatch at target row {target_row}, source row {source_row}: '
                    f'{len(target_h)} versus {len(source_h)}; source={mapped_smiles}; target={self.mapped_smiles}')
            for i, j in zip(target_h, source_h, strict=True):
                rows[i] = j
        rows.update((i, i) for i in range(self.n_solute_atoms, len(self.atoms)))
        permutation = tuple(rows[i] for i in range(len(self.atoms)))
        if sorted(permutation) != list(range(len(self.atoms))):
            raise ValueError('Atom mapping is not a complete permutation')
        reordered = Chem.RenumberAtoms(source, permutation)
        for i, atom in enumerate(reordered.GetAtoms(), 1):
            atom.SetAtomMapNum(i)
        if Chem.MolToSmiles(reordered, isomericSmiles=True) != Chem.MolToSmiles(target, isomericSmiles=True):
            raise ValueError('Mapped traversal changed indexed bonds, solvent topology, or stereochemistry')
        return permutation

    def align_frame(self, frame):
        """Permute Z/XYZ together and keep an explicit mapping for energy scoring."""
        if frame.conformer is None:
            raise ValueError('Traversal alignment requires coordinates')
        rows = self.source_rows(frame.smiles)
        identity = rows == tuple(range(len(rows)))
        values = frame.conformer if identity else frame.conformer[np.asarray(rows, dtype=np.int64)]
        if not np.array_equal(values[:, 0], [a[0] for a in self.atoms]):
            raise ValueError('Stored atomic numbers disagree with the prompt traversal')
        if not identity and (frame.behavior_token_ids is not None or frame.behavior_token_log_probs is not None):
            raise ValueError('Cannot permute a frame with source-order behavior token labels')
        metadata = dict(frame.metadata or {})
        previous = metadata.get('source_atom_rows') if metadata.get('atom_order') in ATOM_ORDER_IMPLEMENTATIONS else None
        metadata.update(atom_order=self.order, atom_order_implementation=ATOM_ORDER_IMPLEMENTATIONS[self.order],
            source_atom_rows=[previous[i] for i in rows] if previous is not None else list(rows))
        if previous is None:
            metadata['source_mapped_smiles'] = frame.smiles
        return replace(frame, smiles=self.mapped_smiles, conformer=values, metadata=metadata)
