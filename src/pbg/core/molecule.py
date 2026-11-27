"""Atomic numbers of mapped SMILES, and checks that zxyz coordinates match them."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from beartype import beartype
from rdkit import Chem
from rdkit.Chem import rdmolops


def _context_suffix(context: str | Path | None) -> str:
    if context is None:
        return ""
    return f"; context={context!r}"


def _explicit_h_mol(smiles: str):
    """Parse the first word of ``smiles`` and add H; None if RDKit cannot parse or sanitize it.

    When the atom maps are exactly ``1..N``, atom ``i`` is map ``i + 1``;
    otherwise the atoms keep the RDKit order.
    """
    mol = Chem.MolFromSmiles(smiles.split()[0], sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException:
        return None
    mol = Chem.AddHs(mol)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if sorted(maps) != list(range(1, len(maps) + 1)):
        return mol
    return rdmolops.RenumberAtoms(mol, sorted(range(len(maps)), key=maps.__getitem__))


def mapped_mol(smiles: str):
    """Return the explicit-H molecule of ``smiles`` with its atoms in atom-map order.

    SMILES without atom maps keep the RDKit atom order. Raises ``ValueError``
    for invalid SMILES and for maps that are not positive and unique on every
    atom. The maps need not be contiguous (a solvated peptide can number its
    waters from 999).
    """
    mol = _explicit_h_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES or sanitization failed: {smiles!r}")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if not any(maps) or maps == sorted(maps) and maps[0] > 0 and len(set(maps)) == len(maps):
        return mol
    if min(maps) <= 0 or len(set(maps)) != len(maps):
        raise ValueError(f"Mapped SMILES need a unique positive atom map on every atom: {smiles!r}")
    return rdmolops.RenumberAtoms(mol, sorted(range(len(maps)), key=maps.__getitem__))


@lru_cache(maxsize=32768)
@beartype
def atom_numbers_from_smiles(smiles: str) -> np.ndarray:
    """Return the read-only int32 atomic numbers of a SMILES, H included.

    Mapped SMILES keep their 1..N order (``_explicit_h_mol``).
    """
    mol = _explicit_h_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES or sanitization failed: {smiles!r}")
    zs = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
    zs.flags.writeable = False
    return zs


@beartype
def check_smiles_is_indexed(
    smiles: str,
    *,
    context: str | Path | None = None,
) -> None:
    """Require explicit-H atom maps to cover every atom exactly once as ``1..N``."""
    mol = _explicit_h_mol(smiles)
    if mol is None:
        raise ValueError(
            f"Invalid SMILES or sanitization failed: {smiles!r}{_context_suffix(context)}"
        )
    maps = [
        int(mol.GetAtomWithIdx(i).GetAtomMapNum()) for i in range(mol.GetNumAtoms())
    ]
    expected = list(range(1, mol.GetNumAtoms() + 1))
    if maps != expected:
        raise ValueError(
            "SMILES must be fully atom-indexed in explicit-H order with maps 1..N: "
            f"maps={maps}, expected={expected}, smiles={smiles!r}"
            f"{_context_suffix(context)}"
        )


@beartype
def check_z_column_fixed_across_frames(
    samples: np.ndarray,
    *,
    context: str | Path | None = None,
) -> None:
    """Require ``samples[k, :, 0]`` (rounded Z) identical for all frame indices ``k``."""
    if samples.ndim != 3 or samples.shape[2] != 4:
        raise ValueError(
            "samples must have shape (K, N, 4) zxyz; "
            f"got {samples.shape}{_context_suffix(context)}"
        )
    z0 = np.rint(samples[0, :, 0]).astype(np.int64)
    for k in range(1, samples.shape[0]):
        zk = np.rint(samples[k, :, 0]).astype(np.int64)
        if not np.array_equal(z0, zk):
            raise ValueError(
                "Atomic number column (index 0) differs between frame 0 and "
                f"frame {k}{_context_suffix(context)}"
            )


@beartype
def check_smiles_z_matches_frame_z(
    smiles: str,
    samples: np.ndarray,
    *,
    context: str | Path | None = None,
) -> None:
    """Require RDKit explicit-H atomic numbers to match ``rint(samples[0, :, 0])`` (no tokeniser)."""
    if samples.ndim != 3 or samples.shape[2] != 4:
        raise ValueError(
            "samples must have shape (K, N, 4) zxyz; "
            f"got {samples.shape}{_context_suffix(context)}"
        )
    z_frame = np.rint(samples[0, :, 0]).astype(np.int64)
    try:
        z_smiles = atom_numbers_from_smiles(smiles)
    except ValueError as exc:
        raise ValueError(f"{exc}{_context_suffix(context)}") from exc
    if z_smiles.shape[0] != z_frame.shape[0]:
        raise ValueError(
            f"Atom count mismatch: SMILES/RDKit has {z_smiles.shape[0]} atoms, "
            f"frame 0 has {z_frame.shape[0]}{_context_suffix(context)}"
        )
    if not np.array_equal(z_smiles, z_frame):
        raise ValueError(
            f"Z from SMILES (RDKit explicit H) != Z from frame 0: "
            f"z_smiles={z_smiles.tolist()}, z_frame={z_frame.tolist()}, "
            f"smiles={smiles!r}{_context_suffix(context)}"
        )
