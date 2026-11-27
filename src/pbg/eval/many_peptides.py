"""Many Peptides test systems and their reference MD trajectories with the fitted tICA projections."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pbg.core.ensemble import TicaParameters, Trajectory
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_smiles

__all__ = [
    "DEFAULT_REFERENCE_ROOT",
    "ManyPeptidesSystem",
    "discover_many_peptides_test_systems",
    "load_many_peptides_test_reference",
]

# Root of the Many Peptides reference trajectories (``test/<n>AA/<SEQ>_subsampled.npz``).
DEFAULT_REFERENCE_ROOT = Path(
    os.environ.get("PBG_MANY_PEPTIDES_REFERENCE", "data/many-peptides-md/trajectories_subsampled")
)


@dataclass(frozen=True, slots=True)
class ManyPeptidesSystem:
    sequence: str
    size: str
    reference_path: Path


def discover_many_peptides_test_systems(
    reference_root: Path = DEFAULT_REFERENCE_ROOT, *, sizes: Sequence[str] | None = None
) -> list[ManyPeptidesSystem]:
    """Test systems sorted by size label and sequence; ``sizes`` keeps only those labels (``2AA``, ...)."""
    test_root = Path(reference_root).expanduser() / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"many-peptides test directory not found: {test_root}")
    systems = [
        ManyPeptidesSystem(path.name.removesuffix("_subsampled.npz"), size_dir.name, path)
        for size_dir in sorted(p for p in test_root.iterdir() if p.is_dir())
        if sizes is None or size_dir.name in {str(size) for size in sizes}
        for path in sorted(size_dir.glob("*_subsampled.npz"))
    ]
    if not systems:
        raise FileNotFoundError(f"no many-peptides test systems found under {test_root}")
    return systems


def load_many_peptides_test_reference(reference_npz_path: Path, *, sequence: str) -> Trajectory:
    """Reference frames of ``sequence`` (Zxyz in nm, mapped-SMILES atom order) with energies and tICA fit."""
    mapped_smiles = sequence_to_smiles(sequence)
    atomic_numbers = atom_numbers_from_smiles(mapped_smiles)
    with np.load(reference_npz_path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (atomic_numbers.size, 3):
            raise ValueError(
                f"{reference_npz_path}: positions {positions.shape} do not fit the {atomic_numbers.size} atoms"
                f" of {sequence}"
            )
        energy = np.asarray(data["energy"], dtype=np.float64) if "energy" in data.files else None
        tica = TicaParameters(
            mean=np.asarray(data["tica_mean"], dtype=np.float64),
            projection=np.asarray(data["tica_projection"], dtype=np.float64),
            dim=np.asarray(data["tica_dim"]),
        )
    samples = np.empty((*positions.shape[:2], 4), dtype=np.float64)
    samples[:, :, 0] = atomic_numbers[None, :]
    samples[:, :, 1:4] = positions
    return Trajectory(
        samples=samples,
        smiles=mapped_smiles,
        id=reference_npz_path.stem,
        energy=energy,
        ensemble_metadata={"sequence": sequence},
        tica=tica,
    )
