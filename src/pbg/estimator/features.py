"""Collective variables that the Wasserstein metrics compare: torsion angles and tICA coordinates.

``samples`` are frames of shape (frames, atoms, 4) with columns Z, x, y, z in nm; the torsion
features also accept (frames, atoms, 3) coordinates.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np

from pbg.core.ensemble import Trajectory
from pbg.core.molecule import (
    check_smiles_z_matches_frame_z,
    check_z_column_fixed_across_frames,
)
from pbg.core.peptide import (
    infer_linear_peptide_sequence_from_smiles,
    sequence_to_pdb_block,
)

__all__ = ["backbone_torsions", "global_torsions", "tica_projection"]


def backbone_torsions(samples: Any, topology: Any) -> np.ndarray:
    """All MDTraj backbone phi angles followed by all psi angles, in radians."""
    import mdtraj as md

    xyz = np.asarray(samples, dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[2] not in {3, 4}:
        raise ValueError(f"samples must have shape (frames, atoms, 3|4), got {xyz.shape}")
    trajectory = md.Trajectory(xyz[:, :, -3:], topology)
    _, phi = md.compute_phi(trajectory)
    _, psi = md.compute_psi(trajectory)
    if phi.shape[1] == 0 or psi.shape[1] == 0:
        raise ValueError("could not compute backbone phi/psi angles")
    return np.concatenate([phi, psi], axis=1).astype(np.float64)


def global_torsions(samples: Any, smiles: str) -> np.ndarray:
    """Every dihedral angle i-j-k-l around a bond j-k between two heavy atoms, in radians.

    The torsions follow the atoms of the mapped ``smiles`` (one per atom quadruple, sorted), so
    the frames must use the SMILES atom order. A molecule without such a torsion gives shape
    (frames, 0).
    """
    from rdkit import Chem

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] not in {3, 4}:
        raise ValueError(f"samples must have shape (frames, atoms, 3|4), got {values.shape}")
    xyz = values[:, :, -3:]
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        raise ValueError(f"RDKit could not parse mapped SMILES: {smiles!r}")
    torsions: set[tuple[int, int, int, int]] = set()
    for bond in molecule.GetBonds():
        j, k = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        if molecule.GetAtomWithIdx(j).GetAtomicNum() == 1 or molecule.GetAtomWithIdx(k).GetAtomicNum() == 1:
            continue
        for i in (int(atom.GetIdx()) for atom in molecule.GetAtomWithIdx(j).GetNeighbors()):
            for last in (int(atom.GetIdx()) for atom in molecule.GetAtomWithIdx(k).GetNeighbors()):
                if k != i != last != j:
                    torsions.add(min((i, j, k, last), (last, k, j, i)))
    indices = np.asarray(sorted(torsions), dtype=np.int64).reshape(-1, 4)

    p0, p1, p2, p3 = (xyz[:, indices[:, column], :] for column in range(4))
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
    b1_unit = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), np.finfo(np.float64).tiny)
    v = b0 - np.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
    return np.arctan2(np.sum(np.cross(b1_unit, v, axis=-1) * w, axis=-1), np.sum(v * w, axis=-1))


def tica_projection(samples: Any, reference: Trajectory, *, dim: int = 2) -> np.ndarray:
    """The first ``dim`` tICA coordinates of each frame, with the tICA fitted to the reference MD.

    The features are the pairwise distances of the heavy atoms (``heavy_pairs``), or of the C, N
    and S atoms followed by sin and cos of the backbone phi, psi and omega angles
    (``cns_dihedrals``), as ``reference.tica.featurization`` says.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError(f"samples must have shape (frames, atoms, 4), got {values.shape}")
    check_z_column_fixed_across_frames(values, context="tica_projection")
    check_smiles_z_matches_frame_z(reference.smiles, values, context="tica_projection")
    tica = reference.tica
    angles: list[np.ndarray] = []
    if tica.featurization == "heavy_pairs":
        points = values[:, np.flatnonzero(np.rint(values[0, :, 0]).astype(np.int64) != 1), 1:4]
    elif tica.featurization == "cns_dihedrals":
        import mdtraj as md
        from openmm.app import PDBFile

        sequence = infer_linear_peptide_sequence_from_smiles(reference.smiles)
        topology = md.Topology.from_openmm(PDBFile(StringIO(sequence_to_pdb_block(sequence))).topology)
        trajectory = md.Trajectory(values[:, :, 1:4].astype(np.float32), topology=topology)
        selection = trajectory.top.select("symbol == C or symbol == N or symbol == S")
        points = trajectory.xyz[:, selection, :].astype(np.float64)
        for compute in (md.compute_phi, md.compute_psi, md.compute_omega):
            _, dihedrals = compute(trajectory)
            if dihedrals.size > 0:
                angles.extend([np.sin(dihedrals), np.cos(dihedrals)])
    else:
        raise ValueError(f"unknown tICA featurization {tica.featurization!r}")
    if points.shape[1] < 2:
        raise ValueError(f"{tica.featurization} featurization needs at least two atoms")
    rows, columns = np.triu_indices(points.shape[1], k=1)
    distances = np.linalg.norm(points[:, None, :, :] - points[:, :, None, :], axis=-1)[:, rows, columns]
    features = np.concatenate([distances, *angles], axis=-1)

    mean = np.asarray(tica.mean, dtype=np.float64).reshape(-1)
    projection = np.asarray(tica.projection, dtype=np.float64)
    if features.shape[1] != mean.shape[0] or projection.shape[0] != mean.shape[0]:
        raise ValueError(
            f"{features.shape[1]} features do not match the tICA mean {mean.shape} and projection {projection.shape}"
        )
    if not 1 <= dim <= min(projection.shape[1], int(np.asarray(tica.dim).item())):
        raise ValueError(f"dim={dim} exceeds the stored tICA dimension {int(np.asarray(tica.dim).item())}")
    return (features - mean.reshape(1, -1)) @ projection[:, :dim]
