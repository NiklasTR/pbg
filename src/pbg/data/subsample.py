"""Keep the solvent molecules nearest to the solute."""

from __future__ import annotations

import numpy as np


def subsample_solvent(zxyz: np.ndarray, *, n_solute: int, k: int, atoms_per_solvent: int = 3) -> np.ndarray:
    """Keep the solute and the ``k`` solvent molecules nearest to it, per frame.

    ``zxyz`` is ``(N, 4)`` or ``(K, N, 4)`` with the solute in the first
    ``n_solute`` rows. The distance of a solvent molecule is from its centroid
    to the nearest solute atom. Kept molecules stay in their stored order.
    """
    single = zxyz.ndim == 2
    if single:
        zxyz = zxyz[np.newaxis]
    if zxyz.ndim != 3 or zxyz.shape[-1] != 4:
        raise ValueError(f"Expected shape (K, N, 4) or (N, 4), got {zxyz.shape}")
    n_frames, n_total = zxyz.shape[:2]
    n_solvent_atoms = n_total - n_solute
    if n_solvent_atoms < 0:
        raise ValueError(f"n_solute ({n_solute}) exceeds total atoms ({n_total})")
    if n_solvent_atoms % atoms_per_solvent:
        raise ValueError(
            f"Solvent atom count ({n_solvent_atoms}) is not divisible by atoms_per_solvent ({atoms_per_solvent})"
        )
    n_molecules = n_solvent_atoms // atoms_per_solvent
    if k > n_molecules:
        raise ValueError(f"Requested k={k} solvent molecules but only {n_molecules} available")
    if k < n_molecules:
        centroids = zxyz[:, n_solute:, 1:4].reshape(n_frames, n_molecules, atoms_per_solvent, 3).mean(axis=2)
        solute = zxyz[:, :n_solute, 1:4]
        distances = np.linalg.norm(centroids[:, :, None, :] - solute[:, None, :, :], axis=-1).min(axis=-1)
        nearest = np.sort(np.argpartition(distances, k, axis=1)[:, :k], axis=1)
        rows = n_solute + (nearest[:, :, None] * atoms_per_solvent + np.arange(atoms_per_solvent)).reshape(n_frames, -1)
        zxyz = np.concatenate([zxyz[:, :n_solute], zxyz[np.arange(n_frames)[:, None], rows]], axis=1)
    return zxyz[0] if single else zxyz
