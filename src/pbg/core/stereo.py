"""Fast stereoisomer sorting for frames with known topology."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem

from pbg.core.molecule import mapped_mol


class StereoSpec(NamedTuple):
    """Tetrahedral probes compiled from a molecular graph."""

    atoms: jax.Array  # (centers, 4): center followed by three ordered neighbors
    parity: jax.Array  # sign mapping from local orientation to S


@lru_cache(maxsize=4096)
def compile_stereo_spec(smiles: str) -> StereoSpec:
    """Compile explicit-H mapped SMILES into tetrahedral geometry probes."""
    mol = mapped_mol(smiles)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    center_indices = {
        int(info.centeredOn)
        for info in Chem.FindPotentialStereo(mol)
        if info.type == Chem.StereoType.Atom_Tetrahedral
    }
    if len(center_indices) > 32:
        raise ValueError("Packed stereo keys currently support at most 32 tetrahedral centers")
    probes: list[tuple[int, int, int, int]] = []
    parity: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in center_indices:
            continue
        neighbors = sorted(atom.GetNeighbors(), key=lambda neighbor: neighbor.GetIdx())
        if len(neighbors) != 4:
            raise ValueError(
                f"Stereo center {atom.GetIdx()} does not have four explicit substituents; "
                "coordinates and topology must include explicit hydrogens"
            )
        if any(not neighbor.HasProp("_CIPRank") for neighbor in neighbors):
            raise ValueError(f"RDKit did not assign CIP ranks around center {atom.GetIdx()}")
        ranks = [int(neighbor.GetProp("_CIPRank")) for neighbor in neighbors]
        if len(set(ranks)) != 4:
            raise ValueError(f"Non-unique CIP priorities around center {atom.GetIdx()}")
        priority_order = np.argsort(ranks)[::-1]
        permutation = tuple(int(index) for index in priority_order)
        inversions = sum(permutation[i] > permutation[j] for i in range(4) for j in range(i + 1, 4))
        probes.append((atom.GetIdx(), *(neighbor.GetIdx() for neighbor in neighbors[:3])))
        parity.append(1 if inversions % 2 == 0 else -1)

    return StereoSpec(
        atoms=jnp.asarray(probes, dtype=jnp.int32).reshape((-1, 4)),
        parity=jnp.asarray(parity, dtype=jnp.int8),
    )


@jax.jit
def stereo_keys(
    spec: StereoSpec,
    xyz: jax.Array,
    tolerance: float = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    """Return packed S-configuration and resolved masks for every frame.

    Bit ``j`` is one in the configuration key when center ``j`` is S. A
    cleared bit denotes R when the corresponding resolved bit is one.
    """
    points = xyz[:, spec.atoms]
    center, a, b, c = (points[:, :, i] for i in range(4))
    u, v, w = a - center, b - center, c - center
    volume = jnp.sum(u * jnp.cross(v, w), axis=-1)
    scale = jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1) * jnp.linalg.norm(w, axis=-1)
    orientation = volume / jnp.maximum(scale, jnp.finfo(xyz.dtype).tiny)
    resolved = jnp.isfinite(orientation) & (jnp.abs(orientation) > tolerance)
    is_s = orientation * spec.parity[None, :] > 0
    powers = jnp.left_shift(jnp.uint32(1), jnp.arange(spec.atoms.shape[0], dtype=jnp.uint32))
    config_key = jnp.sum((is_s & resolved).astype(jnp.uint32) * powers, axis=1, dtype=jnp.uint32)
    resolved_key = jnp.sum(resolved.astype(jnp.uint32) * powers, axis=1, dtype=jnp.uint32)
    return config_key, resolved_key


def stereo_groups(
    config_key: np.ndarray,
    resolved_key: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group packed keys, returning unique config/resolved keys, inverse, counts."""
    config = np.asarray(config_key, dtype=np.uint32)
    resolved = np.asarray(resolved_key, dtype=np.uint32)
    if config.shape != resolved.shape or config.ndim != 1:
        raise ValueError("config_key and resolved_key must be equal-length vectors")
    keys = np.empty(config.size, dtype=[("resolved", np.uint32), ("config", np.uint32)])
    keys["resolved"], keys["config"] = resolved, config
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return unique["config"], unique["resolved"], inverse, counts


def key_to_mapped_smiles(smiles: str, spec: StereoSpec, config_key: int, resolved_key: int) -> str | None:
    """Render a fully resolved key as explicit-H mapped SMILES in frame order."""
    mol = _mol_for_stereo_key(smiles, spec, config_key, resolved_key)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True, isomericSmiles=True)


def configured_stereo_key(smiles: str, spec: StereoSpec) -> int | None:
    """Return the fully specified S/R key encoded by a topology, if present."""
    mol = mapped_mol(smiles)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    key = 0
    for probe_index, center in enumerate(np.asarray(spec.atoms)[:, 0]):
        atom = mol.GetAtomWithIdx(int(center))
        if not atom.HasProp("_CIPCode"):
            return None
        if atom.GetProp("_CIPCode") == "S":
            key |= 1 << probe_index
    return key


def intended_stereo_mask(
    smiles: str,
    xyz: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Return finite frames that match the topology's intended stereochemistry."""
    coordinates = np.asarray(xyz, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"xyz must have shape (frames, atoms, 3), got {coordinates.shape}"
        )
    finite = np.isfinite(coordinates).all(axis=(1, 2))
    spec = compile_stereo_spec(smiles)
    n_centers = int(spec.atoms.shape[0])
    if n_centers == 0:
        return finite

    points = coordinates[:, np.asarray(spec.atoms, dtype=np.int64)]
    center, a, b, c = (points[:, :, index] for index in range(4))
    u, v, w = a - center, b - center, c - center
    volume = np.sum(u * np.cross(v, w), axis=-1)
    scale = (
        np.linalg.norm(u, axis=-1)
        * np.linalg.norm(v, axis=-1)
        * np.linalg.norm(w, axis=-1)
    )
    orientation = volume / np.maximum(scale, np.finfo(coordinates.dtype).tiny)
    centers_resolved = np.isfinite(orientation) & (np.abs(orientation) > tolerance)
    is_s = orientation * np.asarray(spec.parity)[None, :] > 0
    powers = np.left_shift(
        np.uint32(1), np.arange(n_centers, dtype=np.uint32)
    )
    config = np.sum(
        (is_s & centers_resolved).astype(np.uint32) * powers,
        axis=1,
        dtype=np.uint32,
    )
    resolved = np.sum(
        centers_resolved.astype(np.uint32) * powers,
        axis=1,
        dtype=np.uint32,
    )
    full_key = (1 << n_centers) - 1
    valid = finite & (resolved == full_key)
    intended_key = configured_stereo_key(smiles, spec)
    if intended_key is not None:
        valid &= config == intended_key
    return valid


def _mol_for_stereo_key(smiles: str, spec: StereoSpec, config_key: int, resolved_key: int):
    n_centers = int(spec.atoms.shape[0])
    if resolved_key != (1 << n_centers) - 1:
        return None
    mol = mapped_mol(smiles)
    for probe_index, center in enumerate(np.asarray(spec.atoms)[:, 0]):
        desired = "S" if config_key & (1 << probe_index) else "R"
        atom = mol.GetAtomWithIdx(int(center))
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != desired:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


