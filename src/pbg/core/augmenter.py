"""DataFrame-level coordinate augmentation utilities."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from rdkit import Chem

from pbg.core.frame import DataFrame

Augmenter = dict[str, float | bool | str | None]
_ROTATION_ATTEMPTS = 16


class AugmentationBoundsError(ValueError):
    """Raised when bounded augmentation cannot fit a frame in its box."""


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3x3 rotation matrix (Haar measure on SO(3))."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    sign = np.sign(np.diag(R))
    sign = np.where(sign == 0, 1.0, sign)
    Q = Q * sign
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _random_fitting_rotation_matrix(
    positions: np.ndarray,
    *,
    min_val: float,
    max_val: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Try uniform rotations until one can be translated inside the box."""
    for _ in range(_ROTATION_ATTEMPTS):
        rotation = _random_rotation_matrix(rng)
        rotated = positions @ rotation.T
        if np.all(np.ptp(rotated, axis=0) <= max_val - min_val):
            return rotation
    return None


def _random_translation_vector(
    positions: np.ndarray,
    *,
    min_val: float,
    max_val: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a uniform translation that keeps all coordinates inside the box."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    shift_min = float(min_val) - lo
    shift_max = float(max_val) - hi
    if np.any(shift_min > shift_max):
        return np.zeros(3, dtype=positions.dtype)
    return rng.uniform(low=shift_min, high=shift_max, size=3).astype(
        positions.dtype, copy=False
    )


def init_augmenter(
    *,
    center: bool | str = True,
    rotate: bool = False,
    translate: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
    seed: int | None = None,
) -> Augmenter:
    """Build a dict of augmentation knobs (see :func:`apply_augmenter`)."""
    if not (isinstance(center, bool) or center == "mass"):
        raise ValueError(f"center must be True, False or 'mass'; got {center!r}")
    return {
        "center": center,
        "rotate": rotate,
        "translate": translate,
        "min_val": min_val,
        "max_val": max_val,
        "seed": seed,
    }


def apply_augmenter(
    augmenter: Augmenter,
    frame: DataFrame,
    *,
    rng: np.random.Generator | None = None,
) -> DataFrame:
    """Centre, rotate and translate the coordinates of a frame.

    ``center`` subtracts the centroid (or, with ``"mass"``, the centre of
    mass). ``rotate`` applies a uniform random rotation among those whose axis
    extents fit the coordinate box (at most 16 draws). ``translate`` shifts
    the frame uniformly over the positions that keep it inside the box.
    """
    center = augmenter.get("center", True)
    rotate = bool(augmenter.get("rotate", False))
    translate = bool(augmenter.get("translate", False))
    if not (center or rotate or translate):
        return frame
    if frame.conformer is None:
        return frame

    conformer = np.array(frame.conformer, copy=True)
    if conformer.ndim != 2 or conformer.shape[1] != 4:
        raise ValueError("frame.conformer must have shape (n_atoms, 4) as Zxyz")

    z = conformer[:, :1]
    positions = conformer[:, 1:4]

    if center:
        # "mass" weights each atom by its standard atomic weight (ROBIN).
        weights = None
        if center == "mass":
            table = Chem.GetPeriodicTable()
            weights = [table.GetAtomicWeight(int(number)) for number in z[:, 0]]
        centroid = np.average(positions, axis=0, weights=weights)[None, :]
        positions = positions - centroid

    if rotate:
        rng = np.random.default_rng() if rng is None else rng
        min_val = augmenter.get("min_val")
        max_val = augmenter.get("max_val")
        if min_val is None or max_val is None:
            raise ValueError("rotate augmentation requires min_val and max_val")
        rotation = _random_fitting_rotation_matrix(
            positions,
            min_val=float(min_val),
            max_val=float(max_val),
            rng=rng,
        )
        if rotation is None:
            raise AugmentationBoundsError(
                f"no rotation fit box after {_ROTATION_ATTEMPTS} attempts"
            )
        positions = positions @ rotation.T

    if translate:
        min_val = augmenter.get("min_val")
        max_val = augmenter.get("max_val")
        if min_val is None or max_val is None:
            raise ValueError("translate augmentation requires min_val and max_val")
        rng = np.random.default_rng() if rng is None else rng
        positions = positions + _random_translation_vector(
            positions,
            min_val=float(min_val),
            max_val=float(max_val),
            rng=rng,
        )

    augmented = np.concatenate(
        [
            z.astype(conformer.dtype, copy=False),
            positions.astype(conformer.dtype, copy=False),
        ],
        axis=1,
    )
    return replace(frame, conformer=augmented)
