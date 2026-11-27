"""Linear coordinate binning: one token per Cartesian axis."""

from __future__ import annotations

import numpy as np


def geometry_width(*, num_bins: int, coordinate_method: str) -> int:
    """Return the per-atom coordinate-token width: one bin index per axis."""
    if coordinate_method != "bin":
        raise ValueError(f"Unknown coordinate_method {coordinate_method!r}; use 'bin'")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2 for linear-scale binning")
    return 3


def apply_discretise(x: np.ndarray, *, num_bins: int, min_val: float, max_val: float) -> np.ndarray:
    """Return the int32 index of the nearest of ``num_bins`` grid points from ``min_val`` to ``max_val``.

    Values outside the range are clipped to its ends. The output has the shape of ``x``.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2 for linear-scale binning")
    x_clipped = np.clip(np.asarray(x, dtype=np.float32), min_val, max_val)
    x_norm = (x_clipped - min_val) / (max_val - min_val)
    indices = np.rint(x_norm * (num_bins - 1)).astype(np.int32)
    return np.clip(indices, 0, num_bins - 1)


def undo_discretise(
    indices: np.ndarray,
    *,
    num_bins: int,
    min_val: float,
    max_val: float,
    method: str = "center",
    key: object | None = None,
) -> np.ndarray:
    """Return float32 coordinates for bin indices.

    ``method="center"`` returns the grid points. ``method="uniform"`` adds
    uniform noise over one grid spacing, seeded from the uint32 words of the
    JAX-style ``key`` (fresh entropy when ``key`` is None).
    """
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2 for linear-scale binning")
    indices = np.clip(np.asarray(indices, dtype=np.int32), 0, num_bins - 1)
    x_norm = indices.astype(np.float32) / (num_bins - 1)
    centers = x_norm * (max_val - min_val) + min_val
    if method == "center":
        return centers.astype(np.float32)
    if method != "uniform":
        raise ValueError(f"Unknown method {method!r}; use 'center' or 'uniform'")
    seed = None
    if key is not None:
        words = np.asarray(key, dtype=np.uint32).reshape(-1)
        if words.size == 0:
            raise ValueError("key must contain at least one uint32 value")
        seed = 0
        for word in words:
            seed = ((seed << 32) | int(word)) % 2**64
    noise = np.random.default_rng(seed).uniform(size=indices.shape).astype(np.float32) - 0.5
    return (centers + noise * ((max_val - min_val) / (num_bins - 1))).astype(np.float32)
