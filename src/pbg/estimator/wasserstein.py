"""Wasserstein-2 distances between weighted empirical distributions.

``w2_1d`` is exact on the real line. ``w2`` solves the exact transport problem in R^d, or on a
flat torus when it gets a period. ``sliced_torus_w2`` averages exact circular W2 costs over integer
lattice directions of a torus. Weights default to uniform. Samples with a non-finite coordinate or
a zero weight are dropped, and the remaining weights are normalised to sum one.
"""

from __future__ import annotations

from itertools import combinations, islice, product
from typing import Any

import numpy as np

__all__ = ["circular_transport_cost", "sliced_torus_w2", "w2", "w2_1d"]

# Exact circular solver budget n * m * (n + m); larger weighted supports compare 1024 quantiles.
_MAX_EXACT_CIRCULAR_WORK = 5_000_000
_QUANTILE_GRID = 1024


def _finite(points: Any, weights: Any | None) -> tuple[np.ndarray, np.ndarray]:
    """Rows (samples) with finite coordinates and positive weight, and their normalised weights."""
    x = np.asarray(points, dtype=np.float64)
    x = x.reshape(x.shape[0], -1)
    w = np.ones(x.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (x.shape[0],):
        raise ValueError(f"weights must have shape {(x.shape[0],)}, got {w.shape}")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    keep = np.all(np.isfinite(x), axis=1) & np.isfinite(w) & (w > 0.0)
    if not keep.any():
        raise ValueError("no finite positive-weight samples remain")
    return x[keep], w[keep] / float(np.sum(w[keep]))


def w2_1d(x: Any, y: Any, *, x_weights: Any | None = None, y_weights: Any | None = None) -> float:
    """Exact W2 between two weighted samples of real numbers.

    W2 squared integrates the squared difference of the two quantile functions. The integral runs
    from each end to the median, so that an extreme value of tiny weight contributes only its own
    mass.
    """
    sides = []
    for points, weights in ((x, x_weights), (y, y_weights)):
        values, w = _finite(points, weights)
        if values.shape[1] != 1:
            raise ValueError(f"w2_1d needs one value per sample, got shape {np.shape(points)}")
        order = np.argsort(values[:, 0])
        sides.append((values[order, 0], w[order]))
    (xs, xw), (ys, yw) = sides

    squared = 0.0
    for direction in (slice(None), slice(None, None, -1)):
        halves = []
        for values, weights in ((xs[direction], xw[direction]), (ys[direction], yw[direction])):
            cdf = np.cumsum(weights)
            stop = min(int(np.searchsorted(cdf, 0.5, side="left")), values.size - 1)
            cdf = cdf[: stop + 1].copy()
            cdf[-1] = 0.5
            halves.append((values[: stop + 1], cdf))
        (left, left_cdf), (right, right_cdf) = halves
        grid = np.unique(np.concatenate(([0.0], left_cdf, right_cdf)))
        widths = np.diff(grid)
        quantiles = grid[:-1] + 0.5 * widths
        left_indices = np.clip(np.searchsorted(left_cdf, quantiles, side="left"), 0, left.size - 1)
        right_indices = np.clip(np.searchsorted(right_cdf, quantiles, side="left"), 0, right.size - 1)
        difference = left[left_indices] - right[right_indices]
        squared += float(np.sum(widths * difference * difference))
    return float(np.sqrt(squared))


def w2(
    x: Any,
    y: Any,
    *,
    x_weights: Any | None = None,
    y_weights: Any | None = None,
    period: float | None = None,
) -> float:
    """Exact W2 between two weighted point clouds of shape (samples, dimensions).

    The cost is the squared Euclidean distance. With ``period``, every coordinate is an angle and
    its difference wraps the short way round (flat torus). POT's network simplex solves the
    transport problem.
    """
    import ot

    x, wx = _finite(x, x_weights)
    y, wy = _finite(y, y_weights)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dimension mismatch: {x.shape[1]} vs {y.shape[1]}")
    if period is not None and not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    # Each cost keeps the rounding of the published metric: TICA-W2 squared the Euclidean norm, and
    # T-W2 summed the squared wrapped angle differences in blocks of 64 rows (bounded memory for
    # 10^4 x 10^4 problems).
    if period is None:
        cost = np.square(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1))
    else:
        cost = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
        for start in range(0, x.shape[0], 64):
            block = np.zeros((min(64, x.shape[0] - start), y.shape[0]), dtype=np.float64)
            for column in range(x.shape[1]):
                delta = np.abs(x[start:start + 64, None, column] - y[None, :, column])
                np.mod(delta, period, out=delta)
                np.minimum(delta, period - delta, out=delta)
                np.square(delta, out=delta)
                block += delta
            cost[start:start + 64] = block
    return float(np.sqrt(float(ot.emd2(wx, wy, cost, numItermax=int(1e9), numThreads=1))))


def circular_transport_cost(x: Any, x_weights: Any, y: Any, y_weights: Any, *, period: float) -> float:
    """Exact W2 squared between two weighted samples on a circle of circumference ``period``.

    On a circle, an optimal plan couples the two quantile functions monotonically after a shift of
    one CDF. Equal-size uniform samples try every cyclic shift. Weighted samples try every CDF
    breakpoint as the shift, or, above the work budget, compare 1024 quantiles of each sample as
    equal-size uniform samples.
    """
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    sides = []
    for values, weights in ((x, x_weights), (y, y_weights)):
        wrapped = np.mod(np.asarray(values, dtype=np.float64).reshape(-1), period)
        order = np.argsort(wrapped, kind="mergesort")
        sides.append((wrapped[order], np.asarray(weights, dtype=np.float64).reshape(-1)[order]))
    (xs, wxs), (ys, wys) = sides
    if xs.size == 0 or ys.size == 0:
        raise ValueError("circular OT requires at least one sample in each measure")

    def wrap(delta: np.ndarray) -> np.ndarray:
        return (delta + 0.5 * period) % period - 0.5 * period

    n, m = int(xs.size), int(ys.size)
    uniform = n == m and all(np.allclose(w, 1.0 / n, rtol=0.0, atol=1e-12) for w in (wxs, wys))
    if not uniform:
        cdf_x = np.concatenate(([0.0], np.cumsum(wxs)))
        cdf_y = np.concatenate(([0.0], np.cumsum(wys)))
        cdf_x[-1] = cdf_y[-1] = 1.0
        if n * m * (n + m) <= _MAX_EXACT_CIRCULAR_WORK:
            best = np.inf
            for shift in np.unique(np.mod(cdf_y[:, None] - cdf_x[None, :], 1.0)):
                shift = float(shift)
                edges = np.unique(np.concatenate(([0.0, 1.0], cdf_x[1:-1], np.mod(cdf_y - shift, 1.0))))
                mass = np.diff(edges)
                keep = mass > 0.0
                mass = mass[keep]
                mid = edges[:-1][keep] + 0.5 * mass
                x_val = xs[np.searchsorted(cdf_x[1:], mid, side="left")]
                y_val = ys[np.searchsorted(cdf_y[1:], np.mod(mid + shift, 1.0), side="left")]
                best = min(best, float(np.dot(mass, np.abs(wrap(x_val - y_val)) ** 2)))
            return best
        grid = (np.arange(_QUANTILE_GRID, dtype=np.float64) + 0.5) / float(_QUANTILE_GRID)
        xs = xs[np.searchsorted(cdf_x[1:], grid, side="left")]
        ys = ys[np.searchsorted(cdf_y[1:], grid, side="left")]
        n = _QUANTILE_GRID

    best = np.inf
    ys_ext = np.concatenate((ys, ys))
    chunk = 256 if n > 256 else n
    for start in range(0, n, chunk):
        windows = np.stack([ys_ext[shift : shift + n] for shift in range(start, min(n, start + chunk))])
        cost = np.mean(np.abs(wrap(xs[None, :] - windows)) ** 2, axis=1)
        best = min(best, float(np.min(cost)))
    return float(best)


def sliced_torus_w2(
    x: Any,
    y: Any,
    *,
    x_weights: Any | None = None,
    y_weights: Any | None = None,
    period: float = 2.0 * np.pi,
    max_order: int = 3,
    max_entry: int = 2,
    max_nnz: int = 3,
    max_slices: int | None = None,
    n_random: int = 0,
    rng: Any = None,
) -> float:
    """Sliced W2 between two weighted samples of angles on a flat torus.

    A slice is a primitive integer direction k, with |k_i| <= ``max_entry``, at most ``max_nnz``
    nonzero entries and sum |k_i| <= ``max_order``. It maps an angle vector t to the circle
    coordinate k.t / period (mod 1), where the exact circular W2 is cheap. The pool holds the first
    ``max_slices`` directions in order of sum |k_i|, with weights proportional to
    (1 + |k|^2)^-(d/2 + 2). With 0 < ``n_random`` < pool size, the axis directions stay and
    ``n_random`` draws from the other directions, in proportion to their weights, replace them
    (importance sampling with ``rng``).
    """
    x, wx = _finite(x, x_weights)
    y, wy = _finite(y, y_weights)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dimension mismatch: {x.shape[1]} vs {y.shape[1]}")
    dim = int(x.shape[1])
    if dim < 1 or max_order < 1 or max_entry < 1 or max_nnz < 1:
        raise ValueError("dim, max_order, max_entry and max_nnz must be at least 1")
    if not (np.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")

    candidates = (
        (support, entries)
        for size in range(1, min(max_nnz, dim) + 1)
        for support in combinations(range(dim), size)
        for entries in product(range(-max_entry, max_entry + 1), repeat=size)
        if 0 not in entries
        and sum(abs(value) for value in entries) <= max_order
        and entries[0] > 0
        and int(np.gcd.reduce(np.abs(np.asarray(entries, dtype=np.int64)))) == 1
    )
    rows = list(islice(candidates, max_slices))
    if not rows:
        raise ValueError("slice pool is empty")
    lattice = np.zeros((len(rows), dim), dtype=np.int64)
    for row, (support, entries) in enumerate(rows):
        lattice[row, list(support)] = entries
    order = np.abs(lattice).sum(axis=1)
    permutation = np.argsort(order, kind="stable")
    lattice, order = lattice[permutation], order[permutation]
    slice_weights = (1.0 + np.sum(lattice.astype(np.float64) ** 2, axis=1)) ** (-((dim + 2) / 2.0 + 1.0))
    slice_weights = slice_weights / float(np.sum(slice_weights))
    # Each slice maps the torus onto a circle of circumference 1 / scale.
    scale = np.linalg.norm(lattice.astype(np.float64) / period, axis=1)

    index, importance = np.arange(order.size), slice_weights
    rest = np.flatnonzero(order > 1)
    if 0 < n_random < order.size and rest.size:
        head = np.flatnonzero(order <= 1)
        rest_mass = float(np.sum(slice_weights[rest]))
        generator = np.random.default_rng() if rng is None else rng
        drawn = generator.choice(rest, size=n_random, replace=True, p=slice_weights[rest] / rest_mass)
        index = np.concatenate([head, drawn])
        importance = np.concatenate([slice_weights[head], np.full(n_random, rest_mass / n_random, dtype=np.float64)])

    unit_x, unit_y = np.mod(x / period, 1.0), np.mod(y / period, 1.0)
    slice_cost = np.empty(index.size, dtype=np.float64)
    for start in range(0, index.size, 32):
        batch = index[start:start + 32]
        theta_x = (np.mod(unit_x @ lattice[batch].T, 1.0) / scale[batch][None, :]).T
        theta_y = (np.mod(unit_y @ lattice[batch].T, 1.0) / scale[batch][None, :]).T
        for local, slice_id in enumerate(range(start, start + batch.size)):
            slice_cost[slice_id] = circular_transport_cost(
                theta_x[local], wx, theta_y[local], wy, period=float(1.0 / scale[batch][local])
            )
    return float(np.sqrt(float(np.dot(importance, slice_cost))))
