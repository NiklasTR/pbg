"""Self-normalised importance weights and the relative effective sample size (rESS)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["importance_weights", "relative_ess"]


def importance_weights(log_weights: Any, *, drop_top_percent: float = 0.0) -> np.ndarray:
    """Self-normalised importance weights, one per log weight.

    A non-finite log weight gets weight zero. With ``drop_top_percent`` above zero, the log weights
    above the ``100 - drop_top_percent`` percentile of the finite ones also get weight zero
    (Table 1 drops the top 0.2%).
    """
    raw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    keep = np.isfinite(raw)
    if not keep.any():
        raise ValueError("no finite log weights remain")
    if not 0.0 <= drop_top_percent <= 100.0:
        raise ValueError(f"drop_top_percent must be in [0, 100], got {drop_top_percent}")
    if drop_top_percent > 0.0:
        keep &= raw <= float(np.percentile(raw[keep], 100.0 - drop_top_percent))
    kept = raw[keep]
    active = np.exp(kept - float(np.max(kept)))
    active /= float(np.sum(active))
    weights = np.zeros(raw.shape, dtype=np.float64)
    weights[keep] = active
    return weights


def relative_ess(weights: Any, *, n_proposals: int) -> float:
    """Kish effective sample size per generated proposal, (sum w)^2 / (n_proposals * sum w^2).

    ``n_proposals`` counts every generated proposal, including the ones with weight zero.
    """
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("no finite positive weights remain")
    if n_proposals <= 0:
        raise ValueError(f"n_proposals must be positive, got {n_proposals}")
    probabilities = values / float(np.sum(values))
    return float(1.0 / np.sum(probabilities * probabilities) / n_proposals)
