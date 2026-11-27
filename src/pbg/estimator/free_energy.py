"""Free-energy differences from importance weights: one-sided Zwanzig and two-sided BAR.

Both estimators take log weights ``log p_target(x) - log p_proposal(x)`` and return the
target-minus-proposal free energy in kJ/mol.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K

__all__ = ["BarEstimate", "ZwanzigEstimate", "bar", "zwanzig"]


@dataclass(frozen=True, slots=True)
class ZwanzigEstimate:
    delta_f_kj_mol: float
    std_kj_mol: float
    n: int
    ess: float
    max_weight_fraction: float

    @property
    def ress(self) -> float:
        return float(self.ess / self.n)


@dataclass(frozen=True, slots=True)
class BarEstimate:
    delta_f_kj_mol: float
    bar_std_kj_mol: float
    mbar_std_kj_mol: float
    bootstrap_std_kj_mol: float | None
    overlap: float
    bridge_ess_fraction: float

    @property
    def conservative_std_kj_mol(self) -> float | None:
        """The largest of the BAR, MBAR and bootstrap standard errors (infinite if any is)."""
        values = [self.bar_std_kj_mol, self.mbar_std_kj_mol, self.bootstrap_std_kj_mol]
        if any(x is not None and np.isposinf(x) for x in values):
            return float("inf")
        finite = [float(x) for x in values if x is not None and isfinite(float(x))]
        return max(finite) if finite else None


def zwanzig(
    log_weights: Any,
    *,
    temperature_K: float = 300.0,
    clip_percentile: float | None = 99.7,
) -> ZwanzigEstimate:
    """Estimate ``-RT log mean(exp(log_weights))`` from samples of the proposal.

    Non-finite log weights are dropped. Log weights above the ``clip_percentile`` percentile are
    capped at it before the estimate and its diagnostics; ``None`` disables the cap.
    ``std_kj_mol`` is the delta-method standard error.
    """
    rt = GAS_CONSTANT_KJ_MOL_K * float(temperature_K)
    if rt <= 0.0:
        raise ValueError(f"temperature_K must be positive, got {temperature_K}")
    raw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    clean = raw[np.isfinite(raw)]
    n = int(clean.size)
    if n == 0:
        raise ValueError("no finite log weights remain")
    if clip_percentile is not None:
        if not 0.0 <= float(clip_percentile) <= 100.0:
            raise ValueError(f"clip_percentile must be in [0, 100], got {clip_percentile}")
        clean = np.minimum(clean, float(np.percentile(clean, float(clip_percentile))))

    # Log-sum-exp with the maximum factored out; -700 keeps exp() above underflow.
    peak = float(np.max(clean))
    log_mean = float(peak + np.log(np.mean(np.exp(np.clip(clean - peak, -700.0, 0.0)))))
    log_sum = log_mean + np.log(float(n))
    peak2 = float(np.max(2.0 * clean))
    log_sum2 = float(peak2 + np.log(np.sum(np.exp(np.clip(2.0 * clean - peak2, -700.0, 0.0)))))
    ess = float(np.exp(np.clip(2.0 * log_sum - log_sum2, -700.0, 700.0))) if isfinite(log_sum) and isfinite(log_sum2) else 0.0

    std = float("nan")
    if n > 1:
        normalized = np.exp(np.clip(clean - log_mean, -700.0, 700.0))
        variance = float(np.sum((normalized - 1.0) ** 2) / (float(n) * float(n - 1)))
        std = float(rt * np.sqrt(max(variance, 0.0)))
    return ZwanzigEstimate(
        delta_f_kj_mol=float(-rt * log_mean),
        std_kj_mol=std,
        n=n,
        ess=ess,
        max_weight_fraction=float(np.exp(np.max(clean) - log_sum)),
    )


def bar(
    forward_log_weights: Any,
    reverse_log_weights: Any,
    *,
    temperature_K: float = 300.0,
    bootstrap: int = 0,
    seed: int | None = None,
    preserve_zero_weights: bool = False,
) -> BarEstimate:
    """Estimate the target-minus-proposal free energy with the Bennett acceptance ratio.

    ``forward_log_weights`` are ``log p_target(x) - log p_proposal(x)`` for samples of the proposal;
    ``reverse_log_weights`` are ``log p_proposal(x) - log p_target(x)`` for samples of the target.
    Non-finite log weights are dropped. With ``preserve_zero_weights=True``, negative-infinite log
    weights instead stay as zero-density samples that count in the sample sizes; NaN and positive
    infinity are then rejected. ``bootstrap`` resamples both sides that many times for
    ``bootstrap_std_kj_mol``.

    ``overlap`` is the smaller mean bridge acceptance of the two sides, and
    ``bridge_ess_fraction`` the harmonic mean of their Kish ESS fractions.
    """
    rt = GAS_CONSTANT_KJ_MOL_K * float(temperature_K)
    if rt <= 0.0:
        raise ValueError(f"temperature_K must be positive, got {temperature_K}")
    forward = np.asarray(forward_log_weights, dtype=np.float64).reshape(-1)
    reverse = np.asarray(reverse_log_weights, dtype=np.float64).reshape(-1)
    if preserve_zero_weights:
        if any(np.any(np.isnan(a) | np.isposinf(a)) for a in (forward, reverse)):
            raise ValueError("zero-weight BAR accepts finite values or negative infinity only")
    else:
        forward, reverse = forward[np.isfinite(forward)], reverse[np.isfinite(reverse)]
    if not np.any(np.isfinite(forward)) or not np.any(np.isfinite(reverse)):
        raise ValueError("BAR needs at least one finite forward and reverse log weight")

    # Reduced work in units of RT; the root delta of the BAR equation is the free energy over RT.
    forward_work, reverse_work = -forward, -reverse
    n_forward, n_reverse = float(forward_work.size), float(reverse_work.size)
    log_ratio = np.log(n_forward / n_reverse)

    def residual(delta: float) -> float:
        # Compare log sums: an absolute probability tolerance falsely accepts
        # arbitrary points when both bridge acceptances are near zero.
        lhs = np.logaddexp.reduce(-np.logaddexp(0.0, forward_work + log_ratio - delta))
        rhs = np.logaddexp.reduce(-np.logaddexp(0.0, reverse_work + delta - log_ratio))
        return float(lhs - rhs)

    # Bracket the root, widening by 100 RT at a time, then bisect.
    values = np.concatenate([forward_work + log_ratio, -reverse_work + log_ratio])
    values = values[np.isfinite(values)]
    lo = float(np.nanmin(values) - 100.0)
    hi = float(np.nanmax(values) + 100.0)
    rlo, rhi = residual(lo), residual(hi)
    for _ in range(20):
        if rlo <= 0.0 <= rhi:
            break
        lo -= 100.0
        hi += 100.0
        rlo, rhi = residual(lo), residual(hi)
    if not (rlo <= 0.0 <= rhi):
        raise ValueError(f"BAR root is not bracketed: residual({lo})={rlo}, residual({hi})={rhi}")
    delta = None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        rmid = residual(mid)
        if abs(rmid) < 1e-10:
            delta = float(mid)
            break
        if rmid < 0.0:
            lo = mid
        else:
            hi = mid
    if delta is None:
        delta = float(0.5 * (lo + hi))

    # Asymptotic BAR and MBAR variances (Shirts et al. 2003), in units of RT squared.
    var_bar = var_mbar = float("inf")
    if n_forward > 1.0 and n_reverse > 1.0:
        c = log_ratio - delta
        f_forward = _sigmoid_neg(forward_work + c)
        f_reverse = _sigmoid_neg(reverse_work - c)
        af_forward, af_reverse = float(np.mean(f_forward)), float(np.mean(f_reverse))
        af_forward2, af_reverse2 = float(np.mean(f_forward * f_forward)), float(np.mean(f_reverse * f_reverse))
        if min(af_forward, af_reverse) >= np.sqrt(np.finfo(float).tiny):
            nrat = (n_forward + n_reverse) / (n_forward * n_reverse)
            var_bar = (af_forward2 / (af_forward * af_forward)) / n_forward + (
                af_reverse2 / (af_reverse * af_reverse)
            ) / n_reverse - nrat
            denom_mbar = (af_forward - af_forward2) * n_forward + (af_reverse - af_reverse2) * n_reverse
            var_mbar = (1.0 / denom_mbar - nrat) if denom_mbar > 0.0 else float("inf")
    bar_std, mbar_std = (
        float("inf") if not np.isfinite(var) else float(np.sqrt(max(0.0, var))) for var in (var_bar, var_mbar)
    )

    # Bridge diagnostics: acceptance probabilities of each sample into the opposite state.
    forward_accept = _sigmoid_neg(forward_work + log_ratio - delta)
    reverse_accept = _sigmoid_neg(reverse_work + delta - log_ratio)
    fractions = []
    for accept, n in ((forward_accept, n_forward), (reverse_accept, n_reverse)):
        denom = float(np.sum(accept * accept))
        fractions.append((float(np.sum(accept) ** 2 / denom) if denom > 0.0 else 0.0) / n)
    bridge_ess_fraction = 0.0 if min(fractions) <= 0.0 else float(2.0 / (1.0 / fractions[0] + 1.0 / fractions[1]))

    bootstrap_std = None
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(int(bootstrap)):
            resampled_forward = forward[rng.integers(0, forward.size, size=forward.size)]
            resampled_reverse = reverse[rng.integers(0, reverse.size, size=reverse.size)]
            try:
                draws.append(bar(resampled_forward, resampled_reverse, temperature_K=temperature_K,
                                 preserve_zero_weights=preserve_zero_weights).delta_f_kj_mol)
            except ValueError:
                continue
        if draws:
            bootstrap_std = float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0
    return BarEstimate(
        delta_f_kj_mol=float(delta * rt),
        bar_std_kj_mol=float(bar_std * rt),
        mbar_std_kj_mol=float(mbar_std * rt),
        bootstrap_std_kj_mol=bootstrap_std,
        overlap=float(min(np.mean(forward_accept), np.mean(reverse_accept))),
        bridge_ess_fraction=bridge_ess_fraction,
    )


def _sigmoid_neg(x: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(x)), stable for large |x|."""
    return np.exp(-np.logaddexp(0.0, x))
