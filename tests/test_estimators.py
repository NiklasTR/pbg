"""Estimators: rESS, W2 on the line, exact and sliced torus W2, tICA coordinates, BAR and Zwanzig,
including equivalence checks against pymbar and POT."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.core.ensemble import TicaParameters, Trajectory
from pbg.estimator import (
    bar,
    importance_weights,
    relative_ess,
    sliced_torus_w2,
    tica_projection,
    w2,
    w2_1d,
    zwanzig,
)
from pbg.estimator.wasserstein import circular_transport_cost

ETHANE = "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]"
TWO_PI = 2.0 * np.pi
# Order-1 slice pool without sampling: in 1D it is the single slice k = 1.
ORDER1 = dict(max_order=1, max_entry=1, max_nnz=1, n_random=0)


def _normalised(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / float(np.sum(values))


def _ethane_frames(n_frames: int, offset: float = 0.0) -> np.ndarray:
    """Zxyz frames whose C-C distance is 1, 2, ... plus offset."""
    frames = []
    for index in range(1, n_frames + 1):
        x = index + offset
        frames.append(
            [[6, 0, 0, 0], [6, x, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, -1, 0, 0], [1, x, 1, 0], [1, x, 0, 1], [1, x + 1, 0, 0]]
        )
    return np.asarray(frames, dtype=np.float64)


# ── Importance weights and rESS ──────────────────────────────────────


def test_ress_counts_every_generated_proposal_and_can_drop_the_top_weights() -> None:
    log_weights = np.zeros(1000, dtype=np.float64)
    log_weights[-2:] = 10.0

    kept = importance_weights(log_weights)
    dropped = importance_weights(log_weights, drop_top_percent=0.2)

    expected = _normalised(np.exp(log_weights - np.max(log_weights)))
    np.testing.assert_allclose(kept, expected)
    assert relative_ess(kept, n_proposals=2000) == pytest.approx(1.0 / np.sum(expected * expected) / 2000.0)
    assert np.count_nonzero(dropped) == 998
    assert relative_ess(dropped, n_proposals=2000) == pytest.approx(998.0 / 2000.0)


def test_non_finite_log_weights_get_weight_zero() -> None:
    weights = importance_weights([0.0, np.nan, -np.inf, 0.0])

    np.testing.assert_allclose(weights, [0.5, 0.0, 0.0, 0.5])


# ── W2 on the real line ──────────────────────────────────────────────


def test_w2_1d_of_a_translation_is_the_shift() -> None:
    y = np.array([0.0, 1.0, 2.0])
    x = y + 5.0
    weights = [0.2, 0.3, 0.5]

    assert w2_1d(x, y) == pytest.approx(5.0)
    assert w2_1d(x, y, x_weights=weights, y_weights=weights) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("x", "y", "x_weights", "expected"),
    [
        ([0.0, 10.0], [0.0, 0.0], [0.9, 0.1], np.sqrt(10.0)),  # every sample is used once
        ([0.0, 10.0], [0.0, 0.0, 0.0], None, np.sqrt(50.0)),  # unequal uniform supports
    ],
)
def test_w2_1d_is_exact_for_small_supports(x, y, x_weights, expected) -> None:
    assert w2_1d(x, y, x_weights=x_weights) == pytest.approx(expected)


def test_w2_1d_does_not_invent_extreme_tail_mass() -> None:
    x = np.array([0.0, 1.0e15])
    y = np.array([0.0])
    weights = np.array([1.0, 1.0e-300])
    expected = np.sqrt(weights[1] * x[1] ** 2)

    assert w2_1d(x, y, x_weights=weights) == pytest.approx(expected)
    assert w2_1d(x[::-1], y, x_weights=weights[::-1]) == pytest.approx(expected)


@pytest.mark.equivalence
def test_w2_1d_matches_pot_emd2_1d() -> None:
    ot = pytest.importorskip("ot")
    x = np.array([-1.0, 0.5, 3.0, 4.0])
    y = np.array([-2.0, 1.0, 2.0, 5.5])
    x_weights = _normalised([0.1, 0.2, 0.4, 0.3])
    y_weights = _normalised([0.25, 0.15, 0.35, 0.25])

    assert w2_1d(x, y) == pytest.approx(np.sqrt(ot.emd2_1d(x, y, metric="sqeuclidean")), abs=1e-12)
    assert w2_1d(x, y, x_weights=x_weights, y_weights=y_weights) == pytest.approx(
        np.sqrt(ot.emd2_1d(x, y, x_weights, y_weights, metric="sqeuclidean")), abs=1e-12
    )


# ── Exact W2 in R^d and on the torus ─────────────────────────────────


def test_torus_w2_wraps_across_the_periodic_boundary() -> None:
    left = np.array([[np.pi - 1e-3, -(np.pi - 2e-3)]])
    right = np.array([[-np.pi + 1e-3, np.pi - 2e-3]])

    assert w2(left, right, period=TWO_PI) == pytest.approx(np.sqrt((2e-3) ** 2 + (4e-3) ** 2), abs=1e-12)


def test_w2_drops_non_finite_samples() -> None:
    assert w2([[0.0], [np.nan], [1.0]], [[0.0], [1.0]]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.equivalence
def test_torus_w2_matches_pot_with_the_wrapped_cost() -> None:
    ot = pytest.importorskip("ot")
    x = np.array([[np.pi - 1e-3, -np.pi + 2e-3], [-2.4, 0.25], [0.0, 2.9]])
    y = np.array([[-np.pi + 1.5e-3, np.pi - 1e-3], [2.1, -0.4], [0.2, -3.0], [-0.8, 1.7]])
    x_weights = _normalised([0.2, 0.5, 0.3])
    y_weights = _normalised([0.1, 0.2, 0.3, 0.4])
    raw = np.abs(x[:, None, :] - y[None, :, :])
    wrapped = np.minimum(raw, TWO_PI - raw)

    expected = np.sqrt(ot.emd2(x_weights, y_weights, np.sum(wrapped**2, axis=-1), numItermax=int(1e7)))

    assert w2(x, y, x_weights=x_weights, y_weights=y_weights, period=TWO_PI) == pytest.approx(expected, abs=1e-11)


# ── tICA coordinates ─────────────────────────────────────────────────


def test_tica_projection_of_heavy_pair_features() -> None:
    reference = Trajectory(
        samples=_ethane_frames(4),
        smiles=ETHANE,
        id="ref",
        tica=TicaParameters(
            mean=np.array([2.5]), projection=np.array([[1.0, 0.5]]), dim=np.array(2), featurization="heavy_pairs"
        ),
    )

    tics = tica_projection(_ethane_frames(4, offset=0.75), reference)

    # The one heavy-atom pair is the C-C distance 1.75, 2.75, ...
    distance = np.arange(1, 5) + 0.75
    np.testing.assert_allclose(tics, np.stack([distance - 2.5, 0.5 * (distance - 2.5)], axis=1))
    with pytest.raises(ValueError, match="dim=3"):
        tica_projection(reference.samples, reference, dim=3)


@pytest.mark.equivalence
def test_euclidean_w2_matches_pot_on_the_tics() -> None:
    ot = pytest.importorskip("ot")
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(6, 2)), rng.normal(size=(7, 2))
    y_weights = _normalised(rng.random(7))

    expected = np.sqrt(ot.emd2(np.full(6, 1.0 / 6.0), y_weights, cdist(x, y, metric="sqeuclidean")))

    assert w2(x, y, y_weights=y_weights) == pytest.approx(expected, abs=1e-12)


# ── Circular and sliced torus W2 ─────────────────────────────────────


@pytest.mark.equivalence
@pytest.mark.parametrize(("n", "m"), [(4, 4), (5, 7), (8, 8), (9, 3)])
def test_circular_solver_matches_the_transport_linear_program(n, m) -> None:
    ot = pytest.importorskip("ot")
    rng = np.random.default_rng(n * 10 + m)
    x, y = rng.uniform(0.0, TWO_PI, size=n), rng.uniform(0.0, TWO_PI, size=m)
    # Equal sizes take the cyclic-shift solver, unequal sizes the CDF-breakpoint solver.
    wx = _normalised(rng.random(n) if n != m else np.ones(n))
    wy = _normalised(rng.random(m) if n != m else np.ones(m))
    raw = np.abs(x[:, None] - y[None, :])
    wrapped = np.minimum(raw, TWO_PI - raw)

    expected = ot.emd2(wx, wy, wrapped**2, numItermax=int(1e7))

    assert circular_transport_cost(x, wx, y, wy, period=TWO_PI) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_sliced_torus_w2_is_exact_in_one_dimension() -> None:
    rng = np.random.default_rng(1)
    for n, m, period in ((5, 5, TWO_PI), (6, 9, 1.7), (8, 4, 2.0)):
        a, b = rng.uniform(0.0, period, size=(n, 1)), rng.uniform(0.0, period, size=(m, 1))
        weights = dict(x_weights=_normalised(rng.random(n)), y_weights=_normalised(rng.random(m)))

        sliced = sliced_torus_w2(a, b, period=period, **weights, **ORDER1)

        assert sliced == pytest.approx(w2(a, b, period=period, **weights), rel=1e-10, abs=1e-10)


def test_sliced_marginals_lower_bound_the_exact_distance() -> None:
    rng = np.random.default_rng(2)
    a = rng.uniform(-np.pi, np.pi, size=(8, 3))
    b = rng.uniform(-np.pi, np.pi, size=(7, 3))

    assert sliced_torus_w2(a, b, **ORDER1) <= w2(a, b, period=TWO_PI) + 1e-12


def test_coupled_slices_see_a_gap_that_the_marginals_miss() -> None:
    phi = np.linspace(0.0, TWO_PI, 24, endpoint=False)
    nu = np.stack([phi, phi + 0.7], axis=1)
    unique = np.unique(np.mod(nu, TWO_PI), axis=0)
    axes = np.meshgrid(np.unique(unique[:, 0]), np.unique(unique[:, 1]), indexing="ij")
    mu = np.stack([axis.reshape(-1) for axis in axes], axis=1)

    assert sliced_torus_w2(mu, nu, **ORDER1) == pytest.approx(0.0, abs=1e-8)
    assert w2(mu, nu, period=TWO_PI) > 0.0
    assert sliced_torus_w2(mu, nu, max_order=2, max_nnz=2, n_random=0) > 0.0


def test_sliced_torus_w2_is_a_rotation_invariant_symmetric_distance() -> None:
    rng = np.random.default_rng(4)
    a = rng.uniform(-np.pi, np.pi, size=(9, 2))
    b = rng.uniform(-np.pi, np.pi, size=(8, 2))
    shift = np.array([[0.35, -0.8]])

    assert sliced_torus_w2(a, a, n_random=0) == pytest.approx(0.0, abs=1e-12)
    assert sliced_torus_w2(a, b, n_random=0) == pytest.approx(sliced_torus_w2(b, a, n_random=0), rel=1e-10, abs=1e-10)
    assert sliced_torus_w2(a, a + shift, n_random=0) == pytest.approx(
        sliced_torus_w2(a + 0.2, a + shift + 0.2, n_random=0), rel=1e-12, abs=1e-12
    )


def test_sampled_slices_are_reproducible_with_a_seeded_generator() -> None:
    rng = np.random.default_rng(6)
    a = rng.uniform(-np.pi, np.pi, size=(30, 4))
    b = rng.uniform(-np.pi, np.pi, size=(25, 4))
    settings = dict(max_order=2, max_entry=2, max_nnz=2, n_random=8)

    first = sliced_torus_w2(a, b, **settings, rng=np.random.default_rng(0))

    assert first == sliced_torus_w2(a, b, **settings, rng=np.random.default_rng(0))
    assert first > 0.0


def test_wrap_handling_does_not_inflate_the_branch_cut() -> None:
    left = np.array([[0.01], [6.27]])
    right = np.array([[0.02], [6.28]])

    value = sliced_torus_w2(left, right, **ORDER1)

    assert value < 0.05
    assert value == pytest.approx(w2(left, right, period=TWO_PI), rel=1e-10, abs=1e-10)


def test_symmetry_quotient_period() -> None:
    period = TWO_PI / 3.0
    base = np.array([[0.1], [0.4], [0.9]])

    assert sliced_torus_w2(base, base + period, period=period, **ORDER1) == pytest.approx(0.0, abs=1e-12)


# ── BAR and Zwanzig ──────────────────────────────────────────────────


def test_overlap_and_bridge_ess_follow_the_opposite_state_probabilities() -> None:
    forward_log_weights = np.array([-0.2, 0.4, 1.1, -0.7])
    reverse_log_weights = np.array([0.3, -0.5, 0.8])

    estimate = bar(forward_log_weights, reverse_log_weights)

    log_ratio = np.log(forward_log_weights.size / reverse_log_weights.size)
    delta = estimate.delta_f_kj_mol / (GAS_CONSTANT_KJ_MOL_K * 300.0)
    forward = 1.0 / np.exp(np.logaddexp(0.0, -forward_log_weights + log_ratio - delta))
    reverse = 1.0 / np.exp(np.logaddexp(0.0, -reverse_log_weights + delta - log_ratio))
    np.testing.assert_allclose(estimate.overlap, min(np.mean(forward), np.mean(reverse)))
    forward_fraction = np.sum(forward) ** 2 / np.sum(forward**2) / forward.size
    reverse_fraction = np.sum(reverse) ** 2 / np.sum(reverse**2) / reverse.size
    np.testing.assert_allclose(estimate.bridge_ess_fraction, 2.0 / (1.0 / forward_fraction + 1.0 / reverse_fraction))


@pytest.mark.parametrize(("n_forward", "n_reverse", "overlap"), [(10, 10, 0.5), (1, 9, 0.1)])
def test_overlap_of_identical_states_is_the_smaller_sample_fraction(n_forward, n_reverse, overlap) -> None:
    np.testing.assert_allclose(bar(np.zeros(n_forward), np.zeros(n_reverse)).overlap, overlap)


def test_bar_estimate_of_a_symmetric_work_distribution() -> None:
    forward_work = np.array([0.8, 1.0, 1.2])

    estimate = bar(-forward_work, forward_work)

    np.testing.assert_allclose(estimate.delta_f_kj_mol, GAS_CONSTANT_KJ_MOL_K * 300.0)
    np.testing.assert_allclose(estimate.overlap, 0.5)


def test_underflowing_bridge_reports_unbounded_uncertainty_without_crashing() -> None:
    estimate = bar(np.full(100, -1000.0), np.full(100, -1000.0))

    np.testing.assert_allclose(estimate.delta_f_kj_mol, 0.0, atol=1e-7)
    assert estimate.overlap == 0.0
    assert np.isinf(estimate.conservative_std_kj_mol)


def test_small_bridge_root_is_not_stopped_by_an_absolute_probability_tolerance() -> None:
    # Unequal counts make the bracket midpoint differ from the BAR root. In the
    # small-acceptance limit, equal sums give
    # delta = (500 - 700 + log(n_forward / n_reverse)) / 2.
    estimate = bar(np.full(100, -500.0), np.full(200, -700.0))

    expected = (-100.0 + 0.5 * np.log(0.5)) * GAS_CONSTANT_KJ_MOL_K * 300.0
    np.testing.assert_allclose(estimate.delta_f_kj_mol, expected, atol=1e-6)
    assert estimate.overlap < 1e-200


def test_zero_weight_samples_count_in_the_bar_sample_sizes() -> None:
    forward = np.array([-0.3, 0.2, 0.5, -0.1])
    reverse = np.array([0.4, -0.2, 0.1])

    finite_only = bar(np.append(forward, -np.inf), reverse)
    with_zeros = bar(np.append(forward, -np.inf), reverse, preserve_zero_weights=True)

    assert finite_only.delta_f_kj_mol == bar(forward, reverse).delta_f_kj_mol
    # A zero-density proposal sample lowers the forward acceptance, so the target looks less likely.
    assert with_zeros.delta_f_kj_mol > finite_only.delta_f_kj_mol
    with pytest.raises(ValueError, match="negative infinity"):
        bar(np.append(forward, np.nan), reverse, preserve_zero_weights=True)


def test_zwanzig_matches_the_clipped_exponential_average() -> None:
    log_weights = np.random.default_rng(17).normal(loc=-0.2, scale=1.4, size=500)
    rt = GAS_CONSTANT_KJ_MOL_K * 300.0

    estimate = zwanzig(log_weights, temperature_K=300.0)
    unclipped = zwanzig(log_weights, temperature_K=300.0, clip_percentile=None)

    clipped = np.minimum(log_weights, np.percentile(log_weights, 99.7))
    weights = np.exp(clipped - np.max(clipped))
    peak = float(np.max(clipped))
    assert estimate.delta_f_kj_mol == pytest.approx(-rt * (peak + np.log(np.mean(np.exp(clipped - peak)))), abs=1e-12)
    assert estimate.n == 500
    assert estimate.ess == pytest.approx(np.sum(weights) ** 2 / np.sum(weights**2))
    assert estimate.max_weight_fraction == pytest.approx(np.max(weights) / np.sum(weights))
    assert 0.0 < estimate.std_kj_mol < np.inf
    assert unclipped.delta_f_kj_mol < estimate.delta_f_kj_mol


@pytest.mark.equivalence
def test_bar_matches_pymbar() -> None:
    pymbar = pytest.importorskip("pymbar")
    rng = np.random.default_rng(23)
    forward_work = rng.normal(loc=0.7 + 0.8, scale=1.0, size=700)
    reverse_work = rng.normal(loc=-0.7 + 0.8, scale=1.0, size=650)
    rt = GAS_CONSTANT_KJ_MOL_K * 300.0

    estimate = bar(-forward_work, -reverse_work, temperature_K=300.0, bootstrap=20, seed=11)

    expected = pymbar.bar(forward_work, reverse_work, compute_uncertainty=False)
    bar_uncertainty = pymbar.bar(forward_work, reverse_work, uncertainty_method="BAR")
    mbar_uncertainty = pymbar.bar(forward_work, reverse_work, uncertainty_method="MBAR")
    assert estimate.delta_f_kj_mol == pytest.approx(float(expected["Delta_f"] * rt), abs=1e-9)
    assert estimate.bar_std_kj_mol == pytest.approx(float(bar_uncertainty["dDelta_f"] * rt), abs=1e-10)
    assert estimate.mbar_std_kj_mol == pytest.approx(float(mbar_uncertainty["dDelta_f"] * rt), abs=1e-10)
    assert 0.0 < estimate.bootstrap_std_kj_mol < 10.0 * estimate.bar_std_kj_mol
