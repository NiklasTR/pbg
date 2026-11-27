"""The Muon/AdamW optimizer: parameter groups, weight decay, gradient clipping
(GlobalGC and AdaGC), learning-rate schedules and batched Newton-Schulz."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pbg.models.pbg.model import PBG
from pbg.models.pbg.optimizer import (
    AdaGCState,
    OptimizerConfig,
    _build_kimi_rms_scale_transform,
    _build_matrix_batch_plan,
    _build_orthogonalization_transform,
    _muon_branch_params,
    build_grad_clip_transform,
    create_optimizer,
    current_learning_rates,
    get_param_labels,
    newton_schulz,
)

# ── Parameter groups ─────────────────────────────────────────────────


def test_matrix_except_embed_puts_transformer_matrices_in_muon_and_the_rest_in_adam() -> None:
    matrix = jnp.zeros((4, 4))
    params = {
        "embed": jnp.zeros((16, 4)),
        "lm_head": jnp.zeros((16, 4)),
        "blocks": {
            "0": {
                "attn": {"wq": matrix, "wk": matrix, "wv": matrix, "wo": matrix, "q_norm": jnp.zeros((4,))},
                "mlp": {"w_up": matrix, "w_gate": matrix, "w_down": matrix},
                "attn_norm": jnp.zeros((4,)),
            }
        },
        "final_norm": jnp.zeros((4,)),
    }
    config = OptimizerConfig(
        lr=0.001,
        grouping="matrix_except_embed",
        ns_iters=5,
        fan_in_scale=False,
        muon_scaling_patches=True,
        weight_decay=0.1,
        adamw_lr_ratio=1.0,
        adam_b2=0.95,
    )

    optimizer, labels = create_optimizer(params, config=config)
    optimizer.init(params)

    assert labels == get_param_labels(params, grouping="matrix_except_embed")
    block = labels["blocks"]["0"]
    assert all(block["attn"][name] == "muon" for name in ("wq", "wk", "wv", "wo"))
    assert all(block["mlp"][name] == "muon" for name in ("w_up", "w_gate", "w_down"))
    assert labels["embed"] == labels["lm_head"] == labels["final_norm"] == "adam"
    assert block["attn"]["q_norm"] == block["attn_norm"] == "adam"
    with pytest.raises(ValueError, match="unknown optimizer grouping"):
        get_param_labels(params, grouping="stale_grouping")


def test_atom_conditioner_is_grouped_like_the_trunk(tiny_config) -> None:
    model = PBG.init(tiny_config(conditioning="atom"), jax.random.PRNGKey(0))

    labels = get_param_labels(model.qwen, "matrix_except_embed")["atom_conditioner"]

    assert labels["atom_embed"] == labels["residue_embed"] == "adam"
    assert labels["mlp_in_bias"] == labels["attn_out_bias"] == "adam"
    for name in ("mlp_in", "mlp_out", "attn_q", "attn_kv", "attn_out"):
        assert labels[name] == "muon", name


# ── Weight decay ─────────────────────────────────────────────────────


def _one_zero_gradient_step(**overrides):
    """One step with zero gradients isolates the weight decay of each branch."""
    values = {
        "lr": 0.02,
        "adamw_lr_ratio": 0.1,
        "weight_decay": 0.01,
        "grouping": "by_ndim",
        "muon_momentum": 0.0,
        "ns_iters": 0,
        "grad_clip": 0.0,
        "schedule": "wsd",
    }
    values.update(overrides)
    params = {"matrix": jnp.ones((2, 2), dtype=jnp.float32), "vector": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    optimizer, _ = create_optimizer(params, config=OptimizerConfig(**values))
    updates, _ = optimizer.update(
        {key: jnp.zeros_like(value) for key, value in params.items()}, optimizer.init(params), params
    )
    return params, optax.apply_updates(params, updates)


@pytest.mark.parametrize(
    ("overrides", "adam_decay"),
    [({}, 0.01), ({"adam_weight_decay": 0.05}, 0.05), ({"muon_scaling_patches": True}, 0.01)],
    ids=["shared", "separate-adam-decay", "scaling-patches"],
)
def test_decoupled_decay_shrinks_each_branch_with_its_learning_rate(overrides, adam_decay) -> None:
    params, updated = _one_zero_gradient_step(**overrides)

    np.testing.assert_allclose(updated["matrix"], params["matrix"] * (1.0 - 0.02 * 0.01))
    np.testing.assert_allclose(updated["vector"], params["vector"] * (1.0 - 0.002 * adam_decay))


def test_decoupled_decay_uses_the_live_warmup_learning_rate() -> None:
    params, updated = _one_zero_gradient_step(warmup_steps=100, div_factor=500.0)

    muon_lr = 0.02 / 500.0
    np.testing.assert_allclose(updated["matrix"], params["matrix"] * (1.0 - muon_lr * 0.01))
    np.testing.assert_allclose(updated["vector"], params["vector"] * (1.0 - 0.1 * muon_lr * 0.01))


def test_scaling_patches_decay_with_the_squared_learning_rate_during_warmup() -> None:
    params, updated = _one_zero_gradient_step(muon_scaling_patches=True, warmup_steps=100, div_factor=2.0)

    muon_lr = 0.02 / 2.0
    adam_lr = muon_lr * 0.1
    np.testing.assert_allclose(updated["matrix"], params["matrix"] * (1.0 - 0.01 * muon_lr**2 / 0.02))
    np.testing.assert_allclose(updated["vector"], params["vector"] * (1.0 - 0.01 * adam_lr**2 / 0.002))


@pytest.mark.parametrize("shape", [(4, 2), (2, 4), (3, 3)])
def test_kimi_scaling_sets_the_update_rms_across_shapes(shape: tuple[int, int]) -> None:
    update = np.zeros(shape, dtype=np.float32)
    rank = min(shape)
    update[np.arange(rank), np.arange(rank)] = 1.0
    transform = _build_kimi_rms_scale_transform(_build_matrix_batch_plan(update))

    scaled, _ = transform.update(jnp.asarray(update), transform.init(update))

    np.testing.assert_allclose(np.sqrt(np.mean(np.asarray(scaled) ** 2)), 0.2, atol=1e-7)


# ── Gradient clipping: GlobalGC and AdaGC (arXiv:2502.11034) ─────────
#
# AdaGC replaces the absolute clipping threshold with a per-tensor threshold
# lambda_rel * gamma_i, where gamma_i is an EMA of the tensor's own post-clip
# gradient norm. Global clipping is the special case lambda_rel = 0.

LAMBDA_ABS = 1.0
LAMBDA_REL = 1.04  # paper default


def _tree(a: float, b: float) -> dict[str, jnp.ndarray]:
    return {"matrix": jnp.full((2, 2), a, dtype=jnp.float32), "vector": jnp.full((3,), b, dtype=jnp.float32)}


def _adagc_config(**overrides) -> OptimizerConfig:
    values = {
        "lr": 0.02,
        "grouping": "by_ndim",
        "muon_momentum": 0.0,
        "grad_clip": LAMBDA_ABS,
        "grad_clip_rel": LAMBDA_REL,
    }
    values.update(overrides)
    return OptimizerConfig(**values)


def test_zero_lambda_rel_is_exactly_clip_by_global_norm() -> None:
    """The identical transform and state keep checkpoints without AdaGC resumable."""
    ours = build_grad_clip_transform(LAMBDA_ABS, lambda_rel=0.0)
    theirs = optax.clip_by_global_norm(LAMBDA_ABS)
    params, grads = _tree(1.0, 1.0), _tree(5.0, 0.25)

    our_state, their_state = ours.init(params), theirs.init(params)
    assert not isinstance(our_state, AdaGCState)
    assert jax.tree_util.tree_structure(our_state) == jax.tree_util.tree_structure(their_state)

    our_out, _ = ours.update(grads, our_state, params)
    their_out, _ = theirs.update(grads, their_state, params)
    for key in grads:
        np.testing.assert_array_equal(our_out[key], their_out[key])


def test_warmup_clips_globally_like_globalgc() -> None:
    """During warmup AdaGC is GlobalGC (Alg. 1, lines 5-7)."""
    ada = build_grad_clip_transform(LAMBDA_ABS, lambda_rel=LAMBDA_REL, warmup=10)
    ref = optax.clip_by_global_norm(LAMBDA_ABS)
    params, grads = _tree(1.0, 1.0), _tree(5.0, 5.0)

    ada_out, ada_state = ada.update(grads, ada.init(params), params)
    ref_out, _ = ref.update(grads, ref.init(params), params)

    for key in grads:
        np.testing.assert_allclose(ada_out[key], ref_out[key], rtol=1e-6)
    assert int(ada_state.num_clipped) == 0  # warmup does not count as per-tensor clipping


def test_gamma_is_seeded_with_the_running_minimum_during_warmup() -> None:
    """Alg. 1 line 9: gamma_t = min(gamma_{t-1}, |g_t|), gamma_0 = |g_1|."""
    ada = build_grad_clip_transform(100.0, lambda_rel=LAMBDA_REL, warmup=5)
    params = _tree(1.0, 1.0)
    state = ada.init(params)
    sequence = [_tree(3.0, 3.0), _tree(1.0, 5.0), _tree(2.0, 4.0)]

    for grads in sequence:  # lambda_abs is large, so nothing is clipped
        _, state = ada.update(grads, state, params)

    for key in ("matrix", "vector"):
        expected = min(float(jnp.linalg.norm(grads[key])) for grads in sequence)
        assert float(state.gamma[key]) == pytest.approx(expected, rel=1e-6)


def test_steady_state_threshold_is_per_tensor_and_relative() -> None:
    """A spike in one tensor must not scale down its well-behaved neighbour."""
    ada = build_grad_clip_transform(100.0, lambda_rel=LAMBDA_REL, beta=0.99, warmup=1)
    params = calm = _tree(1.0, 1.0)
    _, state = ada.update(calm, ada.init(params), params)  # the warmup step seeds gamma
    gamma = float(state.gamma["matrix"])

    spiked = dict(calm, matrix=calm["matrix"] * 50.0)
    out, state = ada.update(spiked, state, params)

    assert float(jnp.linalg.norm(out["matrix"])) == pytest.approx(LAMBDA_REL * gamma, rel=1e-5)
    # The vector norm equals its gamma and lambda_rel > 1, so it passes untouched.
    np.testing.assert_allclose(out["vector"], spiked["vector"], rtol=1e-6)
    assert int(state.num_clipped) == 1


def test_gamma_ema_uses_the_post_clip_norm() -> None:
    """Alg. 1 line 15: a spike must not inflate the threshold that suppressed it."""
    beta = 0.9
    ada = build_grad_clip_transform(100.0, lambda_rel=LAMBDA_REL, beta=beta, warmup=1)
    params = calm = _tree(1.0, 1.0)
    _, state = ada.update(calm, ada.init(params), params)
    gamma0 = float(state.gamma["matrix"])

    _, state = ada.update(dict(calm, matrix=calm["matrix"] * 1000.0), state, params)

    expected = beta * gamma0 + (1.0 - beta) * LAMBDA_REL * gamma0
    assert float(state.gamma["matrix"]) == pytest.approx(expected, rel=1e-5)


def test_stationary_gradients_are_not_clipped_in_steady_state() -> None:
    transform = build_grad_clip_transform(100.0, lambda_rel=LAMBDA_REL, beta=0.99, warmup=3)
    update = jax.jit(transform.update)
    params = grads = _tree(1.0, 1.0)
    state = transform.init(params)

    for step in range(33):
        out, state = update(grads, state, params)
        if step >= 3:
            np.testing.assert_allclose(out["matrix"], grads["matrix"], rtol=1e-5)
            assert int(state.num_clipped) == 0


def test_non_finite_tensor_is_zeroed_and_leaves_gamma_intact() -> None:
    """A NaN must never enter the EMA, or clipping is disabled for that tensor."""
    ada = build_grad_clip_transform(100.0, lambda_rel=LAMBDA_REL, warmup=1)
    params = _tree(1.0, 1.0)
    _, state = ada.update(_tree(1.0, 1.0), ada.init(params), params)
    gamma_before = float(state.gamma["matrix"])

    bad = dict(_tree(1.0, 1.0), matrix=jnp.full((2, 2), jnp.nan, dtype=jnp.float32))
    out, state = ada.update(bad, state, params)

    np.testing.assert_array_equal(out["matrix"], jnp.zeros((2, 2)))
    assert float(state.gamma["matrix"]) == pytest.approx(gamma_before)
    assert int(state.num_clipped) == 1


@pytest.mark.parametrize(
    ("lambda_abs", "kwargs", "match"),
    [
        (0.0, {}, "grad_clip must be > 0"),
        *((LAMBDA_ABS, {"beta": beta}, "grad_clip_rel_beta") for beta in (0.0, 1.0, -0.1, 1.5)),
        (LAMBDA_ABS, {"warmup": -1}, "grad_clip_rel_warmup"),
    ],
)
def test_grad_clip_transform_rejects_invalid_settings(lambda_abs, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_grad_clip_transform(lambda_abs, lambda_rel=LAMBDA_REL, **kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"grad_clip": 0.0}, "requires grad_clip > 0"),
    ],
)
def test_adagc_rejects_incompatible_optimizer_settings(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        create_optimizer(_tree(1.0, 1.0), config=_adagc_config(**overrides))


def test_adagc_optimizer_suppresses_an_outlier_step_in_a_jitted_loop() -> None:
    params = _tree(1.0, 1.0)
    optimizer, _ = create_optimizer(params, config=_adagc_config(grad_clip_rel_warmup=2))

    @jax.jit
    def step(params, state, grads):
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), updates, state

    state = optimizer.init(params)
    calm = _tree(0.1, 0.1)
    for _ in range(4):
        params, normal, state = step(params, state, calm)
    _, outlier, _ = step(params, state, jax.tree.map(lambda g: g * 500.0, calm))

    assert all(bool(jnp.all(jnp.isfinite(value))) for value in params.values())
    # The 500x gradient must not produce a 500x update.
    assert float(optax.global_norm(outlier)) < 5.0 * float(optax.global_norm(normal))


# ── Learning-rate schedules ──────────────────────────────────────────


def test_wsd_warmup_rises_to_the_peak_and_stays_there() -> None:
    config = OptimizerConfig(schedule="wsd", lr=0.02, adamw_lr_ratio=0.1, warmup_steps=100, div_factor=500.0)

    rates = [current_learning_rates(config, step) for step in (0, 50, 100, 10_000)]

    np.testing.assert_allclose(rates[0], (0.02 / 500.0, 0.002 / 500.0), rtol=1e-5)
    assert rates[0][0] < rates[1][0] < rates[2][0]
    assert rates[0][1] < rates[1][1] < rates[2][1]
    np.testing.assert_allclose(rates[2], (0.02, 0.002), rtol=1e-5)
    np.testing.assert_allclose(rates[3], (0.02, 0.002), rtol=1e-5)


@pytest.mark.parametrize(("shape", "fraction"), [("sqrt", 1.0 - np.sqrt(50 / 99)), ("linear", 1.0 - 50 / 99)])
def test_wsd_cooldown_decays_to_the_final_rate(shape, fraction) -> None:
    kwargs = {} if shape == "sqrt" else {"cooldown_shape": shape}
    config = OptimizerConfig(
        schedule="wsd", lr=0.02, adamw_lr_ratio=0.1, cooldown_steps=100, final_lr_frac=0.0, **kwargs
    )

    np.testing.assert_allclose(current_learning_rates(config, 0), (0.02, 0.002), rtol=1e-5)
    np.testing.assert_allclose(current_learning_rates(config, 50), (0.02 * fraction, 0.002 * fraction), atol=1e-4)
    np.testing.assert_allclose(current_learning_rates(config, 99), (0.0, 0.0), atol=1e-8)


def test_robin_cosine_schedule_starts_and_ends_at_the_peak_over_500() -> None:
    config = OptimizerConfig(
        lr=5e-4,
        schedule="cosine",
        total_steps=1000,
        warmup_steps=10,
        div_factor=500.0,
        final_div_factor=1.0,
        adamw_lr_ratio=0.1,
    )

    np.testing.assert_allclose(current_learning_rates(config, 0), (1e-6, 1e-7), rtol=1e-4)
    np.testing.assert_allclose(current_learning_rates(config, 10), (5e-4, 5e-5), rtol=1e-4)
    np.testing.assert_allclose(current_learning_rates(config, 1000), (1e-6, 1e-7), rtol=1e-4)
    assert 2.4e-4 < current_learning_rates(config, 505)[0] < 2.6e-4


# ── Batched Newton-Schulz orthogonalisation ──────────────────────────


@pytest.fixture
def muon_params(tiny_config):
    """Muon-branch parameters of a two-layer PBG, in the paper's by_ndim grouping."""
    params = PBG.init(tiny_config(num_layers=2), jax.random.PRNGKey(0)).qwen
    return params, _muon_branch_params(params, get_param_labels(params, grouping="by_ndim"))


def test_matrix_batch_plan_stacks_every_muon_matrix_by_shape(muon_params) -> None:
    _, muon = muon_params
    leaves = jax.tree_util.tree_leaves(muon)

    plan = _build_matrix_batch_plan(muon)

    grouped = sorted(index for indices in plan.groups.values() for index in indices)
    assert grouped == [index for index, leaf in enumerate(leaves) if leaf.ndim >= 2]
    for indices in plan.groups.values():
        assert len({leaves[index].shape for index in indices}) == 1
    # Layers share their matrix shapes, so the plan batches across layers.
    assert len(plan.groups) < len(grouped)


def test_batched_orthogonalisation_matches_per_leaf_newton_schulz(muon_params) -> None:
    params, muon = muon_params
    config = OptimizerConfig(ns_iters=3, ns_dtype="bfloat16")
    grads = jax.tree_util.tree_map(jnp.ones_like, muon)
    transform = _build_orthogonalization_transform(config, _build_matrix_batch_plan(muon))

    batched, _ = transform.update(grads, None)

    def per_leaf(path, g):
        if g.ndim < 2:
            return g
        names = "/".join(str(getattr(key, "key", key)) for key in path)
        transpose = g.ndim == 2 and "embed" not in names
        y = newton_schulz(g.T if transpose else g, iters=config.ns_iters, dtype=config.ns_dtype)
        return y.T if transpose else y

    reference = jax.tree_util.tree_map_with_path(per_leaf, grads)
    for got, expected in zip(jax.tree_util.tree_leaves(batched), jax.tree_util.tree_leaves(reference), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-5)

    optimizer, _ = create_optimizer(
        params, config=replace(config, grouping="by_ndim", fan_in_scale=True, lr=0.02)
    )
    updates, _ = optimizer.update(jax.tree.map(lambda p: jnp.full_like(p, 0.01), params), optimizer.init(params), params)
    assert jax.tree_util.tree_structure(updates) == jax.tree_util.tree_structure(params)
