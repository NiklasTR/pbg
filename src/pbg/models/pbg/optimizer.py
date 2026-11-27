"""Muon / AdamW optimizer and learning-rate schedules for PBG training.

PBG uses Kimi-style Muon (update RMS 0.2) with decoupled, LR-squared weight
decay, AdaGC clipping and a warmup-stable-decay schedule (MiniCPM,
arXiv:2404.06395; Hägele et al., arXiv:2405.18392). The ROBIN protocol of the
ablation ladder uses fan-in scaled Muon, global clipping and a cosine schedule.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax.transforms._masking import MaskedNode

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerConfig:
    """Bundles every optimizer knob so callers don't pass 11 kwargs."""

    # Options that earlier configs carried and this release fixes are listed
    # in ``pbg.checkpoint.RETIRED_OPTIONS["optimizer"]``: Nesterov momentum,
    # 5th-order Newton–Schulz after momentum on the PyTorch ``(out, in)`` view
    # of 2-D weights, decoupled weight decay and clipping of the raw gradient.
    lr: float = 0.001
    muon_momentum: float = 0.95
    ns_iters: int = 5
    ns_dtype: str = "float32"
    grouping: str = "matrix_except_embed"  # "matrix_except_embed" | "by_ndim"
    fan_in_scale: bool = False
    weight_decay: float = 0.0
    # Bundle Kimi update-RMS matching with LR-squared effective decay. The
    # configured weight_decay remains the coefficient at the peak/stable LR.
    muon_scaling_patches: bool = False
    adamw_lr_ratio: float = 1.0
    # None uses weight_decay. Set this to keep AdamW embeddings at base λ
    # when Muon hidden decay follows a width rule.
    adam_weight_decay: float | None = None
    # Gradient clipping. ``grad_clip`` is the absolute threshold lambda_abs of
    # classic global clipping (GlobalGC). ``grad_clip_rel`` > 0 additionally
    # enables AdaGC (arXiv:2502.11034): after ``grad_clip_rel_warmup`` steps the
    # threshold becomes per-tensor and *relative*, ``lambda_rel * gamma``, where
    # gamma is an EMA of that tensor's own post-clip gradient norm. At
    # ``grad_clip_rel == 0`` the transform is ``optax.clip_by_global_norm``.
    grad_clip: float = 1.0  # lambda_abs; <=0 disables clipping entirely
    grad_clip_rel: float = 0.0  # lambda_rel; <=0 keeps plain GlobalGC
    grad_clip_rel_beta: float = 0.99  # EMA decay for the per-tensor threshold
    grad_clip_rel_warmup: int = 100  # T_start: GlobalGC steps before going relative
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8

    # LR schedule. ``cosine`` keeps the current OneCycleLR-style warmup + cosine
    # decay. ``wsd`` is split across trainer (warmup + stable) and cooler
    # (standalone cooldown).
    schedule: str = "cosine"  # "cosine" | "wsd"

    # Set total_steps > 0 to enable cosine decay; when 0, a constant LR is used.
    total_steps: int = 0
    warmup_steps: int = 0  # preferred; falls back to int(total_steps * warmup_pct)
    warmup_pct: float = 0.05  # fraction of total_steps for linear warmup
    div_factor: float = 500.0  # init_lr = lr / div_factor
    final_div_factor: float = 1.0  # end_lr = init_lr / final_div_factor
    cooldown_steps: int = 0  # WSD cooldown length; standalone when total_steps is unset
    final_lr_frac: float = 0.0  # final lr = lr * final_lr_frac in WSD cooldown
    cooldown_shape: str = "one_minus_sqrt"  # "one_minus_sqrt" | "linear"


# ---------------------------------------------------------------------------
# Newton–Schulz orthogonalization
# ---------------------------------------------------------------------------

# 5th-order coefficients (Bernstein et al., matches TS zeropower_via_newtonschulz5)
_NS5_A, _NS5_B, _NS5_C = 3.4445, -4.7750, 2.0315


def newton_schulz(G, iters: int = 5, dtype: str = "float32"):
    """Orthogonalize matrix *G* with the 5th-order Newton–Schulz iteration.

    Tall matrices (rows > cols) use the right-multiply form, wide and square
    matrices the left-multiply form. The output has unit scale.
    """
    if G.ndim < 2:
        return G

    original_shape = G.shape
    orig_dtype = G.dtype
    X = G.reshape(-1, G.shape[-1])
    rows, cols = X.shape

    work_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    X = X.astype(work_dtype)

    norm = jnp.linalg.norm(X, ord="fro") + 1e-7
    X = X / norm

    if rows > cols:
        for _ in range(iters):
            A = X.T @ X
            B = _NS5_B * A + _NS5_C * (A @ A)
            X = _NS5_A * X + X @ B
    else:
        for _ in range(iters):
            A = X @ X.T
            B = _NS5_B * A + _NS5_C * (A @ A)
            X = _NS5_A * X + B @ X

    X = X.astype(orig_dtype)
    return X.reshape(original_shape)


# ---------------------------------------------------------------------------
# Param label classification
# ---------------------------------------------------------------------------

def get_param_labels(params, grouping: str = "matrix_except_embed"):
    """Classify parameters into ``'muon'`` or ``'adam'``.

    Parameters
    ----------
    grouping : str
        ``"by_ndim"`` — all ≥2-D parameters get Muon (ROBIN protocol).
        ``"matrix_except_embed"`` — all ≥2-D parameters except embedding and
        output tables get Muon (Kimi/Moonlight convention, PBG).
    """

    valid_groupings = {"by_ndim", "matrix_except_embed"}
    if grouping not in valid_groupings:
        raise ValueError(
            f"unknown optimizer grouping {grouping!r}; expected one of "
            f"{sorted(valid_groupings)!r}"
        )

    def _label(path, leaf):
        if not hasattr(leaf, "ndim") or leaf.ndim < 2:
            return "adam"
        if grouping == "by_ndim":
            return "muon"
        p_str = "/".join(str(getattr(k, "key", k)) for k in path)
        return "adam" if "embed" in p_str or "lm_head" in p_str else "muon"

    return jax.tree_util.tree_map_with_path(_label, params)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _is_embed_path(path) -> bool:
    """Return *True* if *path* (a sequence of ``jax.tree_util`` keys) points to
    an embedding parameter that is stored in the same layout on both JAX and TS.
    """
    p_str = "/".join(str(getattr(k, "key", k)) for k in path)
    return "embed" in p_str


def _effective_rows_cols(shape: tuple[int, ...], do_transpose: bool) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"expected >=2-D shape, got {shape!r}")
    if len(shape) == 2 and do_transpose:
        return shape[1], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    return math.prod(shape[:-1]), shape[-1]


@dataclass(frozen=True)
class MatrixBatchPlan:
    """Static grouping of >=2-D pytree leaves for batched Muon kernels."""

    leaf_keys: tuple[tuple | None, ...]
    groups: dict[tuple, tuple[int, ...]]


def _matrix_layout_key(path, shape: tuple[int, ...]) -> tuple | None:
    if len(shape) < 2:
        return None
    # 2-D non-embedding weights are viewed as PyTorch ``(out, in)`` matrices.
    do_transpose = len(shape) == 2 and not _is_embed_path(path)
    rows, cols = _effective_rows_cols(shape, do_transpose)
    ns_side = "tall" if rows > cols else "wide"
    red_dim = -1 if rows >= cols else -2
    return (shape, do_transpose, ns_side, red_dim)


def _muon_branch_params(params, labels):
    return jax.tree.map(
        lambda lbl, p: p if lbl == "muon" else MaskedNode(),
        labels,
        params,
    )


def _build_matrix_batch_plan(params) -> MatrixBatchPlan:
    paths_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    leaf_keys: list[tuple | None] = []
    groups: dict[tuple, list[int]] = {}
    for idx, (path, leaf) in enumerate(paths_leaves):
        shape = tuple(getattr(leaf, "shape", ()))
        key = _matrix_layout_key(path, shape)
        leaf_keys.append(key)
        if key is not None:
            groups.setdefault(key, []).append(idx)
    return MatrixBatchPlan(
        leaf_keys=tuple(leaf_keys),
        groups={key: tuple(indices) for key, indices in groups.items()},
    )


def _apply_matrix_batch_plan(tree, plan: MatrixBatchPlan, group_fn):
    paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    leaves = [leaf for _, leaf in paths_leaves]
    out = list(leaves)
    for key, indices in plan.groups.items():
        stacked = jnp.stack([leaves[i] for i in indices], axis=0)
        batched = group_fn(key, stacked)
        for batch_idx, leaf_idx in enumerate(indices):
            out[leaf_idx] = batched[batch_idx]
    return jax.tree_util.tree_unflatten(treedef, out)


def _build_orthogonalization_transform(
    cfg: OptimizerConfig,
    batch_plan: MatrixBatchPlan,
):
    """Build an ``optax.stateless`` transform that orthogonalizes gradients.

    Uses Newton–Schulz iteration.

    2-D **non-embedding** arrays are transposed before orthogonalization and transposed back afterwards so that
    the tall/wide classification matches the TS (PyTorch) storage convention
    ``(out_features, in_features)``.  Embedding tables have the same layout in
    both frameworks and are left as-is.
    """
    def _ortho_fn(x, do_transpose: bool):
        if do_transpose:
            x = x.T
        result = newton_schulz(x, iters=cfg.ns_iters, dtype=cfg.ns_dtype)
        if do_transpose:
            result = result.T
        return result

    def _transform(g, _state):
        def group_fn(key, stacked):
            do_transpose = key[1]
            return jax.vmap(lambda x: _ortho_fn(x, do_transpose))(stacked)

        return _apply_matrix_batch_plan(g, batch_plan, group_fn)

    return optax.stateless(_transform)


def _build_fan_in_scale_transform(batch_plan: MatrixBatchPlan):
    """Scale each update by ``sqrt(max(1, fan_in / fan_out))`` (ROBIN protocol).

    The batch plan views 2-D non-embedding arrays as ``(out, in)``, as in the
    PyTorch layout.
    """

    def _scale(g, _state):
        def group_fn(key, stacked):
            shape, do_transpose, _, _ = key
            rows, cols = _effective_rows_cols(shape, do_transpose)
            return stacked * jnp.sqrt(jnp.maximum(1.0, rows / cols))

        return _apply_matrix_batch_plan(g, batch_plan, group_fn)

    return optax.stateless(_scale)


def _build_kimi_rms_scale_transform(batch_plan: MatrixBatchPlan):
    """Scale semi-orthogonal Muon updates to elementwise RMS approximately 0.2."""

    def _scale(g, _state):
        def group_fn(key, stacked):
            shape, do_transpose, _, _ = key
            rows, cols = _effective_rows_cols(shape, do_transpose)
            return stacked * (0.2 * math.sqrt(max(rows, cols)))

        return _apply_matrix_batch_plan(g, batch_plan, group_fn)

    return optax.stateless(_scale)




# ---------------------------------------------------------------------------
# Cautious weight decay
# ---------------------------------------------------------------------------


def _build_scheduled_decoupled_wd_transform(
    weight_decay: float,
    lr_or_schedule,
    peak_lr: float,
):
    """Schedule the WD coefficient linearly with LR for LR-squared shrinkage."""

    def init_fn(params):
        return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("Decoupled WD requires params")
        lr = lr_or_schedule(state.count) if callable(lr_or_schedule) else lr_or_schedule
        coefficient = weight_decay * lr / peak_lr

        new_updates = jax.tree_util.tree_map(
            lambda u, p: u + coefficient * p, updates, params
        )
        return new_updates, optax.ScaleByScheduleState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


class AdaGCState(NamedTuple):
    """State for :func:`build_grad_clip_transform` in AdaGC mode.

    ``gamma`` mirrors the parameter tree and holds one fp32 scalar per tensor:
    the adaptive threshold gamma_{t,i} of AdaGC. ``count`` is the step counter
    used to switch from the absolute warmup phase to the relative phase, and
    ``num_clipped`` counts how many tensors were clipped on the last step (for
    logging the clip rate).
    """

    gamma: optax.Params
    count: jnp.ndarray
    num_clipped: jnp.ndarray


# Guards ``lambda / ||g||`` against a zero-norm tensor. Well below any realistic
# gradient norm, so it never perturbs the clipping decision itself.
_GRAD_CLIP_EPS = 1e-12


def _tensor_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32))))


def build_grad_clip_transform(
    lambda_abs: float,
    *,
    lambda_rel: float = 0.0,
    beta: float = 0.99,
    warmup: int = 100,
) -> optax.GradientTransformation:
    """Global or adaptive gradient clipping.

    With ``lambda_rel <= 0`` this is exactly classic global clipping (GlobalGC):
    a single scale ``min(1, lambda_abs / ||g||)`` applied to the whole tree, with
    no optimizer state. That is the special case; the general case is AdaGC
    (arXiv:2502.11034), enabled by ``lambda_rel > 0``.

    In AdaGC mode each tensor gets its own threshold. For the first ``warmup``
    steps the transform behaves as GlobalGC while seeding ``gamma_i`` with the
    running *minimum* of the per-tensor norms, per Alg. 1 line 9 of the paper.
    Afterwards tensor ``i`` is scaled by ``min(1, lambda_rel * gamma_i / ||g_i||)``
    and ``gamma_i`` is updated as an EMA of the *post-clip* norm, so a spike is
    both suppressed and prevented from raising the threshold that suppressed it.

    Framing the two as one transform is deliberate: GlobalGC is AdaGC at global
    granularity with a frozen threshold, and AdaGC's own warmup *is* GlobalGC.
    Sharing the code path keeps ``lambda_rel = 0`` bit-identical to the previous
    behaviour and therefore checkpoint-compatible.
    """
    if lambda_abs <= 0:
        raise ValueError(f"grad_clip must be > 0, got {lambda_abs!r}")
    if lambda_rel > 0:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"grad_clip_rel_beta must be in (0, 1), got {beta!r}")
        if warmup < 0:
            raise ValueError(f"grad_clip_rel_warmup must be >= 0, got {warmup!r}")

    # Special case: no per-tensor state, no behaviour change, no new state to
    # break checkpoint resumption.
    if lambda_rel <= 0:
        return optax.clip_by_global_norm(lambda_abs)

    def _global_scale(updates) -> jnp.ndarray:
        norm = optax.global_norm(updates).astype(jnp.float32)
        return jnp.minimum(1.0, lambda_abs / (norm + _GRAD_CLIP_EPS))

    def init_fn(params):
        return AdaGCState(
            gamma=jax.tree_util.tree_map(
                lambda p: jnp.zeros((), dtype=jnp.float32), params
            ),
            count=jnp.zeros((), dtype=jnp.int32),
            num_clipped=jnp.zeros((), dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        del params
        warming_up = state.count < warmup

        # --- Warmup branch: GlobalGC, and seed gamma with a running minimum.
        warm_scale = _global_scale(updates)

        # --- Steady branch: per-tensor relative clipping.
        def _clip(u, gamma):
            norm = _tensor_norm(u)
            scale = jnp.where(
                warming_up,
                warm_scale,
                jnp.minimum(1.0, lambda_rel * gamma / (norm + _GRAD_CLIP_EPS)),
            )
            clipped = (u.astype(jnp.float32) * scale).astype(u.dtype)
            clipped_norm = norm * scale

            # A non-finite tensor is zeroed and leaves gamma untouched; letting
            # NaN into the EMA would permanently disable clipping for it.
            finite = jnp.isfinite(clipped_norm)
            clipped = jnp.where(finite, clipped, jnp.zeros_like(clipped))

            first = state.count == 0
            seeded = jnp.where(first, clipped_norm, jnp.minimum(gamma, clipped_norm))
            new_gamma = jnp.where(
                warming_up, seeded, beta * gamma + (1.0 - beta) * clipped_norm
            )
            new_gamma = jnp.where(finite, new_gamma, gamma)

            was_clipped = jnp.logical_or(
                jnp.logical_not(finite),
                jnp.logical_and(jnp.logical_not(warming_up), scale < 1.0),
            )
            return clipped, new_gamma, was_clipped.astype(jnp.int32)

        update_leaves, treedef = jax.tree_util.tree_flatten(updates)
        gamma_leaves = treedef.flatten_up_to(state.gamma)
        clipped = [_clip(u, g) for u, g in zip(update_leaves, gamma_leaves)]

        num_clipped = (
            sum(flag for _, _, flag in clipped)
            if clipped
            else jnp.zeros((), dtype=jnp.int32)
        )
        return jax.tree_util.tree_unflatten(
            treedef, [u for u, _, _ in clipped]
        ), AdaGCState(
            gamma=jax.tree_util.tree_unflatten(treedef, [g for _, g, _ in clipped]),
            count=state.count + 1,
            num_clipped=jnp.asarray(num_clipped, dtype=jnp.int32),
        )

    return optax.GradientTransformation(init_fn, update_fn)

def _append_negative_lr_scale(chain: list, lr_or_schedule) -> None:
    if callable(lr_or_schedule):
        schedule = lr_or_schedule
        chain.append(optax.scale_by_schedule(lambda step: -schedule(step)))
    else:
        chain.append(optax.scale(-lr_or_schedule))


def _adam_weight_decay(cfg: OptimizerConfig) -> float:
    if cfg.adam_weight_decay is None:
        return cfg.weight_decay
    return cfg.adam_weight_decay


def _validate_config(cfg: OptimizerConfig) -> None:
    """Reject optimizer settings that this release does not implement."""
    if cfg.schedule not in {"wsd", "cosine"}:
        raise ValueError(f"unsupported optimizer settings: schedule {cfg.schedule!r}")
    if cfg.adam_weight_decay is not None and cfg.adam_weight_decay < 0:
        raise ValueError(
            f"adam_weight_decay must be >= 0, got {cfg.adam_weight_decay!r}"
        )
    if cfg.grad_clip_rel > 0 and cfg.grad_clip <= 0:
        raise ValueError(
            "grad_clip_rel requires grad_clip > 0: AdaGC's warmup phase is "
            "GlobalGC with lambda_abs = grad_clip"
        )


def create_optimizer(params, *, config: OptimizerConfig):
    """Create the Muon / AdamW multi-transform optimizer.

    Muon (``"muon"`` leaves): Nesterov momentum, Newton–Schulz
    orthogonalization, then either Kimi update-RMS matching or fan-in scaling,
    decoupled weight decay and the negative learning rate. AdamW (``"adam"``
    leaves): Adam moments, decoupled weight decay and the negative learning
    rate. Global or adaptive (AdaGC) clipping runs on the raw gradient first.
    """
    cfg = config
    _validate_config(cfg)

    labels = get_param_labels(params, grouping=cfg.grouping)
    batch_plan = _build_matrix_batch_plan(_muon_branch_params(params, labels))

    # ---- LR schedule ----
    muon_lr, adam_lr_sched = _build_learning_rate_schedules(cfg)

    # ---- Muon branch: momentum + Nesterov first, then orthogonalization ----
    muon_chain: list = [
        optax.trace(decay=cfg.muon_momentum, nesterov=True),
        _build_orthogonalization_transform(cfg, batch_plan),
    ]
    if cfg.muon_scaling_patches:
        muon_chain.append(_build_kimi_rms_scale_transform(batch_plan))
    elif cfg.fan_in_scale:
        muon_chain.append(_build_fan_in_scale_transform(batch_plan))
    if cfg.weight_decay > 0:
        if cfg.muon_scaling_patches:
            muon_chain.append(
                _build_scheduled_decoupled_wd_transform(cfg.weight_decay, muon_lr, cfg.lr)
            )
        else:
            muon_chain.append(optax.add_decayed_weights(cfg.weight_decay))
    _append_negative_lr_scale(muon_chain, muon_lr)
    muon_opt = optax.chain(*muon_chain)

    # ---- Adam branch ----
    adam_wd = _adam_weight_decay(cfg)
    adam_chain: list = [
        optax.scale_by_adam(b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps)
    ]
    if adam_wd > 0:
        if cfg.muon_scaling_patches:
            adam_chain.append(
                _build_scheduled_decoupled_wd_transform(
                    adam_wd, adam_lr_sched, cfg.lr * cfg.adamw_lr_ratio
                )
            )
        else:
            adam_chain.append(optax.add_decayed_weights(adam_wd))
    _append_negative_lr_scale(adam_chain, adam_lr_sched)
    adam_opt = optax.chain(*adam_chain)

    # ---- Global / adaptive clip on the raw gradient ----
    # ``grad_clip_rel == 0`` yields plain ``clip_by_global_norm``; > 0 switches to
    # per-tensor AdaGC thresholds.
    inner = optax.multi_transform({"muon": muon_opt, "adam": adam_opt}, labels)
    if cfg.grad_clip > 0:
        optimizer = optax.chain(
            build_grad_clip_transform(
                cfg.grad_clip,
                lambda_rel=cfg.grad_clip_rel,
                beta=cfg.grad_clip_rel_beta,
                warmup=cfg.grad_clip_rel_warmup,
            ),
            inner,
        )
    else:
        optimizer = inner

    # Logging
    counts = {"muon": 0, "adam": 0}
    jax.tree_util.tree_map(
        lambda lbl: counts.update({lbl: counts.get(lbl, 0) + 1}), labels
    )
    log.info(
        "optimizer_assignment  muon=%d  adam=%d  grouping=%s  "
        "ns_iters=%d  batched_groups=%d  scaling_patches=%s  "
        "lr=%.4g  "
        "adam_lr_ratio=%.4g  schedule=%s  "
        "warmup_steps=%d  cooldown_steps=%d  final_lr_frac=%.4g  "
        "grad_clip=%.4g  grad_clip_rel=%.4g",
        counts["muon"],
        counts["adam"],
        cfg.grouping,
        cfg.ns_iters,
        len(batch_plan.groups),
        cfg.muon_scaling_patches,
        cfg.lr,
        cfg.adamw_lr_ratio,
        schedule_kind(cfg),
        resolve_warmup_steps(cfg),
        cfg.cooldown_steps,
        cfg.final_lr_frac,
        cfg.grad_clip,
        cfg.grad_clip_rel,
    )

    return optimizer, labels


def resolve_warmup_steps(cfg: OptimizerConfig) -> int:
    """Return the effective warmup length for *cfg*."""
    if cfg.warmup_steps > 0:
        return int(cfg.warmup_steps)
    if cfg.total_steps > 0:
        return int(cfg.total_steps * cfg.warmup_pct)
    return 0


def schedule_kind(cfg: OptimizerConfig) -> str:
    """Human-readable schedule name for logging/metrics."""
    if cfg.schedule == "wsd":
        return "wsd_cooldown" if cfg.cooldown_steps > 0 else "wsd_stable"
    if cfg.total_steps > 0:
        return "cosine"
    return "constant"


def _cooldown_progress(step, cooldown_steps: int):
    if cooldown_steps <= 1:
        return 1.0
    return jnp.clip(step / float(cooldown_steps - 1), 0.0, 1.0)


def _build_cooldown_schedule(
    *,
    start_value: float,
    end_value: float,
    cooldown_steps: int,
    shape: str,
):
    if shape == "linear":
        return lambda step: end_value + (start_value - end_value) * (
            1.0 - _cooldown_progress(step, cooldown_steps)
        )
    if shape == "one_minus_sqrt":
        return lambda step: end_value + (start_value - end_value) * (
            1.0 - jnp.sqrt(_cooldown_progress(step, cooldown_steps))
        )
    raise ValueError(f"unknown WSD cooldown_shape={shape!r}")


def _build_wsd_schedule(
    *,
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    cooldown_steps: int,
    end_value: float,
    cooldown_shape: str,
    total_steps: int = 0,
):
    if cooldown_steps > 0:
        if total_steps > cooldown_steps:
            cooldown_start = total_steps - cooldown_steps
            if warmup_steps >= cooldown_start:
                raise ValueError(
                    "WSD warmup_steps must be smaller than total_steps - cooldown_steps"
                )
            cooldown = _build_cooldown_schedule(
                start_value=peak_value,
                end_value=end_value,
                cooldown_steps=cooldown_steps,
                shape=cooldown_shape,
            )
            if warmup_steps > 0:
                return optax.join_schedules(
                    schedules=[
                        optax.linear_schedule(
                            init_value=init_value,
                            end_value=peak_value,
                            transition_steps=warmup_steps,
                        ),
                        optax.constant_schedule(peak_value),
                        cooldown,
                    ],
                    boundaries=[warmup_steps, cooldown_start],
                )
            return optax.join_schedules(
                schedules=[optax.constant_schedule(peak_value), cooldown],
                boundaries=[cooldown_start],
            )
        return _build_cooldown_schedule(
            start_value=peak_value,
            end_value=end_value,
            cooldown_steps=cooldown_steps,
            shape=cooldown_shape,
        )
    if warmup_steps > 0:
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=init_value,
                    end_value=peak_value,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(peak_value),
            ],
            boundaries=[warmup_steps],
        )
    return peak_value


def _build_learning_rate_schedules(cfg: OptimizerConfig):
    """Return ``(muon_lr, adam_lr)`` for the configured schedule family."""
    init_lr = cfg.lr / cfg.div_factor
    warmup_steps = resolve_warmup_steps(cfg)

    if cfg.schedule == "wsd":
        end_lr = cfg.lr * cfg.final_lr_frac
        muon_lr = _build_wsd_schedule(
            init_value=init_lr,
            peak_value=cfg.lr,
            warmup_steps=warmup_steps,
            cooldown_steps=int(cfg.cooldown_steps),
            end_value=end_lr,
            cooldown_shape=cfg.cooldown_shape,
            total_steps=int(cfg.total_steps),
        )
        adam_lr_sched = _build_wsd_schedule(
            init_value=init_lr * cfg.adamw_lr_ratio,
            peak_value=cfg.lr * cfg.adamw_lr_ratio,
            warmup_steps=warmup_steps,
            cooldown_steps=int(cfg.cooldown_steps),
            end_value=end_lr * cfg.adamw_lr_ratio,
            cooldown_shape=cfg.cooldown_shape,
            total_steps=int(cfg.total_steps),
        )
        return muon_lr, adam_lr_sched

    if cfg.total_steps > 0:
        end_lr = init_lr / cfg.final_div_factor
        muon_lr = optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=cfg.lr,
            warmup_steps=warmup_steps,
            decay_steps=cfg.total_steps,
            end_value=end_lr,
        )
        adam_lr_sched = optax.warmup_cosine_decay_schedule(
            init_value=init_lr * cfg.adamw_lr_ratio,
            peak_value=cfg.lr * cfg.adamw_lr_ratio,
            warmup_steps=warmup_steps,
            decay_steps=cfg.total_steps,
            end_value=end_lr * cfg.adamw_lr_ratio,
        )
        return muon_lr, adam_lr_sched

    return cfg.lr, cfg.lr * cfg.adamw_lr_ratio


def current_learning_rates(config: OptimizerConfig, step: int) -> tuple[float, float]:
    """Return the effective Muon and Adam learning rates at *step*."""
    muon_lr, adam_lr = _build_learning_rate_schedules(config)
    mu = float(muon_lr(step)) if callable(muon_lr) else float(muon_lr)
    ad = float(adam_lr(step)) if callable(adam_lr) else float(adam_lr)
    return mu, ad


