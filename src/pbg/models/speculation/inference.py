"""Base-model inference helpers for autoregressive and speculative decoding."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from pbg.models.pbg.model import PBG
from pbg.models.pbg.sample import inference_model
from pbg.models.pbg.transformer import (
    DEFAULT_ATTENTION_BUCKET_SIZE,
    apply_qwen,
    atom_rope,
)


def cast_base_inference(base: PBG, *, mixed_precision: bool | None = None) -> PBG:
    """Strip training config and optionally cast a native base model."""
    base = inference_model(base)
    use_mixed_precision = (
        base.config.mixed_precision if mixed_precision is None else mixed_precision
    )
    base = replace(
        base, config=replace(base.config, mixed_precision=use_mixed_precision)
    )
    if not use_mixed_precision:
        return base

    def cast(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(jnp.bfloat16)
        return value

    return PBG(jax.tree.map(cast, base.qwen), base.config, base.vocab)


def _inputs(base: PBG, atom_index, level_index, sequence_index):
    angles = atom_rope(base.config, atom_index[None], sequence_index[None])
    bias = base._level_bias(level_index[None])
    if base.config.mixed_precision:
        angles = angles.astype(jnp.bfloat16)
        bias = bias.astype(jnp.bfloat16)
    return angles, bias


def _batched_inputs(base: PBG, atom_index, level_index, sequence_index):
    angles = atom_rope(base.config, atom_index, sequence_index)
    bias = base._level_bias(level_index)
    if base.config.mixed_precision:
        angles = angles.astype(jnp.bfloat16)
        bias = bias.astype(jnp.bfloat16)
    return angles, bias


def make_base_forward(
    base: PBG,
    *,
    extension_len: int = 1,
    output_token_ids=None,
    max_cache_len: int | None = None,
    attention_bucket_size: int = DEFAULT_ATTENTION_BUCKET_SIZE,
    cache_dtype: str = "bfloat16",
):
    """Build the native frozen-base callback used by compiled AR."""
    cache_len = base.config.max_seq_len if max_cache_len is None else max_cache_len

    def forward(
        ids,
        proposal_mask,
        atom_index,
        level_index,
        sequence_index,
        caches,
        cache_index,
    ):
        del proposal_mask
        if caches is not None and ids.shape[0] > extension_len:
            raise ValueError(
                f"decoder extension {ids.shape[0]} exceeds compiled width {extension_len}"
            )
        angles, bias = _inputs(base, atom_index, level_index, sequence_index)
        logits, new_caches = apply_qwen(
            base.qwen,
            base.config,
            ids[None],
            kv_caches=caches,
            cache_index=cache_index,
            max_cache_len=cache_len,
            angles=angles,
            bias=bias,
            output_token_ids=output_token_ids,
            attention_bucket_size=attention_bucket_size,
            cache_dtype=cache_dtype,
        )
        return logits[0], tuple(new_caches)

    return forward


def make_batched_base_forward(
    base: PBG,
    *,
    extension_len: int = 1,
    output_token_ids=None,
    max_cache_len: int | None = None,
    attention_bucket_size: int = DEFAULT_ATTENTION_BUCKET_SIZE,
    cache_dtype: str = "bfloat16",
):
    """Build the native frozen-base callback for native batched R-ISD."""
    cache_len = base.config.max_seq_len if max_cache_len is None else max_cache_len

    def forward(
        ids,
        proposal_mask,
        atom_index,
        level_index,
        sequence_index,
        caches,
        cache_index,
    ):
        del proposal_mask
        if caches is not None and ids.shape[1] > extension_len:
            raise ValueError(
                f"decoder extension {ids.shape[1]} exceeds compiled width {extension_len}"
            )
        angles, bias = _batched_inputs(base, atom_index, level_index, sequence_index)
        logits, new_caches = apply_qwen(
            base.qwen,
            base.config,
            ids,
            kv_caches=caches,
            cache_index=cache_index,
            max_cache_len=cache_len,
            angles=angles,
            bias=bias,
            output_token_ids=output_token_ids,
            attention_bucket_size=attention_bucket_size,
            cache_dtype=cache_dtype,
        )
        return logits, tuple(new_caches)

    return forward
