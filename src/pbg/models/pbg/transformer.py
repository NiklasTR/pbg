from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from pbg.models.pbg.config import PBGConfig

KVCache = tuple[Array, Array, Array, Array]
QwenParams = dict[str, Any]
DEFAULT_ATTENTION_BUCKET_SIZE = 64
_BFLOAT16_DTYPE = jax.dtypes.bfloat16


def _rms_norm(scale: Array, x: Array, eps: float) -> Array:
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps) * scale


def atom_rope(config: PBGConfig, atom_index: Array, sequence_index: Array) -> Array:
    """Return PBG RoPE angles for atom and sequence positions.

    The first ``atom_rope_frac`` of the frequency pairs rotate with the atom
    index and the remaining pairs with the logical sequence position.
    """
    num_pairs = config.head_dim // 2
    n_atom = round(num_pairs * config.atom_rope_frac)
    bands = []
    for pairs, positions in ((n_atom, atom_index), (num_pairs - n_atom, sequence_index)):
        idx = jnp.arange(0, 2 * pairs, 2, dtype=jnp.float32) / max(2 * pairs, 1)
        inv_freqs = 1.0 / (config.rope_theta**idx)
        bands.append(positions.astype(jnp.float32)[..., None] * inv_freqs)
    return jnp.concatenate(bands, axis=-1)


def apply_rope_halved(x: Array, freqs: Array) -> Array:
    """Apply rotary embedding using the halved convention (HF Llama/Qwen)."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    if freqs.ndim == 2:
        freqs = freqs[:, None, :]
    elif freqs.ndim == 3:
        freqs = freqs[:, :, None, :]
    cos = jnp.cos(freqs).astype(x.dtype)
    sin = jnp.sin(freqs).astype(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        axis=-1,
    )


def _init_mlp(key: Array, cfg: PBGConfig) -> QwenParams:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w_gate": jax.random.normal(k1, (cfg.dim, cfg.intermediate_size))
        / jnp.sqrt(cfg.dim),
        "w_up": jax.random.normal(k2, (cfg.dim, cfg.intermediate_size))
        / jnp.sqrt(cfg.dim),
        "w_down": jax.random.normal(k3, (cfg.intermediate_size, cfg.dim))
        / jnp.sqrt(cfg.intermediate_size),
    }


def _init_attention(key: Array, cfg: PBGConfig) -> QwenParams:
    kq, kk, kv, ko = jax.random.split(key, 4)
    std = jnp.sqrt(cfg.dim)
    return {
        "wq": jax.random.normal(kq, (cfg.dim, cfg.num_heads * cfg.head_dim)) / std,
        "wk": jax.random.normal(kk, (cfg.dim, cfg.num_kv_heads * cfg.head_dim)) / std,
        "wv": jax.random.normal(kv, (cfg.dim, cfg.num_kv_heads * cfg.head_dim)) / std,
        "wo": jax.random.normal(ko, (cfg.num_heads * cfg.head_dim, cfg.dim)) / std,
        "q_norm": jnp.ones((cfg.head_dim,)),
        "k_norm": jnp.ones((cfg.head_dim,)),
    }


def init_qwen(
    key: Array, config: PBGConfig, vocab_size: int, std: float = 0.02
) -> QwenParams:
    _, k_embed, k_blocks = jax.random.split(key, 3)
    blocks = []
    for _ in range(config.num_layers):
        k_blocks, k_block = jax.random.split(k_blocks)
        k_attn, k_mlp = jax.random.split(k_block)
        blocks.append(
            {
                "attn_norm": jnp.ones((config.dim,)),
                "attn": _init_attention(k_attn, config),
                "mlp_norm": jnp.ones((config.dim,)),
                "mlp": _init_mlp(k_mlp, config),
            }
        )
    return {
        "embed": jax.random.normal(k_embed, (vocab_size, config.dim)) * std,
        "blocks": tuple(blocks),
        "final_norm": jnp.ones((config.dim,)),
    }


def qwen_params(raw_params: dict[str, Any], config: PBGConfig) -> QwenParams:
    """Return the stored Qwen parameters as a JAX pytree."""
    params = {
        "embed": raw_params["embed"],
        "blocks": raw_params["blocks"],
        "final_norm": raw_params["final_norm"],
    }
    # The optional ROBIN atom conditioner is an adapter on the trunk.
    for name in ("lm_head", "atom_conditioner"):
        if name in raw_params:
            params[name] = raw_params[name]
    return jax.tree.map(jnp.asarray, params)


def qwen_vocab_size(params: QwenParams) -> int:
    return int(params["embed"].shape[0])


def prefix_bucket_lengths(cache_len: int, bucket: int) -> tuple[int, ...]:
    """Geometric ladder of static attention prefix lengths.

    Incremental attention must pass a statically shaped key/value slice to the
    attention kernel, so a runtime cache length can only be honoured by
    dispatching over a fixed set of candidate lengths. A geometric ladder needs
    ``O(log2(cache_len / bucket))`` branches where a linear ladder needs
    ``O(cache_len / bucket)``; on a 1028-entry cache with ``bucket=64`` that is
    5 branches instead of 17. Fewer branches means less compile time and less
    risk that the dispatch is materialised as unconditional work.
    """
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    if bucket <= 0:
        raise ValueError(f"bucket must be positive, got {bucket}")
    lengths: list[int] = []
    length = min(bucket, cache_len)
    while length < cache_len:
        # Drop a rung that is within 1.5x of the full cache: dispatching to it
        # would save almost no traffic, so it is pure branch overhead. This is
        # what keeps a 1028-entry cache from carrying both 1024 and 1028.
        if 2 * cache_len >= 3 * length:
            lengths.append(length)
        length *= 2
    lengths.append(cache_len)
    return tuple(lengths)


def _attention(
    params: QwenParams,
    x: Array,
    cfg: PBGConfig,
    *,
    angles: Array | None,
    mask: Array | None,
    kv_cache: KVCache | None,
    cache_index: int | Array | None,
    attention_bucket_size: int,
    cache_dtype: str = "bfloat16",
) -> Array | tuple[Array, KVCache]:
    if (kv_cache is None) != (cache_index is None):
        raise ValueError("kv_cache and cache_index must both be set, or both omitted")
    batch, seq_len, _ = x.shape
    q = (x @ params["wq"]).reshape(batch, seq_len, cfg.num_heads, cfg.head_dim)
    k = (x @ params["wk"]).reshape(batch, seq_len, cfg.num_kv_heads, cfg.head_dim)
    v = (x @ params["wv"]).reshape(batch, seq_len, cfg.num_kv_heads, cfg.head_dim)
    q = _rms_norm(params["q_norm"], q, cfg.rms_norm_eps)
    k = _rms_norm(params["k_norm"], k, cfg.rms_norm_eps)
    if angles is not None:
        q, k = apply_rope_halved(q, angles), apply_rope_halved(k, angles)
    if kv_cache is not None:
        cached_k, cached_v, cached_k_scale, cached_v_scale = kv_cache
        if jnp.ndim(cache_index) == 0:
            indices = jnp.arange(seq_len) + cache_index
            if cache_dtype == "float8_e4m3fn":
                k_scale = jnp.maximum(jnp.max(jnp.abs(k), axis=(0, 1, 3), keepdims=True) / 448.0, 1e-6)
                v_scale = jnp.maximum(jnp.max(jnp.abs(v), axis=(0, 1, 3), keepdims=True) / 448.0, 1e-6)
                cached_k = cached_k.at[:, indices].set((k / k_scale).astype(jnp.float8_e4m3fn))
                cached_v = cached_v.at[:, indices].set((v / v_scale).astype(jnp.float8_e4m3fn))
                cached_k_scale = cached_k_scale.at[:, indices].set(
                    jnp.broadcast_to(k_scale, (batch, seq_len, cfg.num_kv_heads, 1))
                )
                cached_v_scale = cached_v_scale.at[:, indices].set(
                    jnp.broadcast_to(v_scale, (batch, seq_len, cfg.num_kv_heads, 1))
                )
            else:
                cached_k = cached_k.at[:, indices].set(k)
                cached_v = cached_v.at[:, indices].set(v)
            kv_len = cache_index + seq_len
            q_pos = jnp.arange(seq_len) + cache_index
        else:
            batch_ids = jnp.arange(batch)[:, None]
            indices = cache_index[:, None] + jnp.arange(seq_len)[None]
            if cache_dtype == "float8_e4m3fn":
                k_scale = jnp.maximum(jnp.max(jnp.abs(k), axis=(1, 3), keepdims=True) / 448.0, 1e-6)
                v_scale = jnp.maximum(jnp.max(jnp.abs(v), axis=(1, 3), keepdims=True) / 448.0, 1e-6)
                cached_k = cached_k.at[batch_ids, indices].set((k / k_scale).astype(jnp.float8_e4m3fn))
                cached_v = cached_v.at[batch_ids, indices].set((v / v_scale).astype(jnp.float8_e4m3fn))
                cached_k_scale = cached_k_scale.at[batch_ids, indices].set(
                    jnp.broadcast_to(k_scale, (batch, seq_len, cfg.num_kv_heads, 1))
                )
                cached_v_scale = cached_v_scale.at[batch_ids, indices].set(
                    jnp.broadcast_to(v_scale, (batch, seq_len, cfg.num_kv_heads, 1))
                )
            else:
                cached_k = cached_k.at[batch_ids, indices].set(k)
                cached_v = cached_v.at[batch_ids, indices].set(v)
            kv_len = cache_index[:, None] + seq_len
            q_pos = cache_index[:, None] + jnp.arange(seq_len)[None]
        k_pos = jnp.arange(cached_k.shape[1])
        if jnp.ndim(cache_index) == 0:
            attn_mask = (
                (q_pos[:, None] >= k_pos[None, :]) & (k_pos[None, :] < kv_len)
            )[None, None]
        else:
            attn_mask = (
                (q_pos[:, :, None] >= k_pos[None, None, :])
                & (k_pos[None, None, :] < kv_len[:, :, None])
            )[:, None]
        if mask is not None:
            attn_mask &= mask
        bucket_lengths = prefix_bucket_lengths(cached_k.shape[1], attention_bucket_size)
        # Every lane must see its own committed prefix, so the batched path
        # dispatches on the longest logical length in the batch. That is a
        # conservative upper bound: shorter lanes read a few extra masked
        # entries but never miss a valid one.
        longest = jnp.max(kv_len)
        bucket_index = jnp.sum(
            jnp.asarray(bucket_lengths[:-1], dtype=jnp.int32) < longest
        )

        def attend(length: int) -> Array:
            # Explicit FP32 diagnostic/scoring contract. Preserve the historical
            # BF16 behavior for both existing cache modes.
            attention_dtype = jnp.float32 if cache_dtype == "float32" else _BFLOAT16_DTYPE
            attention_k = jax.lax.convert_element_type(
                cached_k[:, :length], attention_dtype
            )
            attention_v = jax.lax.convert_element_type(
                cached_v[:, :length], attention_dtype
            )
            if cache_dtype == "float8_e4m3fn":
                attention_k = attention_k * jax.lax.convert_element_type(
                    cached_k_scale[:, :length], _BFLOAT16_DTYPE
                )
                attention_v = attention_v * jax.lax.convert_element_type(
                    cached_v_scale[:, :length], _BFLOAT16_DTYPE
                )
            return jax.nn.dot_product_attention(
                q,
                attention_k,
                attention_v,
                mask=attn_mask[..., :length],
                scale=cfg.head_dim**-0.5,
            )

        out = jax.lax.switch(
            bucket_index,
            tuple(lambda length=length: attend(length) for length in bucket_lengths),
        )
        out = out.reshape(batch, seq_len, cfg.num_heads * cfg.head_dim) @ params["wo"]
        return out, (cached_k, cached_v, cached_k_scale, cached_v_scale)
    else:
        attn_mask = mask
    out = jax.nn.dot_product_attention(
        q,
        k,
        v,
        mask=attn_mask,
        is_causal=kv_cache is None,
        scale=cfg.head_dim**-0.5,
    )
    out = out.reshape(batch, seq_len, cfg.num_heads * cfg.head_dim) @ params["wo"]
    if kv_cache is not None:
        return out, (k, v, jnp.ones_like(k), jnp.ones_like(v))
    return out


def apply_qwen(
    params: QwenParams,
    config: PBGConfig,
    ids: Array,
    *,
    mask: Array | None = None,
    kv_caches: list[KVCache] | None = None,
    cache_index: int | Array | None = None,
    max_cache_len: int | None = None,
    angles: Array | None = None,
    bias: Array | None = None,
    output_token_ids: Array | None = None,
    attention_bucket_size: int = DEFAULT_ATTENTION_BUCKET_SIZE,
    cache_dtype: str = "bfloat16",
    remat_blocks: bool = False,
) -> Array | tuple[Array, list[KVCache]]:
    if cache_dtype not in {"bfloat16", "float8_e4m3fn", "float32"}:
        raise ValueError(f"unsupported cache dtype: {cache_dtype}")
    ids = jnp.asarray(ids, dtype=jnp.int32)
    batch, _ = ids.shape
    h = params["embed"][ids]
    if bias is not None:
        h += bias.astype(h.dtype)
    incremental = cache_index is not None
    if incremental and kv_caches is None:
        # Default to the trained sequence length, not max_position. The
        # incremental path streams the whole allocated cache each step, so a
        # max_position-sized default (40960) would silently cost ~40x the
        # memory traffic of the real context.
        length = max_cache_len or config.max_seq_len
        shape = (batch, length, config.num_kv_heads, config.head_dim)
        storage_dtype = jnp.float8_e4m3fn if cache_dtype == "float8_e4m3fn" else h.dtype
        kv_caches = [
             (jnp.zeros(shape, storage_dtype), jnp.zeros(shape, storage_dtype),
              jnp.ones(shape, _BFLOAT16_DTYPE), jnp.ones(shape, _BFLOAT16_DTYPE))
            for _ in params["blocks"]
        ]
    new_caches = []
    for index, block in enumerate(params["blocks"]):
        if remat_blocks and not incremental:
            def block_forward(weights, hidden, rope, attention_mask):
                attn_x = _rms_norm(weights["attn_norm"], hidden, config.rms_norm_eps)
                attn_out = _attention(weights["attn"], attn_x, config, angles=rope,
                    mask=attention_mask, kv_cache=None, cache_index=None,
                    attention_bucket_size=attention_bucket_size, cache_dtype=cache_dtype)
                scale = jnp.asarray(config.residual_scale, dtype=attn_out.dtype)
                hidden = hidden + scale * attn_out
                mlp_x = _rms_norm(weights["mlp_norm"], hidden, config.rms_norm_eps)
                mlp = weights["mlp"]
                return hidden + scale * ((jax.nn.silu(mlp_x @ mlp["w_gate"]) *
                                           (mlp_x @ mlp["w_up"])) @ mlp["w_down"])
            h = jax.checkpoint(block_forward)(block, h, angles, mask)
            continue
        attn_x = _rms_norm(block["attn_norm"], h, config.rms_norm_eps)
        result = _attention(
            block["attn"],
            attn_x,
            config,
            angles=angles,
            mask=mask,
            kv_cache=None if kv_caches is None else kv_caches[index],
            cache_index=cache_index,
            attention_bucket_size=attention_bucket_size,
            cache_dtype=cache_dtype,
        )
        if incremental:
            attn_out, cache = result
            new_caches.append(cache)
        else:
            attn_out = result
        residual_scale = jnp.asarray(config.residual_scale, dtype=attn_out.dtype)
        h += residual_scale * attn_out
        mlp_x = _rms_norm(block["mlp_norm"], h, config.rms_norm_eps)
        mlp = block["mlp"]
        h += residual_scale * (
            (jax.nn.silu(mlp_x @ mlp["w_gate"]) * (mlp_x @ mlp["w_up"]))
            @ mlp["w_down"]
        )
    output_embed = params.get("lm_head", params["embed"])
    if output_token_ids is not None:
        output_embed = output_embed[output_token_ids]
    logits = _rms_norm(params["final_norm"], h, config.rms_norm_eps) @ output_embed.T
    logits = logits * jnp.asarray(config.unembedding_scale, dtype=logits.dtype)
    return (logits, new_caches) if incremental else logits

