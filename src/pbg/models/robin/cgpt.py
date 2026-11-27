"""CGPT inference of ROBIN, with the parameter layout of a converted ROBIN checkpoint.

    Preserve float32 cache allocation, RMSNorm epsilon, GELU, biased SwiGLU,
    and cross-attention behavior used by the converted ROBIN sampler.
"""

import math

import jax
import jax.numpy as jnp


def sinusoidal_embed(data, dim):
    position = jnp.expand_dims(data.astype(jnp.float32), -1)
    angles = position * jnp.exp(jnp.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    out = jnp.zeros((*data.shape, dim))
    out = out.at[..., 0::2].set(jnp.sin(angles))
    return out.at[..., 1::2].set(jnp.cos(angles))


def rms_norm(params, x):
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6) * params['scale']


def apply_model(params, config, ids, encodings, mask=None, *, kv_cache=None, cache_index=None):
    """Return logits; in cached mode also return the updated self-attention cache."""
    batch, length = ids.shape
    dim, heads = config['dim'], config['num_heads']
    head_dim = dim // heads
    offset = 0 if cache_index is None else cache_index
    positions = jnp.arange(length) + offset
    h = jnp.take(params['token_embed'], ids, axis=0)
    h += jax.lax.dynamic_slice(params['pos_embed'], (offset, 0), (length, dim))
    h += jnp.take(params['coord_dim_embed'], positions % 3, axis=0)
    h += jnp.take(params['atom_pos_embed'], positions // 3, axis=0)

    ce = params['cond_embed']
    atoms = encodings['atom_type'].shape[1]
    cond = jnp.concatenate((
        jnp.take(ce['atom_embed'], encodings['atom_type'], axis=0),
        jnp.take(ce['residue_embed'], encodings['aa_type'], axis=0),
        sinusoidal_embed(encodings['aa_pos'], dim),
        jnp.broadcast_to(sinusoidal_embed(encodings['seq_len'], dim), (batch, atoms, dim)),
        jnp.broadcast_to(sinusoidal_embed(jnp.arange(atoms)[None, :], dim), (batch, atoms, dim)),
    ), axis=-1)
    cond = jnp.dot(cond, ce['proj']['w1']) + ce['proj']['b1']
    cond = jnp.dot(jax.nn.gelu(cond), ce['proj']['w2']) + ce['proj']['b2']
    expanded = jnp.repeat(cond, 3, axis=1)
    if length > atoms * 3:
        h += jnp.pad(expanded, ((0, 0), (0, length - atoms * 3), (0, 0)))
    else:
        h += jax.lax.dynamic_slice(expanded, (0, offset, 0), (batch, length, dim))

    # The historical predict loop recomputes conditioning K/V on each step.
    ca = params['global_cond_attn']
    q = (h @ ca['q_w'] + ca.get('q_b', 0)).reshape(batch, length, heads, head_dim)
    kv = (cond @ ca['kv_w'] + ca.get('kv_b', 0)).reshape(batch, atoms, 2, heads, head_dim)
    cross_mask = None if mask is None else mask[:, None, None, :].astype(bool)
    cross = jax.nn.dot_product_attention(q, kv[:, :, 0], kv[:, :, 1], mask=cross_mask, scale=head_dim**-0.5)
    h += cross.reshape(batch, length, dim) @ ca['proj_w'] + ca.get('proj_b', 0)

    attn_mask = None
    if mask is not None:
        coord_mask = jnp.repeat(mask, 3, axis=1)
        if length > coord_mask.shape[1]:
            coord_mask = jnp.pad(coord_mask, ((0, 0), (0, length - coord_mask.shape[1])))
        attn_mask = coord_mask[:, None, None, :length].astype(bool)
    if (kv_cache is None) != (cache_index is None):
        raise ValueError('Set both kv_cache and cache_index, or neither')
    caches = []
    for index, block in enumerate(params['blocks']):
        x = rms_norm(block['norm1'], h)
        attn = block['attn']
        qkv = (x @ attn['qkv_w'] + attn.get('qkv_b', 0)).reshape(batch, length, 3, heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if kv_cache is not None:
            ck, cv = kv_cache[index]
            ck = ck.at[:, positions].set(k)
            cv = cv.at[:, positions].set(v)
            keys = jnp.arange(ck.shape[1])
            causal = (positions[:, None] >= keys[None, :]) & (keys[None, :] < offset + length)
            causal = causal[None, None, :, :]
            if attn_mask is not None:
                causal &= attn_mask
            out = jax.nn.dot_product_attention(q, ck, cv, mask=causal, is_causal=False, scale=head_dim**-0.5)
            caches.append((ck, cv))
        else:
            out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask, is_causal=True, scale=head_dim**-0.5)
        h += out.reshape(batch, length, dim) @ attn['proj_w'] + attn.get('proj_b', 0)
        ff = block['ff']
        x = rms_norm(block['norm2'], h)
        h += (jax.nn.silu(x @ ff['w1']) * (x @ ff['w2'])) @ ff['w3'] + ff['b3']
    logits = rms_norm(params['final_norm'], h) @ params['output_proj_w']
    return (logits, caches) if kv_cache is not None else logits
