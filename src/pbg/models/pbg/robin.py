"""ROBIN atom-wise conditioning on the PBG trunk, for the prefix ablation.

This reproduces the conditioning of the ROBIN v14 checkpoint that the paper
evaluates (transferable-samplers ``scripts/alex/h100_multi_v14.sh`` with
``add_cond_to_input``, ``use_global_cond`` and ``cond_embed.use_atom_pos_embed``
all True). ``pbg.models.robin.cgpt`` has the same structure:

* Five features per atom (atom type, residue type, residue position, peptide
  length and atom index) pass through a GELU MLP to one vector per atom.
* Each input position receives the vector of the atom whose coordinate it
  predicts.
* One non-causal attention layer, applied once before the transformer blocks,
  lets every position read all atoms of its own peptide.

The result enters the trunk as the per-token ``bias`` of ``apply_qwen``. The
trunk, optimizer, loss and data pipeline therefore stay the same as for prefix
training. The weights live under ``qwen['atom_conditioner']``. They are an
adapter, and ``planning.non_embedding_params`` does not count them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pbg.models.pbg.config import PBGConfig

FEATURES = ("atom_type", "aa_type", "aa_pos", "seq_len", "atom_pos")
# ROBIN's table sizes: 54 atom names and 20 residues, plus padding rows.
NUM_ATOM_TYPES = 64
NUM_RESIDUE_TYPES = 32

# The ROBIN helpers are imported inside the functions below: the
# ``pbg.models.robin`` package imports the PBG sampler, which imports
# this module through ``pbg.models.pbg.model``.


def atom_features(sequence: str | None, atomic_numbers: np.ndarray) -> np.ndarray:
    """Return ROBIN's int32 ``(n_atoms, 5)`` features in the frame's atom order.

    The features come from the shared ff14SB templates of the ROBIN port. A
    frame whose atom count or elements differ from the template order is
    refused, because its features would describe the wrong atoms.
    """
    from pbg.models.robin.conditioning import ATOM_NAMES, encode_sequences

    if not sequence:
        raise ValueError("atom conditioning requires a peptide sequence")
    z = np.asarray(atomic_numbers).astype(np.int64)
    encodings, mask = encode_sequences([sequence], int(z.shape[0]))
    if int(mask.sum()) != z.shape[0]:
        raise ValueError(
            f"{sequence}: template has {int(mask.sum())} atoms, frame has {z.shape[0]}"
        )
    atom_type = encodings["atom_type"][0]
    known = atom_type > 0
    # Atomic number of each ROBIN atom type; type 0 is an unknown name.
    elements = np.asarray(
        [0] + [{"C": 6, "H": 1, "N": 7, "O": 8, "S": 16}[name[0]] for name in ATOM_NAMES]
    )
    if np.any(elements[atom_type[known]] != z[known]):
        raise ValueError(f"{sequence}: frame atoms are not in ff14SB template order")
    seq_len = np.full_like(atom_type, encodings["seq_len"][0, 0])
    return np.stack(
        (atom_type, encodings["aa_type"][0], encodings["aa_pos"][0], seq_len,
         encodings["atom_pos"][0]),
        axis=1,
    ).astype(np.int32)


def init_atom_conditioner(
    key: Array, config: PBGConfig, std: float = 0.02
) -> dict[str, Array]:
    """Initialise the conditioner. Its tables use the trunk's embedding scale."""
    dim = config.dim
    if dim % config.num_heads:
        raise ValueError("atom conditioning needs dim divisible by num_heads")
    keys = jax.random.split(key, 7)
    params = {
        "atom_embed": jax.random.normal(keys[0], (NUM_ATOM_TYPES, dim)) * std,
        "residue_embed": jax.random.normal(keys[1], (NUM_RESIDUE_TYPES, dim)) * std,
    }
    shapes = {
        "mlp_in": (len(FEATURES) * dim, dim),
        "mlp_out": (dim, dim),
        "attn_q": (dim, dim),
        "attn_kv": (dim, 2 * dim),
        "attn_out": (dim, dim),
    }
    for subkey, (name, shape) in zip(keys[2:], shapes.items()):
        params[name] = jax.random.normal(subkey, shape) / jnp.sqrt(shape[0])
        params[f"{name}_bias"] = jnp.zeros((shape[1],))
    return params


def encode_atoms(
    params: dict[str, Array], config: PBGConfig, features: Array, dtype
) -> tuple[Array, Array]:
    """Return the per-atom vectors and the keys and values of their attention.

    The result depends only on the atom features. Decoding computes it once per
    batch and passes it to every ``atom_conditioning_bias`` step.
    """
    from pbg.models.robin.cgpt import sinusoidal_embed

    dim, heads = config.dim, config.num_heads
    features = features.astype(jnp.int32)
    x = jnp.concatenate(
        [params["atom_embed"][features[..., 0]], params["residue_embed"][features[..., 1]]]
        + [sinusoidal_embed(features[..., index], dim).astype(dtype) for index in (2, 3, 4)],
        axis=-1,
    )
    hidden = jax.nn.gelu(x @ params["mlp_in"] + params["mlp_in_bias"], approximate=False)
    cond = hidden @ params["mlp_out"] + params["mlp_out_bias"]
    kv = (cond @ params["attn_kv"] + params["attn_kv_bias"]).reshape(
        *cond.shape[:2], 2, heads, dim // heads
    )
    return cond, kv


def atom_conditioning_bias(
    params: dict[str, Array],
    config: PBGConfig,
    embed: Array,
    batch: dict[str, Array],
    bias: Array,
    encoded: tuple[Array, Array] | None = None,
) -> Array:
    """Return the trunk's input ``bias`` plus additive and global conditioning.

    ``bias`` is the per-token bias that the trunk already adds (the level
    embedding). As in ROBIN, the global attention reads the embedded input:
    token embedding, ``bias`` and the additive conditioning. Collation supplies
    ``atom_features`` and ``atom_sample`` per atom slot and ``atom_slot`` per
    position (the slot of the predicted atom, or -1). ``encoded`` is the result
    of ``encode_atoms``; without it, the atoms are encoded from
    ``batch["atom_features"]``.
    """
    dim, heads = config.dim, config.num_heads
    head_dim = dim // heads
    cond, kv = (
        encode_atoms(params, config, batch["atom_features"], embed.dtype)
        if encoded is None
        else encoded
    )

    slot = batch["atom_slot"].astype(jnp.int32)
    additive = jax.vmap(lambda row, index: row[index])(cond, jnp.maximum(slot, 0))
    additive = jnp.where((slot >= 0)[..., None], additive, 0)

    h = embed[batch["tokens"]] + bias + additive
    batch_size, length, _ = h.shape
    q = (h @ params["attn_q"] + params["attn_q_bias"]).reshape(
        batch_size, length, heads, head_dim
    )
    owner = batch["atom_sample"].astype(jnp.int32)
    visible = (owner[:, None, :] >= 0) & (
        batch["sample_id"].astype(jnp.int32)[:, :, None] == owner[:, None, :]
    )
    attended = jax.nn.dot_product_attention(
        q, kv[:, :, 0], kv[:, :, 1], mask=visible[:, None], scale=head_dim**-0.5
    )
    global_cond = attended.reshape(batch_size, length, dim) @ params["attn_out"]
    global_cond = jnp.where(
        visible.any(axis=-1)[..., None], global_cond + params["attn_out_bias"], 0
    )
    return bias + additive + global_cond
