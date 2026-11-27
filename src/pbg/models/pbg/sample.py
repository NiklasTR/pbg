"""KV-cached sampling and likelihood scoring of PBG conformers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, random

from pbg.core.frame import DataFrame
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.tokeniser import (
    VocabSchema,
    apply_tokeniser_conformer,
    undo_tokeniser_coordinate,
)
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.model import PBG
from pbg.models.pbg.robin import (
    atom_conditioning_bias,
    atom_features,
    encode_atoms,
)
from pbg.models.pbg.system import MolecularSystem, SamplingPolicy, prompt_arrays
from pbg.models.pbg.transformer import QwenParams, apply_qwen, qwen_vocab_size

_BFLOAT16_DTYPE = jax.dtypes.bfloat16


@partial(
    jax.jit,
    static_argnames=("config", "max_cache_len", "prompt_len", "coord_len"),
)
def decode_coordinates(
    qwen: QwenParams,
    config: PBGConfig,
    prompt: Array,
    angles: Array,
    level_bias: Array,
    key: Array,
    temperature: Array,
    decode_bias: Array,
    *,
    max_cache_len: int,
    prompt_len: int,
    coord_len: int,
    atom_batch: dict[str, Array] | None = None,
) -> tuple[Array, Array]:
    """Sample ``coord_len`` coordinate tokens after ``prompt``.

    ``atom_batch`` is required for ``conditioning='atom'``. It holds the
    collation arrays that ``atom_conditioning_bias`` reads: ``atom_features``
    and ``atom_sample`` per atom, ``atom_slot`` and ``sample_id`` per position.
    """
    # Joint checkpoints trained with BF16 angles before sin/cos. Preserve that
    # numerical convention; changing angle precision changes the learned model.
    if config.mixed_precision and (config.vocab_name or '').startswith('joint_'):
        angles = angles.astype(_BFLOAT16_DTYPE)
        level_bias = level_bias.astype(_BFLOAT16_DTYPE)
    if (config.conditioning == "atom") != (atom_batch is not None):
        raise ValueError("atom_batch must be given exactly for conditioning='atom'")
    conditioner = qwen["atom_conditioner"] if atom_batch is not None else None
    if conditioner is not None:
        # Training adds the conditioning to a bias in the trunk's dtype.
        level_bias = level_bias.astype(qwen["embed"].dtype)
        encoded = encode_atoms(
            conditioner, config, atom_batch["atom_features"], qwen["embed"].dtype
        )

    def condition(inputs: Array, start: int | Array, bias: Array) -> Array:
        if conditioner is None:
            return bias
        window = {
            "tokens": inputs,
            "atom_sample": atom_batch["atom_sample"],
            **{
                name: jax.lax.dynamic_slice_in_dim(
                    atom_batch[name], start, inputs.shape[1], axis=1
                )
                for name in ("atom_slot", "sample_id")
            },
        }
        return atom_conditioning_bias(
            conditioner, config, qwen["embed"], window, bias, encoded=encoded
        )

    inv_temp = 1.0 / jnp.maximum(temperature, 1e-6)

    def draw(logits: Array, k: Array) -> tuple[Array, Array, Array]:
        scaled = logits * inv_temp + decode_bias
        k, sub = random.split(k)
        sampled = random.categorical(sub, scaled, axis=-1)
        greedy = jnp.argmax(scaled, axis=-1)
        token = jnp.where(temperature > 0.0, sampled, greedy)
        log_probs = jax.nn.log_softmax(scaled, axis=-1)
        token_lp = jnp.take_along_axis(log_probs, token[:, None], axis=-1).squeeze(-1)
        return token, token_lp, k

    # A float32 trunk (``mixed_precision=False``) attends in float32 as in
    # training; the bf16 cache would fail on a float32 query.
    cache_dtype = "float32" if qwen["embed"].dtype == jnp.float32 else "bfloat16"
    prefill, caches = apply_qwen(
        qwen,
        config,
        prompt,
        kv_caches=None,
        cache_index=0,
        max_cache_len=max_cache_len,
        angles=angles[:, :prompt_len],
        bias=condition(prompt, 0, level_bias[:, :prompt_len]),
        cache_dtype=cache_dtype,
    )
    last_logits = prefill[:, -1, :]
    batch_size = prompt.shape[0]
    tokens = jnp.zeros((batch_size, coord_len), dtype=jnp.int32)
    token_log_probs = jnp.zeros((batch_size, coord_len), dtype=jnp.float32)

    def step(i: int, state):
        tokens, caches, last_logits, k, token_log_probs = state
        token, token_lp, k = draw(last_logits, k)
        tokens = tokens.at[:, i].set(token)
        token_log_probs = token_log_probs.at[:, i].set(token_lp)
        pos = prompt_len + i
        step_angles = jax.lax.dynamic_slice_in_dim(angles, pos, 1, axis=1)
        step_bias = jax.lax.dynamic_slice_in_dim(level_bias, pos, 1, axis=1)
        logits, caches = apply_qwen(
            qwen,
            config,
            token[:, None],
            kv_caches=caches,
            cache_index=pos,
            max_cache_len=max_cache_len,
            angles=step_angles,
            bias=condition(token[:, None], pos, step_bias),
            cache_dtype=cache_dtype,
        )
        return tokens, caches, logits[:, 0, :], k, token_log_probs

    tokens, _, _, _, token_log_probs = jax.lax.fori_loop(
        0, coord_len, step, (tokens, caches, last_logits, key, token_log_probs)
    )
    return tokens, jnp.concatenate(
        [token_log_probs, jnp.zeros((batch_size, 1), dtype=jnp.float32)], axis=1
    )


def inference_model(model: PBG) -> PBG:
    cfg = replace(model.config, augment=False, translate=False)
    return PBG(qwen=model.qwen, config=cfg, vocab=model.vocab)


def _coordinate_decode_bias(model: PBG) -> np.ndarray:
    return VocabSchema.of(model.vocab).decode_bias(
        qwen_vocab_size(model.qwen), allow_terminator=False
    )


def score_frames(
    model: PBG,
    frames: list[DataFrame] | tuple[DataFrame, ...],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    model = inference_model(model)
    if SamplingPolicy.from_model(model).generation == 'joint':
        # Density is evaluated at the supplied grid point, not at a centred or
        # otherwise augmented conformer. Replay the same cached decoder as draw.
        model = replace(model, config=replace(model.config, center=False))
        qwen = model.qwen
        if model.config.mixed_precision:
            qwen = jax.tree.map(lambda x: jnp.asarray(x, _BFLOAT16_DTYPE), qwen)
        scores = []
        for start in range(0, len(frames), batch_size):
            chunk = list(frames[start:start + batch_size])
            first = chunk[0]
            system = MolecularSystem(id=first.id or 'score', mapped_smiles=first.smiles,
                helm=first.helm, environment=first.environment, temperature_K=first.temperature,
                sequence=first.sequence)
            if any((f.smiles, f.helm, f.environment, f.temperature) !=
                   (first.smiles, first.helm, first.environment, first.temperature) for f in chunk):
                scores.extend(score_frames(model, [frame], batch_size=1) for frame in chunk)
                continue
            SamplingPolicy.from_model(model).system(system)
            prompt, atom, level, plen, clen, _ = prompt_arrays(model, system, len(chunk))
            batch = model.collate([[frame] for frame in chunk])
            np.testing.assert_array_equal(batch['tokens'][:, :plen], prompt)
            values = score_coordinate_tokens(qwen, model.config, jnp.asarray(prompt),
                jnp.asarray(batch['tokens'][:, plen:plen + clen]),
                model._atom_rope(jnp.asarray(atom)), model._level_bias(jnp.asarray(level)),
                jnp.asarray(_coordinate_decode_bias(model)),
                max_cache_len=model.config.max_seq_len, prompt_len=plen, coord_len=clen)
            scores.append(np.asarray(values))
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    scores: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        chunk = list(frames[start : start + batch_size])
        scores.append(np.asarray(jax.device_get(model.score(chunk))))
    return (
        np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)
    )


@partial(jax.jit, static_argnames=('config', 'max_cache_len', 'prompt_len', 'coord_len'))
def score_coordinate_tokens(qwen, config, prompt, tokens, angles, level_bias, decode_bias,
                            *, max_cache_len, prompt_len, coord_len):
    """Teacher-force the exact fixed-length, coordinate-only sampling density."""
    if config.mixed_precision and (config.vocab_name or '').startswith('joint_'):
        angles = angles.astype(_BFLOAT16_DTYPE)
        level_bias = level_bias.astype(_BFLOAT16_DTYPE)
    logits, caches = apply_qwen(qwen, config, prompt, cache_index=0,
        max_cache_len=max_cache_len, angles=angles[:, :prompt_len], bias=level_bias[:, :prompt_len])
    token_lp = jnp.zeros(tokens.shape, dtype=jnp.float32)

    def step(i, state):
        caches, logits, token_lp = state
        token = tokens[:, i]
        lp = jax.nn.log_softmax(logits + decode_bias, axis=-1)
        token_lp = token_lp.at[:, i].set(jnp.take_along_axis(lp, token[:, None], axis=-1)[:, 0])
        pos = prompt_len + i
        next_logits, caches = apply_qwen(qwen, config, token[:, None], kv_caches=caches,
            cache_index=pos, max_cache_len=max_cache_len,
            angles=jax.lax.dynamic_slice_in_dim(angles, pos, 1, axis=1),
            bias=jax.lax.dynamic_slice_in_dim(level_bias, pos, 1, axis=1))
        return caches, next_logits[:, 0], token_lp

    _, _, token_lp = jax.lax.fori_loop(0, coord_len, step, (caches, logits[:, -1], token_lp))
    return token_lp.sum(axis=1)


@dataclass(frozen=True, slots=True)
class SampleBatch:
    samples: np.ndarray
    logq: np.ndarray
    behavior_token_log_probs: np.ndarray | None
    behavior_token_ids: np.ndarray | None


def sample_molecular_conformers(
    model: PBG,
    system: MolecularSystem,
    key: Array,
    *,
    num_samples: int,
    batch_size: int = 64,
    proposal_temperature: float = 1.0,
    model_generation: str = 'auto',
) -> SampleBatch:
    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive")
    SamplingPolicy.from_model(model, model_generation).system(system)
    model = inference_model(model)
    qwen = model.qwen
    if model.config.mixed_precision:
        qwen = jax.tree.map(
            lambda value: (
                jax.lax.convert_element_type(value, _BFLOAT16_DTYPE)
                if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
                else value
            ),
            qwen,
        )
    samples, scalar_logq = [], []
    schema = VocabSchema.of(model.vocab)
    for start in range(0, num_samples, batch_size):
        n_batch = min(batch_size, num_samples - start)
        key, subkey = jax.random.split(key)
        prompt, atom_index, level_index, prompt_len, coord_len, z = prompt_arrays(
            model, system, n_batch
        )
        atom_batch = None
        if model.config.conditioning == "atom":
            # Mirror collation: the position before each coordinate token
            # predicts it, and every segment position belongs to sample 0.
            stop = prompt_len + coord_len
            atom_slot = np.full_like(atom_index, -1)
            atom_slot[:, prompt_len - 1 : stop - 1] = atom_index[:, prompt_len:stop] - 1
            sample_id = np.full_like(atom_index, -1)
            sample_id[:, : stop + 1] = 0
            features = atom_features(system.sequence, z)
            atom_batch = {
                "atom_features": jnp.asarray(
                    np.broadcast_to(features, (n_batch, *features.shape))
                ),
                "atom_sample": jnp.zeros((n_batch, features.shape[0]), dtype=jnp.int32),
                "atom_slot": jnp.asarray(atom_slot),
                "sample_id": jnp.asarray(sample_id),
            }
        tokens, token_lp = decode_coordinates(
            qwen,
            model.config,
            jnp.asarray(prompt),
            model._atom_rope(jnp.asarray(atom_index)),
            model._level_bias(jnp.asarray(level_index)),
            subkey,
            jnp.asarray(float(proposal_temperature), dtype=jnp.float32),
            jnp.asarray(_coordinate_decode_bias(model)),
            max_cache_len=model.config.max_seq_len,
            prompt_len=prompt_len,
            coord_len=coord_len,
            atom_batch=atom_batch,
        )
        token_ids = np.asarray(jax.device_get(tokens), dtype=np.int32)
        token_log_probs = np.asarray(jax.device_get(token_lp), dtype=np.float32)
        local_ids = token_ids - schema.coord_base_id
        depth = coord_len // int(z.size)
        geometry = local_ids.reshape(n_batch, z.size, depth)
        xyz = undo_tokeniser_coordinate(
            geometry,
            "bin",
            num_bins=model.config.bin_tokens,
            min_val=model.config.min_val,
            max_val=model.config.max_val,
            decode_method="center",
        )
        samples.append(
            np.concatenate(
                [np.broadcast_to(z[None, :, None], (n_batch, z.size, 1)), xyz], axis=2
            )
        )
        scalar_logq.append(np.sum(token_log_probs[:, :-1], axis=1))
    return SampleBatch(
        samples=np.concatenate(samples).astype(np.float64),
        logq=np.concatenate(scalar_logq).astype(np.float64),
        behavior_token_log_probs=None,
        behavior_token_ids=None,
    )


def sample_waters_given_solute(
    model: PBG,
    system: MolecularSystem,
    key: Array,
    *,
    solute_frames: np.ndarray,
    n_solute_atoms: int,
    solute_smiles: str,
    batch_size: int = 8,
    proposal_temperature: float = 1.0,
    model_generation: str = 'auto',
) -> SampleBatch:
    """Sample water coordinates conditioned on fixed solute coordinate tokens.

    Prefills discretised solute tokens after the SMILES/env prompt, then decodes
    only water-atom coordinate tokens. Returned samples keep the input solute
    block bit-identical and append generated solvent atoms.
    """
    frames = np.asarray(solute_frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[-1] != 4:
        raise ValueError(f"solute_frames must have shape (F, N, 4), got {frames.shape}")
    if int(frames.shape[1]) != int(n_solute_atoms):
        raise ValueError(
            f"solute_frames atom count {frames.shape[1]} != n_solute_atoms {n_solute_atoms}"
        )
    num_samples = int(frames.shape[0])
    if num_samples < 1:
        raise ValueError("solute_frames must be non-empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model = inference_model(model)
    cfg = model.config
    if cfg.atom_order != "source_v1":
        from pbg.core.topology import PromptTopology
        if not PromptTopology.compile(system.helm, "vacuum", cfg.atom_order).agrees_with(solute_smiles):
            raise ValueError("Fixed solute does not match the checkpoint's prompt-derived atom traversal")
    SamplingPolicy.from_model(model, model_generation).system(system)
    qwen = model.qwen
    if model.config.mixed_precision:
        qwen = jax.tree.map(
            lambda value: (
                jax.lax.convert_element_type(value, _BFLOAT16_DTYPE)
                if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
                else value
            ),
            qwen,
        )

    z = atom_numbers_from_smiles(system.mapped_smiles).astype(np.float32)
    n_atoms = int(z.size)
    if n_atoms <= n_solute_atoms:
        raise ValueError(
            f"system has {n_atoms} atoms; need water atoms beyond {n_solute_atoms} solute atoms"
        )
    n_water_atoms = n_atoms - int(n_solute_atoms)
    depth = cfg.geom_width
    solute_coord_len = int(n_solute_atoms) * depth
    water_coord_len = n_water_atoms * depth

    base_prompt, _atom_t, _level_t, smiles_prompt_len, full_coord_len, _z = prompt_arrays(
        model, system, 1
    )
    if full_coord_len != solute_coord_len + water_coord_len:
        raise ValueError(
            f"coord length mismatch: full={full_coord_len} "
            f"solute={solute_coord_len} water={water_coord_len}"
        )
    extended_prompt_len = smiles_prompt_len + solute_coord_len
    if extended_prompt_len + water_coord_len >= cfg.max_seq_len:
        raise ValueError("fixed-solute sample prompt exceeds config.max_seq_len")

    # Atom-major slot layout matches prompt_arrays for bin / inserting.
    atoms = np.repeat(np.arange(1, n_atoms + 1, dtype=np.int32), depth)
    levels = np.tile(np.arange(1, depth + 1, dtype=np.int32), n_atoms)

    samples: list[np.ndarray] = []
    scalar_logq: list[np.ndarray] = []
    schema = VocabSchema.of(model.vocab)
    for start in range(0, num_samples, batch_size):
        n_batch = min(batch_size, num_samples - start)
        key, subkey = jax.random.split(key)
        batch_solute = frames[start : start + n_batch]
        prompt = np.full((n_batch, extended_prompt_len), schema.bos_id, dtype=np.int32)
        prompt[:, :smiles_prompt_len] = base_prompt[0, :smiles_prompt_len]
        atom_index = np.zeros((n_batch, cfg.max_seq_len), dtype=np.int32)
        level_index = np.zeros_like(atom_index)
        atom_index[:, smiles_prompt_len:extended_prompt_len] = atoms[:solute_coord_len]
        level_index[:, smiles_prompt_len:extended_prompt_len] = levels[:solute_coord_len]
        atom_index[:, extended_prompt_len : extended_prompt_len + water_coord_len] = (
            atoms[solute_coord_len:]
        )
        level_index[:, extended_prompt_len : extended_prompt_len + water_coord_len] = (
            levels[solute_coord_len:]
        )
        for row in range(n_batch):
            # Discretise the fixed solute coordinates into decoder token ids.
            conformer = np.asarray(batch_solute[row], dtype=np.float64)
            if conformer.ndim != 2 or conformer.shape[1] != 4:
                raise ValueError(f"solute frames must have shape (N, 4), got {conformer.shape}")
            decoder_ids, _atom, _level = apply_tokeniser_conformer(
                DataFrame(id="fixed_solute", frame_id=0, smiles=solute_smiles, sequence=None,
                          helm="", environment="water", temperature=300.0, conformer=conformer),
                method="bin",
                num_bins=model.config.bin_tokens,
                min_val=model.config.min_val,
                max_val=model.config.max_val,
            )
            solute_tokens = np.asarray(decoder_ids, dtype=np.int32) + np.int32(schema.coord_base_id)
            if solute_tokens.size != solute_coord_len:
                raise ValueError(
                    f"solute token count {solute_tokens.size} != expected {solute_coord_len}"
                )
            prompt[row, smiles_prompt_len:extended_prompt_len] = solute_tokens

        tokens, token_lp = decode_coordinates(
            qwen,
            model.config,
            jnp.asarray(prompt),
            model._atom_rope(jnp.asarray(atom_index)),
            model._level_bias(jnp.asarray(level_index)),
            subkey,
            jnp.asarray(float(proposal_temperature), dtype=jnp.float32),
            jnp.asarray(_coordinate_decode_bias(model)),
            max_cache_len=model.config.max_seq_len,
            prompt_len=extended_prompt_len,
            coord_len=water_coord_len,
        )
        token_ids = np.asarray(jax.device_get(tokens), dtype=np.int32)
        token_log_probs = np.asarray(jax.device_get(token_lp), dtype=np.float32)
        local_ids = token_ids - schema.coord_base_id
        geometry = local_ids.reshape(n_batch, n_water_atoms, depth)
        water_xyz = undo_tokeniser_coordinate(
            geometry,
            "bin",
            num_bins=model.config.bin_tokens,
            min_val=model.config.min_val,
            max_val=model.config.max_val,
            decode_method="center",
        )
        water_z = z[n_solute_atoms:]
        water_block = np.concatenate(
            [
                np.broadcast_to(water_z[None, :, None], (n_batch, n_water_atoms, 1)),
                water_xyz,
            ],
            axis=2,
        )
        full = np.concatenate([batch_solute, water_block], axis=1)
        if not np.array_equal(full[:, :n_solute_atoms, :], batch_solute):
            raise RuntimeError("solute block changed after water sampling")
        samples.append(full.astype(np.float64))
        scalar_logq.append(np.sum(token_log_probs[:, :-1], axis=1))

    return SampleBatch(
        samples=np.concatenate(samples).astype(np.float64),
        logq=np.concatenate(scalar_logq).astype(np.float64),
        behavior_token_log_probs=None,
        behavior_token_ids=None,
    )
